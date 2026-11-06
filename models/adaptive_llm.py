import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from layers.attention import AttentionLayer, FullAttention
from layers.decoder import Decoder, DecoderLayer
from layers.embedding import DataEmbedding
from models.components import (
    CovariateBranch,
    TargetPeriodDecomp,
)
from utils.numerical_safety import require_finite


def _fp32_context(tensor):
    return torch.autocast(device_type=tensor.device.type, enabled=False)


class MemoryEfficientCrossAttentionLayer(nn.Module):
    """Cross-attention backed by PyTorch SDPA without materializing scores in Python."""

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        if d_keys < 1:
            raise ValueError("d_keys must be positive")
        if d_llm is None:
            raise ValueError("d_llm must be provided")

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = int(n_heads)
        self.attention_dropout = float(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        bsz, target_len, _ = target_embedding.shape
        source_len, _ = source_embedding.shape
        heads = self.n_heads

        query = self.query_projection(target_embedding)
        query = query.view(bsz, target_len, heads, -1).transpose(1, 2)

        key = self.key_projection(source_embedding)
        key = key.view(source_len, heads, -1).permute(1, 0, 2).unsqueeze(0)
        key = key.expand(bsz, -1, -1, -1)

        value = self.value_projection(value_embedding)
        value = value.view(source_len, heads, -1).permute(1, 0, 2).unsqueeze(0)
        value = value.expand(bsz, -1, -1, -1)

        dropout_p = self.attention_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, target_len, -1)
        return self.out_projection(out)


class AdaptivePromptGenerator(nn.Module):
    """Create balanced V2 soft prompts conditioned on each input sample."""

    def __init__(
        self,
        context_dim,
        stats_dim,
        prompt_hidden,
        llm_dim,
        n_heads,
        d_keys,
        prompt_len,
        dropout,
        alpha_init,
    ):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.prompt_hidden = int(prompt_hidden)
        self.llm_dim = int(llm_dim)

        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, prompt_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stats_projection = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, prompt_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.prompt_slots = nn.Parameter(torch.empty(1, prompt_len, prompt_hidden))
        nn.init.normal_(self.prompt_slots, mean=0.0, std=0.02)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=prompt_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(prompt_hidden)

        self.vocab_attention = MemoryEfficientCrossAttentionLayer(
            d_model=prompt_hidden,
            n_heads=n_heads,
            d_keys=d_keys,
            d_llm=llm_dim,
            attention_dropout=dropout,
        )
        self.prompt_base = nn.Parameter(torch.empty(1, prompt_len, llm_dim))
        nn.init.normal_(self.prompt_base, mean=0.0, std=0.02)

        # Normalize the global and vocabulary branches independently before
        # applying the gate. A small epsilon matters because prompt_base starts
        # at a much smaller scale than a reprogrammed vocabulary embedding.
        self.base_norm = nn.LayerNorm(llm_dim, eps=1e-8)
        self.vocab_norm = nn.LayerNorm(llm_dim, eps=1e-8)

        alpha = torch.tensor(float(alpha_init), dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        self.prompt_alpha = nn.Parameter(torch.logit(alpha))
        self.vocab_dropout = nn.Dropout(dropout)
        self.last_prompt_diversity_loss = None

    @staticmethod
    def _off_diagonal(values):
        prompt_len = values.shape[-1]
        mask = ~torch.eye(prompt_len, dtype=torch.bool, device=values.device)
        return values[:, mask]

    def _update_diversity_loss(self, prompt):
        if self.prompt_len > 1:
            with torch.autocast(device_type=prompt.device.type, enabled=False):
                normalized = F.normalize(prompt.float(), dim=-1)
                similarities = torch.matmul(normalized, normalized.transpose(1, 2))
                off_diagonal = self._off_diagonal(similarities)
                self.last_prompt_diversity_loss = off_diagonal.square().mean()
        else:
            zero = prompt.sum() * 0.0
            self.last_prompt_diversity_loss = zero

    def forward(
        self,
        context,
        stats,
        source_embeddings,
        reference_tokens,
    ):
        bsz = reference_tokens.shape[0]
        base = self.base_norm(self.prompt_base.expand(bsz, -1, -1))
        sequence_memory = self.context_projection(context)
        stats_memory = self.stats_projection(stats).unsqueeze(1)
        context_memory = torch.cat([stats_memory, sequence_memory], dim=1)
        slots = self.prompt_slots.expand(bsz, -1, -1)
        context_delta, _ = self.context_attention(
            slots, context_memory, context_memory, need_weights=False
        )
        prompt_query = self.context_norm(slots + context_delta)
        vocab_prompt = self.vocab_attention(prompt_query, source_embeddings, source_embeddings)
        vocab = self.vocab_norm(self.vocab_dropout(vocab_prompt))
        raw_prompt = base + torch.sigmoid(self.prompt_alpha) * vocab

        # Match each sample's prompt scale to its time-token scale. The detached
        # reference prevents the model from shrinking/growing the residual branch
        # merely to manipulate the prompt regularizer.
        prompt_direction = F.normalize(raw_prompt.float(), dim=-1).to(raw_prompt.dtype)
        reference_norm = reference_tokens.detach().float().norm(dim=-1).mean(dim=1).clamp_min(1e-6)
        prompt = prompt_direction * reference_norm.to(raw_prompt.dtype)[:, None, None]
        prompt = prompt.to(dtype=reference_tokens.dtype)
        self._update_diversity_loss(prompt)
        return prompt


class FrozenGPT2Encoder(nn.Module):
    """Residual reprogramming and Prompt V2; legacy state-dict names retained."""

    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        path = configs.llm_model_path
        self.gpt2_config = GPT2Config.from_pretrained(path, local_files_only=True)
        self.gpt2_config.num_hidden_layers = configs.llm_layers
        self.gpt2_config.output_attentions = False
        self.gpt2_config.output_hidden_states = False
        self.gpt2_config.use_cache = False
        if self.gpt2_config.n_embd != configs.llm_dim:
            raise ValueError("llm_dim must match the local GPT-2 embedding dimension")
        if configs.seq_len + configs.adaptive_prompt_len > self.gpt2_config.n_positions:
            raise ValueError("seq_len + adaptive_prompt_len exceeds GPT-2 positional capacity")
        self.llm_model = GPT2Model.from_pretrained(
            path,
            local_files_only=True,
            config=self.gpt2_config,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        self.llm_model.requires_grad_(False)
        self.llm_model.eval()
        self.input_embeddings = self.llm_model.get_input_embeddings()
        vocabulary_size = self.input_embeddings.weight.shape[0]
        # GPT-2 uses 50,257 entries; alternate large-vocabulary backbones are not supported.
        if vocabulary_size > 100000:
            raise ValueError("This MVP requires a GPT-2-sized vocabulary")
        self.word_projection = nn.Linear(vocabulary_size, min(3000, vocabulary_size))
        self.crossattention_layer = MemoryEfficientCrossAttentionLayer(
            d_model=configs.c_out,
            n_heads=configs.n_heads,
            d_keys=configs.d_ff,
            d_llm=configs.llm_dim,
            attention_dropout=configs.dropout,
        )
        self.adaptive_prompt_generator = AdaptivePromptGenerator(
            context_dim=3 * configs.c_out + configs.d_model,
            stats_dim=5 * configs.c_out,
            prompt_hidden=configs.adaptive_prompt_hidden,
            llm_dim=configs.llm_dim,
            n_heads=configs.n_heads,
            d_keys=configs.d_ff,
            prompt_len=configs.adaptive_prompt_len,
            dropout=configs.adaptive_prompt_dropout,
            alpha_init=configs.adaptive_prompt_alpha_init,
        )
        self.prompt_len = configs.adaptive_prompt_len

    def prompt_diversity_loss(self, reference_tensor=None):
        value = self.adaptive_prompt_generator.last_prompt_diversity_loss
        if value is not None:
            return value
        if reference_tensor is not None:
            return reference_tensor.sum() * 0.0
        return torch.zeros((), device=self.device)

    @staticmethod
    def _positions(batch_size, length, device):
        return torch.arange(length, dtype=torch.long, device=device)[None, :].expand(batch_size, -1)

    def forward(self, residual, context, stats):
        # Explicitly disable any outer autocast: the MVP supports FP32 only.
        with _fp32_context(residual):
            residual = residual.float()
            embeddings = self.input_embeddings.weight
            prototypes = self.word_projection(embeddings.permute(1, 0)).permute(1, 0)
            require_finite("LLM shared prototype memory", prototypes)
            tokens = self.crossattention_layer(residual, prototypes, prototypes)
            prompt = self.adaptive_prompt_generator(
                context.float(), stats.float(), prototypes, tokens
            )
            inputs = torch.cat([prompt, tokens], dim=1)
            batch_size = residual.shape[0]
            # Prompt and time positions both start at zero, as in the reference.
            positions = torch.cat(
                [
                    self._positions(batch_size, self.prompt_len, residual.device),
                    self._positions(batch_size, tokens.shape[1], residual.device),
                ],
                dim=1,
            )
            require_finite("frozen GPT2 input", inputs)
            self.llm_model.eval()
            output = self.llm_model(
                inputs_embeds=inputs,
                position_ids=positions,
                attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        return output[:, -tokens.shape[1] :, :]


class AdaptiveLLM(nn.Module):
    """Full periodic-residual model; no experimental replacement branches."""

    def __init__(self, configs):
        super().__init__()
        if configs.enc_in <= configs.c_out:
            raise ValueError("AdaptiveLLM requires at least one auxiliary covariate")
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        # Attribute names deliberately preserve existing checkpoint keys.
        self.covariate_branch = CovariateBranch(configs, configs.enc_in - self.c_out)
        self.LLM_encoder = FrozenGPT2Encoder(configs)
        self.target_decomp = TargetPeriodDecomp(configs.top_k)
        self.period_lstm = nn.LSTM(
            input_size=configs.c_out,
            hidden_size=configs.period_lstm_dim,
            num_layers=configs.period_lstm_layers,
            batch_first=True,
            dropout=configs.period_lstm_dropout if configs.period_lstm_layers > 1 else 0.0,
        )
        self.period_to_llm = nn.Linear(configs.period_lstm_dim, configs.llm_dim)
        initial_gate = torch.tensor(configs.period_lstm_alpha_init, dtype=torch.float32)
        self.period_alpha = nn.Parameter(torch.logit(initial_gate.clamp(1e-4, 1 - 1e-4)))
        self.target_fusion_norm = nn.LayerNorm(configs.llm_dim)
        self.dec_embedding = DataEmbedding(
            configs.llm_dim, configs.d_model, "timeF", "t", configs.dropout
        )
        self.selfattention_layer = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    4 * configs.d_model,
                    dropout=configs.dropout,
                    activation="gelu",
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.out_projection = nn.Linear(configs.d_model, configs.c_out)
        self.linear_predict = nn.Linear(configs.seq_len, configs.pred_len + configs.label_len)

    def prompt_diversity_loss(self, reference_tensor=None):
        return self.LLM_encoder.prompt_diversity_loss(reference_tensor)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        auxiliary = x_enc[:, :, : -self.c_out]
        target = x_enc[:, :, -self.c_out :]
        target_mean = target.mean(1, keepdim=True).detach()
        centered = target - target_mean
        target_std = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        normalized = centered / target_std
        with _fp32_context(x_enc):
            periodic, residual = self.target_decomp(normalized.float())
        covariates = self.covariate_branch(auxiliary, x_mark_enc)
        context = torch.cat([normalized, periodic, residual, covariates], dim=-1)
        target_energy = normalized.square().mean(dim=1).clamp_min(1e-6)
        stats = torch.cat(
            [
                target_mean.squeeze(1),
                target_std.squeeze(1),
                normalized[:, -1, :],
                normalized[:, -1, :] - normalized[:, 0, :],
                periodic.square().mean(dim=1) / target_energy,
            ],
            dim=-1,
        )
        residual_hidden = self.LLM_encoder(residual, context, stats)
        periodic_hidden, _ = self.period_lstm(periodic)
        periodic_hidden = self.period_to_llm(periodic_hidden)
        fused = self.target_fusion_norm(
            residual_hidden + torch.sigmoid(self.period_alpha) * periodic_hidden
        )
        decoder_input = self.linear_predict(fused.permute(0, 2, 1)).permute(0, 2, 1)
        decoder_input = self.dec_embedding(decoder_input, x_mark_dec)
        decoded = self.selfattention_layer(decoder_input, covariates)
        return self.out_projection(decoded) * target_std + target_mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Keep the external model interface; x_dec/mask are unused compatibility inputs."""
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len :, :]
