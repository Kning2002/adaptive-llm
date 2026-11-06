"""Only the current AdaptiveLLM Full model is exposed."""

from models.adaptive_llm import AdaptiveLLM

# The package-level alias keeps existing `from models import Model` callers working.
Model = AdaptiveLLM

__all__ = ["AdaptiveLLM", "Model"]
