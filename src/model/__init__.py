"""
Model module for GRAM and GRAM-C
Contains FiD-T5 and related model implementations
"""

from .gram import GRAM
from .gram_c import GRAM_C
from .collaborative_adapter import CollaborativeAdapter


def create_model(model_type, config=None, **kwargs):
    """
    Factory function to create models

    Args:
        model_type (str): 'gram', 'gram_c' or other model types
        config: Model configuration
        **kwargs: Additional arguments

    Returns:
        Model instance
    """
    if model_type == "gram":
        return GRAM(config=config, **kwargs)
    elif model_type == "gram_c":
        return GRAM_C(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["GRAM", "GRAM_C", "CollaborativeAdapter", "create_model"]
