"""Models module for legal summarization."""

from .gemini_client import GeminiClient
from .prompts import LegalPrompts

__all__ = ["GeminiClient", "LegalPrompts"]