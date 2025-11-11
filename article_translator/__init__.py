"""Article Translation System.

Automated LaTeX article translation using LLM (OpenAI GPT-o3/o3-mini).
"""

from .main import ArticleTranslator
from .models import TranslationResult

__version__ = "0.1.0"
__all__ = ["ArticleTranslator", "TranslationResult"]
