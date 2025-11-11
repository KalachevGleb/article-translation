"""Translation engine for document sections."""

from typing import Dict, List, Optional

from .models import Document, Section
from .openai_client import OpenAIClient


class TranslationEngine:
    """Handles translation of document sections."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        source_language: str = "russian",
        target_language: str = "english",
        max_retries: int = 2,
    ):
        """Initialize translation engine.

        Args:
            openai_client: OpenAI client instance
            source_language: Source language name
            target_language: Target language name
            max_retries: Maximum translation retry attempts
        """
        self.client = openai_client
        self.source_language = source_language
        self.target_language = target_language
        self.max_retries = max_retries

    def translate_section(
        self,
        section: Section,
        dictionary: Dict[str, str],
        dependency_translations: Dict[str, str],
        strict_formulas: bool = False,
    ) -> str:
        """Translate a single section.

        Args:
            section: Section to translate
            dictionary: Terminology dictionary
            dependency_translations: Translations of dependent sections
            strict_formulas: Whether to use strict formula preservation mode

        Returns:
            Translated section content
        """
        # Build context from dependencies
        context = self._build_context(dependency_translations)

        # Build prompt
        prompt = self._build_translation_prompt(
            section.content,
            dictionary,
            context,
            strict_formulas,
        )

        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # Perform translation
        translation = self.client.chat_completion(messages)

        return translation.strip()

    def _get_system_prompt(self) -> str:
        """Get system prompt for translation."""
        return f"""You are a professional scientific translator from {self.source_language} to {self.target_language}.

Your key responsibilities:
1. NEVER modify LaTeX formulas (in $...$ or $$...$$, \\[...\\], equation, align, etc.)
2. Translate text naturally and idiomatically
3. Restructure sentences as needed for natural {self.target_language}
4. Use provided terminology dictionary consistently
5. Maintain LaTeX structure and commands
"""

    def _build_translation_prompt(
        self,
        content: str,
        dictionary: Dict[str, str],
        context: str,
        strict_formulas: bool,
    ) -> str:
        """Build translation prompt."""
        # Format dictionary
        dict_text = "\n".join([
            f"- {src} → {tgt}"
            for src, tgt in dictionary.items()
        ])

        formula_instruction = ""
        if strict_formulas:
            formula_instruction = """
⚠️ CRITICAL: FORMULA PRESERVATION ⚠️
All formulas MUST remain IDENTICAL to the source.
Check EVERY formula TWICE before outputting.
This is a retry due to formula mismatches in the previous attempt.
"""

        prompt = f"""Translate the following scientific text from {self.source_language} to {self.target_language}.

TERMINOLOGY DICTIONARY (use these translations):
{dict_text if dict_text else "(no specific terms)"}

CONTEXT FROM PREVIOUS SECTIONS:
{context if context else "(no dependencies)"}

CRITICAL RULES:
1. DO NOT MODIFY any LaTeX formulas
2. Keep all LaTeX commands and environments intact
3. Translate text naturally and idiomatically
4. Use terminology dictionary consistently
5. Restructure sentences for natural {self.target_language} when needed
{formula_instruction}

TEXT TO TRANSLATE:
{content}

Provide ONLY the translated text, without explanations or comments.
"""
        return prompt

    def _build_context(self, dependency_translations: Dict[str, str]) -> str:
        """Build context string from dependency translations."""
        if not dependency_translations:
            return ""

        context_parts = []
        for section_id, translation in dependency_translations.items():
            # Take first 500 characters as context
            preview = translation[:500]
            if len(translation) > 500:
                preview += "..."
            context_parts.append(f"[{section_id}]: {preview}")

        return "\n\n".join(context_parts)

    def translate_document(
        self,
        document: Document,
        sorted_sections: List[Section],
        dictionary: Dict[str, str],
    ) -> Document:
        """Translate entire document in dependency order.

        Args:
            document: Document to translate
            sorted_sections: Sections in topological order
            dictionary: Terminology dictionary

        Returns:
            Document with translations
        """
        translations = {}  # section_id -> translation

        for i, section in enumerate(sorted_sections, 1):
            print(f"Translating section {i}/{len(sorted_sections)}: {section.title}")

            # Get translations of dependencies
            dep_translations = {
                dep_id: translations[dep_id]
                for dep_id in section.dependencies
                if dep_id in translations
            }

            # Translate
            try:
                translation = self.translate_section(
                    section,
                    dictionary,
                    dep_translations,
                    strict_formulas=False,
                )

                section.translation = translation
                section.translation_attempts = 1
                translations[section.id] = translation

            except Exception as e:
                print(f"Error translating section {section.id}: {e}")
                section.translation = section.content  # Fallback to original
                translations[section.id] = section.content

        return document

    def retry_translation(
        self,
        section: Section,
        dictionary: Dict[str, str],
        dependency_translations: Dict[str, str],
    ) -> str:
        """Retry translation with stricter formula preservation.

        Args:
            section: Section to retry
            dictionary: Terminology dictionary
            dependency_translations: Dependency translations

        Returns:
            New translation attempt
        """
        print(f"  Retrying translation for section: {section.title}")

        translation = self.translate_section(
            section,
            dictionary,
            dependency_translations,
            strict_formulas=True,
        )

        section.translation_attempts += 1
        return translation
