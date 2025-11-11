"""Translation engine for document sections."""

from typing import Dict, List, Optional

from .models import Document, Section
from .openai_client import OpenAIClient
from .prompt_loader import PromptLoader


class TranslationEngine:
    """Handles translation of document sections."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        source_language: str = "russian",
        target_language: str = "english",
        max_retries: int = 2,
        prompt_loader: Optional[PromptLoader] = None,
    ):
        """Initialize translation engine.

        Args:
            openai_client: OpenAI client instance
            source_language: Source language name
            target_language: Target language name
            max_retries: Maximum translation retry attempts
            prompt_loader: Prompt loader instance (creates default if None)
        """
        self.client = openai_client
        self.source_language = source_language
        self.target_language = target_language
        self.max_retries = max_retries
        self.prompt_loader = prompt_loader or PromptLoader()

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
        # Load appropriate prompt config
        prompt_name = "translation_strict_formulas" if strict_formulas else "translation"
        prompt_config = self.prompt_loader.load(prompt_name)

        # Build context and dictionary strings
        context = self._build_context(dependency_translations)
        dict_text = self._format_dictionary(dictionary)

        # Format prompts
        system_prompt = prompt_config.format_system_prompt(
            source_language=self.source_language,
            target_language=self.target_language,
        )
        user_prompt = prompt_config.format_user_prompt(
            source_language=self.source_language,
            target_language=self.target_language,
            dictionary=dict_text,
            context=context,
            content=section.content,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Perform translation
        translation = self.client.chat_completion(
            messages,
            temperature=prompt_config.temperature,
            max_tokens=prompt_config.max_tokens,
        )

        return translation.strip()

    def _format_dictionary(self, dictionary: Dict[str, str]) -> str:
        """Format dictionary for prompt."""
        if not dictionary:
            return "(no specific terms)"

        return "\n".join([
            f"- {src} → {tgt}"
            for src, tgt in dictionary.items()
        ])

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

    def fix_cyrillic(
        self,
        text: str,
        marked_text: str,
        highlighted_fragments: List[str],
        dictionary: Dict[str, str],
    ) -> str:
        """Fix untranslated Cyrillic fragments.

        Args:
            text: Original text with Cyrillic
            marked_text: Text with marked Cyrillic (>>> <<<)
            highlighted_fragments: List of Cyrillic fragments
            dictionary: Terminology dictionary

        Returns:
            Text with Cyrillic fragments translated
        """
        print(f"  Fixing {len(highlighted_fragments)} Cyrillic fragment(s)")

        # Load prompt configuration
        prompt_config = self.prompt_loader.load("cyrillic_fix")

        # Format dictionary
        dict_text = self._format_dictionary(dictionary)

        # Format highlighted fragments
        fragments_text = "\n".join([
            f"{i+1}. >>>{frag}<<<"
            for i, frag in enumerate(highlighted_fragments)
        ])

        # Format prompts
        system_prompt = prompt_config.format_system_prompt(
            source_language=self.source_language,
            target_language=self.target_language,
        )
        user_prompt = prompt_config.format_user_prompt(
            source_language=self.source_language,
            target_language=self.target_language,
            dictionary=dict_text,
            highlighted_fragments=fragments_text,
            marked_text=marked_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Get fixed translation
        fixed_text = self.client.chat_completion(
            messages,
            temperature=prompt_config.temperature,
            max_tokens=prompt_config.max_tokens,
        )

        return fixed_text.strip()
