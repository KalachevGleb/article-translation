"""Cyrillic text validator for translation quality checking."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .latex_parser import LaTeXParser


@dataclass
class CyrillicFragment:
    """Represents a fragment containing Cyrillic text."""
    text: str
    start_pos: int
    end_pos: int
    context: str  # Surrounding context


class CyrillicValidator:
    """Validates that no Cyrillic text remains in translation."""

    # Cyrillic Unicode ranges
    CYRILLIC_PATTERN = re.compile(r'[А-Яа-яЁё]+')

    def __init__(self, latex_parser: LaTeXParser):
        """Initialize validator.

        Args:
            latex_parser: LaTeX parser for formula extraction
        """
        self.parser = latex_parser

    def has_cyrillic(self, text: str, exclude_formulas: bool = True) -> bool:
        """Check if text contains Cyrillic characters.

        Args:
            text: Text to check
            exclude_formulas: If True, ignore Cyrillic in formulas

        Returns:
            True if Cyrillic found
        """
        if exclude_formulas:
            text = self._remove_formulas(text)

        return bool(self.CYRILLIC_PATTERN.search(text))

    def find_cyrillic_fragments(
        self,
        text: str,
        context_chars: int = 50,
        exclude_formulas: bool = True,
    ) -> List[CyrillicFragment]:
        """Find all Cyrillic fragments in text.

        Args:
            text: Text to search
            context_chars: Number of chars to include as context
            exclude_formulas: If True, ignore Cyrillic in formulas

        Returns:
            List of Cyrillic fragments with context
        """
        # Remove formulas if requested
        search_text = text
        if exclude_formulas:
            search_text = self._remove_formulas(text)

        fragments = []

        for match in self.CYRILLIC_PATTERN.finditer(search_text):
            start, end = match.span()

            # Get context
            context_start = max(0, start - context_chars)
            context_end = min(len(search_text), end + context_chars)
            context = search_text[context_start:context_end]

            fragments.append(CyrillicFragment(
                text=match.group(),
                start_pos=start,
                end_pos=end,
                context=context,
            ))

        return fragments

    def _remove_formulas(self, text: str) -> str:
        """Remove all LaTeX formulas from text.

        Args:
            text: Text to process

        Returns:
            Text with formulas replaced by spaces
        """
        # Remove display formulas
        for pattern in self.parser.DISPLAY_PATTERNS:
            text = re.sub(pattern, ' ', text, flags=re.DOTALL)

        # Remove inline formulas
        text = re.sub(self.parser.INLINE_PATTERN, ' ', text)

        return text

    def mark_cyrillic_fragments(
        self,
        text: str,
        marker_start: str = ">>>",
        marker_end: str = "<<<",
        exclude_formulas: bool = True,
    ) -> Tuple[str, int]:
        """Mark all Cyrillic fragments in text.

        Args:
            text: Text to mark
            marker_start: Start marker
            marker_end: End marker
            exclude_formulas: If True, ignore Cyrillic in formulas

        Returns:
            Tuple of (marked text, number of fragments)
        """
        fragments = self.find_cyrillic_fragments(text, exclude_formulas=exclude_formulas)

        if not fragments:
            return text, 0

        # Sort by position (reverse order to preserve positions)
        fragments.sort(key=lambda f: f.start_pos, reverse=True)

        marked_text = text
        for fragment in fragments:
            marked_text = (
                marked_text[:fragment.start_pos] +
                marker_start +
                marked_text[fragment.start_pos:fragment.end_pos] +
                marker_end +
                marked_text[fragment.end_pos:]
            )

        return marked_text, len(fragments)

    def extract_highlighted_fragments(
        self,
        text: str,
        marker_start: str = ">>>",
        marker_end: str = "<<<",
    ) -> List[str]:
        """Extract text between markers.

        Args:
            text: Text with markers
            marker_start: Start marker
            marker_end: End marker

        Returns:
            List of fragments between markers
        """
        pattern = re.escape(marker_start) + r'(.*?)' + re.escape(marker_end)
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def validate_section(
        self,
        source: str,
        translation: str,
        source_language: str = "russian",
    ) -> Tuple[bool, List[CyrillicFragment]]:
        """Validate that translation has no Cyrillic (for Russian source).

        Args:
            source: Source text
            translation: Translated text
            source_language: Source language name

        Returns:
            Tuple of (is_valid, list of Cyrillic fragments)
        """
        # Only validate if source is Russian/Cyrillic-based
        if source_language.lower() not in ["russian", "ukrainian", "belarusian"]:
            return True, []

        # Check for Cyrillic in translation
        fragments = self.find_cyrillic_fragments(
            translation,
            context_chars=100,
            exclude_formulas=True,
        )

        is_valid = len(fragments) == 0
        return is_valid, fragments

    def format_fragment_report(self, fragments: List[CyrillicFragment]) -> str:
        """Format Cyrillic fragments for reporting.

        Args:
            fragments: List of fragments

        Returns:
            Formatted report string
        """
        if not fragments:
            return "No Cyrillic fragments found."

        lines = [f"Found {len(fragments)} Cyrillic fragment(s):"]
        lines.append("")

        for i, fragment in enumerate(fragments, 1):
            lines.append(f"{i}. Text: '{fragment.text}'")
            lines.append(f"   Position: {fragment.start_pos}-{fragment.end_pos}")
            lines.append(f"   Context: ...{fragment.context}...")
            lines.append("")

        return "\n".join(lines)
