"""Formula validation for translated documents."""

import re
from typing import List, Tuple, Dict

from .models import Document, Section, ParagraphValidation
from .latex_parser import LaTeXParser


class FormulaValidator:
    """Validates that formulas are preserved in translation."""

    def __init__(self, latex_parser: LaTeXParser):
        """Initialize validator.

        Args:
            latex_parser: LaTeX parser instance
        """
        self.parser = latex_parser

    def validate_document(self, document: Document) -> List[ParagraphValidation]:
        """Validate all sections in document.

        Args:
            document: Document to validate

        Returns:
            List of validation results for problematic paragraphs
        """
        all_validations = []

        for section in document.sections:
            if not section.translation:
                continue

            validations = self.validate_section(section)
            all_validations.extend(validations)

        return all_validations

    def validate_section(self, section: Section) -> List[ParagraphValidation]:
        """Validate formulas in a section.

        Args:
            section: Section to validate

        Returns:
            List of validation results (only problematic ones)
        """
        # Split into paragraphs
        source_paragraphs = self._split_paragraphs(section.content)
        target_paragraphs = self._split_paragraphs(section.translation)

        # If paragraph count differs significantly, validate as single block
        if abs(len(source_paragraphs) - len(target_paragraphs)) > 1:
            return self._validate_as_whole(section)

        # Validate paragraph by paragraph
        problematic = []

        for i in range(min(len(source_paragraphs), len(target_paragraphs))):
            src_para = source_paragraphs[i]
            tgt_para = target_paragraphs[i]

            validation = self._validate_paragraph(i, src_para, tgt_para)

            if not validation.inline_match or not validation.display_match:
                problematic.append(validation)

        return problematic

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split by double newline or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _validate_paragraph(
        self,
        index: int,
        source: str,
        target: str,
    ) -> ParagraphValidation:
        """Validate formulas in a paragraph pair.

        Args:
            index: Paragraph index
            source: Source paragraph
            target: Target paragraph

        Returns:
            Validation result
        """
        # Extract formulas
        src_inline, src_display = self.parser.extract_formulas_from_paragraph(source)
        tgt_inline, tgt_display = self.parser.extract_formulas_from_paragraph(target)

        # Normalize formulas (remove extra whitespace)
        src_inline = [self._normalize_formula(f) for f in src_inline]
        tgt_inline = [self._normalize_formula(f) for f in tgt_inline]
        src_display = [self._normalize_formula(f) for f in src_display]
        tgt_display = [self._normalize_formula(f) for f in tgt_display]

        # Check inline formulas (order doesn't matter)
        inline_match = set(src_inline) == set(tgt_inline)

        # Check display formulas (order matters)
        display_match = src_display == tgt_display

        # Generate diff if mismatch
        diff = None
        if not inline_match or not display_match:
            diff = self._generate_diff(src_inline, tgt_inline, src_display, tgt_display)

        return ParagraphValidation(
            paragraph_index=index,
            source_inline=src_inline,
            target_inline=tgt_inline,
            source_display=src_display,
            target_display=tgt_display,
            inline_match=inline_match,
            display_match=display_match,
            diff=diff,
        )

    def _validate_as_whole(self, section: Section) -> List[ParagraphValidation]:
        """Validate section as a whole when paragraph alignment fails.

        Args:
            section: Section to validate

        Returns:
            Single validation result
        """
        validation = self._validate_paragraph(
            0,
            section.content,
            section.translation,
        )

        return [validation] if not (validation.inline_match and validation.display_match) else []

    def _normalize_formula(self, formula: str) -> str:
        """Normalize formula for comparison.

        Args:
            formula: Formula to normalize

        Returns:
            Normalized formula
        """
        # Remove extra whitespace
        formula = re.sub(r'\s+', ' ', formula)
        return formula.strip()

    def _generate_diff(
        self,
        src_inline: List[str],
        tgt_inline: List[str],
        src_display: List[str],
        tgt_display: List[str],
    ) -> str:
        """Generate human-readable diff of formula mismatches.

        Args:
            src_inline: Source inline formulas
            tgt_inline: Target inline formulas
            src_display: Source display formulas
            tgt_display: Target display formulas

        Returns:
            Diff string
        """
        diff_parts = []

        # Check inline formulas
        src_inline_set = set(src_inline)
        tgt_inline_set = set(tgt_inline)

        missing_inline = src_inline_set - tgt_inline_set
        extra_inline = tgt_inline_set - src_inline_set

        if missing_inline:
            diff_parts.append(f"Missing inline formulas: {', '.join(f'${f}$' for f in missing_inline)}")

        if extra_inline:
            diff_parts.append(f"Extra inline formulas: {', '.join(f'${f}$' for f in extra_inline)}")

        # Check display formulas
        if src_display != tgt_display:
            diff_parts.append(f"Display formulas mismatch:")
            diff_parts.append(f"  Source: {len(src_display)} formulas")
            diff_parts.append(f"  Target: {len(tgt_display)} formulas")

            # Show differences
            for i, (src, tgt) in enumerate(zip(src_display, tgt_display)):
                if src != tgt:
                    diff_parts.append(f"  Formula {i+1} differs")

            # Show missing/extra
            if len(src_display) != len(tgt_display):
                diff_parts.append(f"  Count mismatch: {len(src_display)} vs {len(tgt_display)}")

        return "; ".join(diff_parts)

    def mark_problematic_paragraph(
        self,
        paragraph: str,
        validation: ParagraphValidation,
        color: str = "red",
    ) -> str:
        """Mark a problematic paragraph with color and footnote.

        Args:
            paragraph: Paragraph text
            validation: Validation result
            color: LaTeX color to use

        Returns:
            Marked paragraph
        """
        # Escape special characters in diff
        diff_escaped = validation.diff.replace('_', '\\_').replace('$', '\\$')

        marked = f"{{\\color{{{color}}} {paragraph}\\footnote{{{diff_escaped}}}}}"
        return marked
