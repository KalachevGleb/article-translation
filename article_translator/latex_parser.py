"""LaTeX document parser and flattener."""

import re
import os
from typing import List, Tuple, Optional
from pathlib import Path

from .models import Document, Section, Formula, FormulaType


class LaTeXParser:
    """Parser for LaTeX documents."""

    # Regex patterns for formulas
    INLINE_PATTERN = r'\$([^\$]+)\$'
    DISPLAY_PATTERNS = [
        r'\$\$([^\$]+)\$\$',
        r'\\\[([^\]]+)\\\]',
        r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
        r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
        r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}',
        r'\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}',
        r'\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}',
    ]

    # Regex for sections
    SECTION_PATTERN = r'\\(section|subsection|subsubsection)\{([^}]+)\}'

    def __init__(self, preserve_comments: bool = False):
        """Initialize parser.

        Args:
            preserve_comments: Whether to preserve LaTeX comments
        """
        self.preserve_comments = preserve_comments

    def flatten_document(self, main_file: str) -> str:
        """Flatten LaTeX document by resolving \\input and \\include commands.

        Args:
            main_file: Path to main .tex file

        Returns:
            Flattened content
        """
        main_path = Path(main_file)
        base_dir = main_path.parent

        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove comments if needed
        if not self.preserve_comments:
            content = self._remove_comments(content)

        # Recursively flatten includes
        content = self._resolve_includes(content, base_dir)

        return content

    def _remove_comments(self, content: str) -> str:
        """Remove LaTeX comments."""
        lines = []
        for line in content.split('\n'):
            # Remove comments but preserve escaped %
            line = re.sub(r'(?<!\\)%.*$', '', line)
            lines.append(line)
        return '\n'.join(lines)

    def _resolve_includes(self, content: str, base_dir: Path) -> str:
        """Recursively resolve \\input and \\include commands."""
        # Pattern for \input{file} and \include{file}
        include_pattern = r'\\(input|include)\{([^}]+)\}'

        def replace_include(match):
            cmd, filename = match.groups()

            # Add .tex extension if not present
            if not filename.endswith('.tex'):
                filename += '.tex'

            file_path = base_dir / filename

            if not file_path.exists():
                print(f"Warning: included file not found: {file_path}")
                return match.group(0)

            with open(file_path, 'r', encoding='utf-8') as f:
                included_content = f.read()

            if not self.preserve_comments:
                included_content = self._remove_comments(included_content)

            # Recursively resolve includes in the included file
            included_content = self._resolve_includes(included_content, base_dir)

            return included_content

        return re.sub(include_pattern, replace_include, content)

    def parse_document(self, file_path: str) -> Document:
        """Parse LaTeX document into structured format.

        Args:
            file_path: Path to .tex file

        Returns:
            Document object with sections
        """
        # Flatten the document
        content = self.flatten_document(file_path)

        # Extract preamble (everything before \begin{document})
        preamble_match = re.search(r'^(.*?)\\begin\{document\}', content, re.DOTALL)
        preamble = preamble_match.group(1) if preamble_match else ""

        # Extract main content
        main_match = re.search(
            r'\\begin\{document\}(.*?)\\end\{document\}',
            content,
            re.DOTALL
        )
        main_content = main_match.group(1) if main_match else content

        # Extract postamble (everything after \end{document})
        postamble_match = re.search(r'\\end\{document\}(.*)$', content, re.DOTALL)
        postamble = postamble_match.group(1) if postamble_match else ""

        # Parse sections
        sections = self._parse_sections(main_content)

        return Document(
            source_path=file_path,
            content=main_content,
            sections=sections,
            preamble=preamble,
            postamble=postamble,
        )

    def _parse_sections(self, content: str) -> List[Section]:
        """Parse content into sections."""
        sections = []

        # Find all section commands
        section_matches = list(re.finditer(self.SECTION_PATTERN, content))

        if not section_matches:
            # No sections found, treat entire content as one section
            return [
                Section(
                    id="main",
                    title="Main Content",
                    content=content,
                    level=0,
                    formulas=self._extract_formulas(content),
                )
            ]

        # Process sections
        for i, match in enumerate(section_matches):
            level_name, title = match.groups()
            level = {"section": 1, "subsection": 2, "subsubsection": 3}.get(level_name, 1)

            # Extract content until next section
            start = match.end()
            end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(content)
            section_content = content[start:end].strip()

            section_id = f"sec_{len(sections)}"
            formulas = self._extract_formulas(section_content)

            sections.append(
                Section(
                    id=section_id,
                    title=title,
                    content=section_content,
                    level=level,
                    formulas=formulas,
                )
            )

        return sections

    def _extract_formulas(self, text: str) -> List[Formula]:
        """Extract all formulas from text."""
        formulas = []
        position = 0

        # Extract display formulas first (they can contain $)
        for pattern in self.DISPLAY_PATTERNS:
            for match in re.finditer(pattern, text, re.DOTALL):
                formulas.append(
                    Formula(
                        content=match.group(1).strip(),
                        formula_type=FormulaType.DISPLAY,
                        position=match.start(),
                    )
                )

        # Remove display formulas temporarily to avoid conflicts
        temp_text = text
        for pattern in self.DISPLAY_PATTERNS:
            temp_text = re.sub(pattern, '', temp_text, flags=re.DOTALL)

        # Extract inline formulas
        for match in re.finditer(self.INLINE_PATTERN, temp_text):
            formulas.append(
                Formula(
                    content=match.group(1).strip(),
                    formula_type=FormulaType.INLINE,
                    position=match.start(),
                )
            )

        # Sort by position
        formulas.sort(key=lambda f: f.position)

        return formulas

    def extract_formulas_from_paragraph(self, paragraph: str) -> Tuple[List[str], List[str]]:
        """Extract inline and display formulas from a paragraph.

        Args:
            paragraph: Text to extract from

        Returns:
            Tuple of (inline_formulas, display_formulas)
        """
        formulas = self._extract_formulas(paragraph)
        inline = [f.content for f in formulas if f.formula_type == FormulaType.INLINE]
        display = [f.content for f in formulas if f.formula_type == FormulaType.DISPLAY]
        return inline, display
