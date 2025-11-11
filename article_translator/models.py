"""Data models for the article translation system."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


class FormulaType(Enum):
    """Type of LaTeX formula."""
    INLINE = "inline"  # $...$
    DISPLAY = "display"  # $$...$$, \[...\], equation, align, etc.


@dataclass
class Formula:
    """Represents a LaTeX formula."""
    content: str
    formula_type: FormulaType
    position: int  # Position in the text


@dataclass
class Section:
    """Represents a section or subsection of the document."""
    id: str
    title: str
    content: str
    level: int  # 1 for section, 2 for subsection, etc.
    formulas: List[Formula] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)  # IDs of dependent sections
    translation: Optional[str] = None
    translation_attempts: int = 0


@dataclass
class Term:
    """Represents a terminology term with translation."""
    source: str
    target: str
    context: str = ""
    confidence: float = 1.0
    approved: bool = False


@dataclass
class ParagraphValidation:
    """Result of formula validation for a paragraph."""
    paragraph_index: int
    source_inline: List[str]
    target_inline: List[str]
    source_display: List[str]
    target_display: List[str]
    inline_match: bool
    display_match: bool
    diff: Optional[str] = None


@dataclass
class TranslationResult:
    """Result of the translation process."""
    exit_code: int
    translated_content: str
    report_path: str
    statistics: Dict[str, any] = field(default_factory=dict)
    problematic_paragraphs: List[ParagraphValidation] = field(default_factory=list)


@dataclass
class Document:
    """Represents the complete document structure."""
    source_path: str
    content: str
    sections: List[Section] = field(default_factory=list)
    preamble: str = ""
    postamble: str = ""

    def get_section(self, section_id: str) -> Optional[Section]:
        """Get section by ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None
