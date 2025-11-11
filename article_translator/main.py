"""Main translation orchestrator."""

import os
import time
from pathlib import Path
from typing import Optional, Dict, List
import yaml

from .models import Document, TranslationResult, Term
from .latex_parser import LaTeXParser
from .openai_client import OpenAIClient
from .dependency_analyzer import DependencyAnalyzer
from .terminology_manager import TerminologyManager
from .translation_engine import TranslationEngine
from .formula_validator import FormulaValidator
from .report_generator import ReportGenerator
from .prompt_loader import PromptLoader
from .cyrillic_validator import CyrillicValidator


class ArticleTranslator:
    """Main orchestrator for article translation process."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize translator with configuration.

        Args:
            config_path: Path to config YAML file
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        # Expand environment variables in API key
        api_key = self.config["openai"]["api_key"]
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)

        # Initialize components
        self.openai_client = OpenAIClient(
            api_key=api_key,
            model=self.config["openai"]["model"],
            temperature=self.config["openai"]["temperature"],
            max_tokens=self.config["openai"]["max_tokens"],
        )

        self.latex_parser = LaTeXParser(
            preserve_comments=self.config["latex"].get("preserve_comments", False)
        )

        # Initialize prompt loader
        self.prompt_loader = PromptLoader()

        self.dependency_analyzer = DependencyAnalyzer(
            self.openai_client,
            prompt_loader=self.prompt_loader,
        )

        self.terminology_manager = TerminologyManager(
            self.openai_client,
            db_path=self.config["terminology"]["database_path"],
            embedding_model=self.config["terminology"]["embedding_model"],
            similarity_threshold=self.config["terminology"]["similarity_threshold"],
            prompt_loader=self.prompt_loader,
        )

        self.translation_engine = TranslationEngine(
            self.openai_client,
            source_language=self.config["translation"]["source_language"],
            target_language=self.config["translation"]["target_language"],
            max_retries=self.config["translation"]["max_retries"],
            prompt_loader=self.prompt_loader,
        )

        self.formula_validator = FormulaValidator(self.latex_parser)

        self.cyrillic_validator = CyrillicValidator(self.latex_parser)

        self.report_generator = ReportGenerator()

    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "model": "o3-mini",
                "temperature": 0.3,
                "max_tokens": 16000,
            },
            "translation": {
                "source_language": "russian",
                "target_language": "english",
                "max_retries": 2,
                "auto_mode": True,
            },
            "terminology": {
                "database_path": "terms.db",
                "embedding_model": "text-embedding-3-large",
                "similarity_threshold": 0.85,
            },
            "output": {
                "report_format": "html",
                "mark_problematic": True,
                "problem_color": "red",
            },
            "latex": {
                "preserve_comments": False,
            },
        }

    def translate(
        self,
        source_file: str,
        output_file: str,
        terminology_mode: str = "auto",
        report_path: Optional[str] = None,
    ) -> TranslationResult:
        """Translate a LaTeX article.

        Args:
            source_file: Path to source .tex file
            output_file: Path to output .tex file
            terminology_mode: "auto" or "interactive"
            report_path: Optional custom report path

        Returns:
            TranslationResult with exit code and statistics
        """
        start_time = time.time()
        print("=" * 60)
        print("Article Translation System")
        print("=" * 60)

        try:
            # Phase 1: Parse document
            print("\n[1/6] Parsing LaTeX document...")
            document = self.latex_parser.parse_document(source_file)
            print(f"  Found {len(document.sections)} sections")

            # Phase 2: Analyze dependencies
            print("\n[2/6] Analyzing section dependencies...")
            document = self.dependency_analyzer.analyze_dependencies(document)

            # Get topological order
            sorted_sections = self.dependency_analyzer.topological_sort(document)
            print(f"  Computed translation order")

            # Phase 3: Extract terminology
            print("\n[3/6] Extracting terminology...")
            terms = self.terminology_manager.extract_terms(
                document,
                self.config["translation"]["source_language"],
                self.config["translation"]["target_language"],
            )
            print(f"  Extracted {len(terms)} terms")

            # Interactive review if requested
            if terminology_mode == "interactive":
                terms = self.terminology_manager.interactive_review(terms)
                print(f"  Approved {len(terms)} terms")

            # Save terms to database
            if terms:
                self.terminology_manager.save_terms(terms)

            dictionary = self.terminology_manager.build_dictionary(terms)

            # Phase 4: Translate sections
            print("\n[4/6] Translating sections...")
            document = self.translation_engine.translate_document(
                document,
                sorted_sections,
                dictionary,
            )

            # Phase 5: Validate formulas
            print("\n[5/6] Validating formulas...")
            problematic = self.formula_validator.validate_document(document)
            print(f"  Found {len(problematic)} problematic paragraphs")

            # Retry problematic translations
            if problematic:
                print("  Retrying problematic sections...")
                problematic = self._retry_problematic(document, problematic, dictionary)
                print(f"  {len(problematic)} paragraphs still problematic after retries")

            # Mark problematic paragraphs if configured
            if self.config["output"]["mark_problematic"] and problematic:
                self._mark_problematic_paragraphs(document, problematic)

            # Phase 5.5: Validate Cyrillic (for Russian source)
            cyrillic_issues_count = 0
            if self.config["translation"]["source_language"].lower() in ["russian", "ukrainian", "belarusian"]:
                print("\n[5.5/6] Validating Cyrillic text...")
                cyrillic_issues_count = self._validate_and_fix_cyrillic(document, dictionary)
                if cyrillic_issues_count > 0:
                    print(f"  Fixed {cyrillic_issues_count} section(s) with Cyrillic text")
                else:
                    print(f"  No Cyrillic text found in translation")

            # Phase 6: Generate output
            print("\n[6/6] Generating output...")
            self._write_translated_document(document, output_file)
            print(f"  Written: {output_file}")

            # Generate report
            if not report_path:
                report_path = str(Path(output_file).with_suffix('.html'))

            report_path = self.report_generator.generate_report(
                document=document,
                terms=terms,
                problematic=problematic,
                execution_time=time.time() - start_time,
                exit_code=0 if not problematic else 1,
                output_path=report_path,
            )
            print(f"  Report: {report_path}")

            # Determine exit code
            exit_code = 0 if not problematic else 1

            execution_time = time.time() - start_time
            print("\n" + "=" * 60)
            print(f"Translation completed in {execution_time:.1f}s")
            print(f"Exit code: {exit_code}")
            print("=" * 60)

            return TranslationResult(
                exit_code=exit_code,
                translated_content=self._build_full_document(document),
                report_path=report_path,
                statistics={
                    "sections": len(document.sections),
                    "terms": len(terms),
                    "problematic": len(problematic),
                    "execution_time": execution_time,
                },
                problematic_paragraphs=problematic,
            )

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

            return TranslationResult(
                exit_code=2,
                translated_content="",
                report_path="",
                statistics={"error": str(e)},
            )

    def _retry_problematic(
        self,
        document: Document,
        problematic: List,
        dictionary: Dict[str, str],
    ) -> List:
        """Retry translation for problematic paragraphs.

        Args:
            document: Document being translated
            problematic: List of problematic validations
            dictionary: Terminology dictionary

        Returns:
            Updated list of still-problematic paragraphs
        """
        max_retries = self.config["translation"]["max_retries"]

        # Group by section
        section_issues = {}
        for validation in problematic:
            # Find section - simplified approach
            for section in document.sections:
                if section.translation:
                    if section.id not in section_issues:
                        section_issues[section.id] = []
                    section_issues[section.id].append(validation)
                    break

        # Retry each section
        for section_id in section_issues:
            section = document.get_section(section_id)
            if not section or section.translation_attempts >= max_retries:
                continue

            # Get dependency translations
            dep_translations = {
                dep_id: document.get_section(dep_id).translation
                for dep_id in section.dependencies
                if document.get_section(dep_id) and document.get_section(dep_id).translation
            }

            # Retry translation
            new_translation = self.translation_engine.retry_translation(
                section,
                dictionary,
                dep_translations,
            )

            section.translation = new_translation

        # Re-validate
        return self.formula_validator.validate_document(document)

    def _mark_problematic_paragraphs(self, document: Document, problematic: List):
        """Mark problematic paragraphs in translations.

        Args:
            document: Document to mark
            problematic: List of problematic validations
        """
        color = self.config["output"]["problem_color"]

        for section in document.sections:
            if not section.translation:
                continue

            paragraphs = section.translation.split('\n\n')
            marked_paragraphs = []

            for i, paragraph in enumerate(paragraphs):
                # Find if this paragraph has issues
                validation = None
                for v in problematic:
                    if v.paragraph_index == i:
                        validation = v
                        break

                if validation and not (validation.inline_match and validation.display_match):
                    marked = self.formula_validator.mark_problematic_paragraph(
                        paragraph,
                        validation,
                        color,
                    )
                    marked_paragraphs.append(marked)
                else:
                    marked_paragraphs.append(paragraph)

            section.translation = '\n\n'.join(marked_paragraphs)

    def _write_translated_document(self, document: Document, output_file: str):
        """Write translated document to file.

        Args:
            document: Translated document
            output_file: Output file path
        """
        content = self._build_full_document(document)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _build_full_document(self, document: Document) -> str:
        """Build complete LaTeX document with translations.

        Args:
            document: Document with translations

        Returns:
            Full LaTeX content
        """
        parts = []

        # Add preamble
        if document.preamble:
            parts.append(document.preamble.strip())

        parts.append("\\begin{document}\n")

        # Add translated sections
        for section in document.sections:
            # Add section header
            level_cmd = {1: "section", 2: "subsection", 3: "subsubsection"}.get(section.level, "section")
            if section.level > 0:
                parts.append(f"\\{level_cmd}{{{section.title}}}\n")

            # Add translation or original
            content = section.translation if section.translation else section.content
            parts.append(content)
            parts.append("\n")

        parts.append("\\end{document}")

        # Add postamble
        if document.postamble:
            parts.append(document.postamble.strip())

        return "\n".join(parts)

    def _validate_and_fix_cyrillic(
        self,
        document: Document,
        dictionary: Dict[str, str],
    ) -> int:
        """Validate and fix Cyrillic text in translations.

        Args:
            document: Document to validate
            dictionary: Terminology dictionary

        Returns:
            Number of sections with fixed Cyrillic
        """
        source_lang = self.config["translation"]["source_language"]
        fixed_count = 0

        for section in document.sections:
            if not section.translation:
                continue

            # Check for Cyrillic
            is_valid, fragments = self.cyrillic_validator.validate_section(
                section.content,
                section.translation,
                source_lang,
            )

            if not is_valid:
                print(f"  Found Cyrillic in section: {section.title}")
                print(f"    Fragments: {len(fragments)}")

                # Mark fragments
                marked_text, count = self.cyrillic_validator.mark_cyrillic_fragments(
                    section.translation,
                    marker_start=">>>",
                    marker_end="<<<",
                )

                # Extract highlighted fragments
                highlighted = self.cyrillic_validator.extract_highlighted_fragments(
                    marked_text,
                    marker_start=">>>",
                    marker_end="<<<",
                )

                # Fix with LLM
                try:
                    fixed_text = self.translation_engine.fix_cyrillic(
                        section.translation,
                        marked_text,
                        highlighted,
                        dictionary,
                    )

                    # Update translation
                    section.translation = fixed_text
                    section.translation_attempts += 1
                    fixed_count += 1

                    # Verify fix
                    is_valid_after, fragments_after = self.cyrillic_validator.validate_section(
                        section.content,
                        section.translation,
                        source_lang,
                    )

                    if not is_valid_after:
                        print(f"    Warning: Still has {len(fragments_after)} Cyrillic fragment(s) after fix")
                    else:
                        print(f"    Successfully fixed all Cyrillic fragments")

                except Exception as e:
                    print(f"    Error fixing Cyrillic: {e}")

        return fixed_count
