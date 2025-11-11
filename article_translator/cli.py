"""Command-line interface for article translation."""

import argparse
import sys
from pathlib import Path

from .main import ArticleTranslator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automated LaTeX article translation using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate article with default config
  translate-article input.tex output.tex

  # Use custom config
  translate-article input.tex output.tex --config my_config.yaml

  # Interactive terminology review
  translate-article input.tex output.tex --interactive

  # Specify custom report location
  translate-article input.tex output.tex --report report.html
        """
    )

    parser.add_argument(
        "source",
        help="Source LaTeX file"
    )

    parser.add_argument(
        "output",
        help="Output LaTeX file"
    )

    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive terminology review mode"
    )

    parser.add_argument(
        "-r", "--report",
        help="Custom report output path (default: output_file.html)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    args = parser.parse_args()

    # Validate source file
    if not Path(args.source).exists():
        print(f"Error: Source file not found: {args.source}", file=sys.stderr)
        return 1

    # Initialize translator
    try:
        translator = ArticleTranslator(config_path=args.config)
    except Exception as e:
        print(f"Error initializing translator: {e}", file=sys.stderr)
        return 2

    # Perform translation
    terminology_mode = "interactive" if args.interactive else "auto"

    result = translator.translate(
        source_file=args.source,
        output_file=args.output,
        terminology_mode=terminology_mode,
        report_path=args.report,
    )

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
