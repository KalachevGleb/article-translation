"""HTML report generation for translation process."""

from datetime import datetime
from typing import List, Dict
from pathlib import Path

from jinja2 import Template

from .models import Document, ParagraphValidation, Term


class ReportGenerator:
    """Generates HTML reports for translation results."""

    TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Translation Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .status-success {
            color: #27ae60;
            font-weight: bold;
        }
        .status-warning {
            color: #f39c12;
            font-weight: bold;
        }
        .status-error {
            color: #e74c3c;
            font-weight: bold;
        }
        .section {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-item {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
        }
        .stat-label {
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .term-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .term-item {
            background: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }
        .term-source {
            font-weight: bold;
            color: #2c3e50;
        }
        .term-target {
            color: #27ae60;
        }
        .problem-list {
            margin-top: 15px;
        }
        .problem-item {
            background: #fee;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 4px solid #e74c3c;
        }
        .problem-title {
            font-weight: bold;
            color: #c0392b;
            margin-bottom: 5px;
        }
        .diff {
            font-family: monospace;
            background: #fff;
            padding: 10px;
            border-radius: 3px;
            margin-top: 5px;
            font-size: 0.9em;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📄 Article Translation Report</h1>
        <p><strong>Source:</strong> {{ source_file }}</p>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <p><strong>Status:</strong>
            <span class="status-{{ status_class }}">{{ status_text }}</span>
        </p>
    </div>

    <div class="section">
        <h2>📊 Statistics</h2>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">Total Sections</div>
                <div class="stat-value">{{ stats.total_sections }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Translated Sections</div>
                <div class="stat-value">{{ stats.translated_sections }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Terms Extracted</div>
                <div class="stat-value">{{ stats.total_terms }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Problematic Paragraphs</div>
                <div class="stat-value">{{ stats.problematic_paragraphs }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Retry Attempts</div>
                <div class="stat-value">{{ stats.total_retries }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Execution Time</div>
                <div class="stat-value">{{ stats.execution_time }}</div>
            </div>
        </div>
    </div>

    {% if terms %}
    <div class="section">
        <h2>📚 Terminology Dictionary</h2>
        <div class="term-list">
            {% for term in terms %}
            <div class="term-item">
                <span class="term-source">{{ term.source }}</span> →
                <span class="term-target">{{ term.target }}</span>
                {% if term.context %}
                <div style="font-size: 0.85em; color: #7f8c8d; margin-top: 3px;">
                    {{ term.context }}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    {% if problems %}
    <div class="section">
        <h2>⚠️ Problematic Paragraphs</h2>
        <div class="problem-list">
            {% for problem in problems %}
            <div class="problem-item">
                <div class="problem-title">
                    Section: {{ problem.section_title }} (Paragraph {{ problem.paragraph_index + 1 }})
                </div>
                <div class="diff">{{ problem.diff }}</div>
                <div style="margin-top: 10px; font-size: 0.9em;">
                    <strong>Inline formulas:</strong>
                    {{ problem.source_inline|length }} source,
                    {{ problem.target_inline|length }} target
                    {% if not problem.inline_match %}❌{% else %}✓{% endif %}
                    <br>
                    <strong>Display formulas:</strong>
                    {{ problem.source_display|length }} source,
                    {{ problem.target_display|length }} target
                    {% if not problem.display_match %}❌{% else %}✓{% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <div class="footer">
        <p>Generated by Article Translation System</p>
        <p>Exit Code: {{ exit_code }}</p>
    </div>
</body>
</html>
"""

    def generate_report(
        self,
        document: Document,
        terms: List[Term],
        problematic: List[ParagraphValidation],
        execution_time: float,
        exit_code: int,
        output_path: str,
    ) -> str:
        """Generate HTML report.

        Args:
            document: Translated document
            terms: List of terms used
            problematic: List of problematic paragraphs
            execution_time: Total execution time in seconds
            exit_code: Process exit code
            output_path: Path to save report

        Returns:
            Path to generated report
        """
        # Calculate statistics
        total_sections = len(document.sections)
        translated_sections = sum(1 for s in document.sections if s.translation)
        total_retries = sum(s.translation_attempts - 1 for s in document.sections if s.translation_attempts > 1)

        stats = {
            "total_sections": total_sections,
            "translated_sections": translated_sections,
            "total_terms": len(terms),
            "problematic_paragraphs": len(problematic),
            "total_retries": total_retries,
            "execution_time": f"{execution_time:.1f}s",
        }

        # Determine status
        if exit_code == 0:
            if len(problematic) == 0:
                status_text = "SUCCESS"
                status_class = "success"
            else:
                status_text = f"COMPLETED WITH WARNINGS ({len(problematic)} issues)"
                status_class = "warning"
        else:
            status_text = f"FAILED (Exit code: {exit_code})"
            status_class = "error"

        # Enrich problematic paragraphs with section info
        problems_with_context = []
        for problem in problematic:
            # Find which section this problem belongs to
            section_title = "Unknown"
            for section in document.sections:
                if section.translation:
                    paragraphs = section.translation.split('\n\n')
                    if problem.paragraph_index < len(paragraphs):
                        section_title = section.title
                        break

            problems_with_context.append({
                "section_title": section_title,
                "paragraph_index": problem.paragraph_index,
                "diff": problem.diff,
                "source_inline": problem.source_inline,
                "target_inline": problem.target_inline,
                "source_display": problem.source_display,
                "target_display": problem.target_display,
                "inline_match": problem.inline_match,
                "display_match": problem.display_match,
            })

        # Render template
        template = Template(self.TEMPLATE)
        html = template.render(
            source_file=document.source_path,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status_text=status_text,
            status_class=status_class,
            stats=stats,
            terms=terms,
            problems=problems_with_context,
            exit_code=exit_code,
        )

        # Save report
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return str(report_path)
