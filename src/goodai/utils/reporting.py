"""
Reporting Module

Generate reports and summaries from analysis results.
Supports JSON, Markdown, and HTML output.

Good AI Philosophy: Evidence over opinions.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class ReportGenerator:
    """
    Generate reports from analysis results.

    Supports multiple output formats and customization.

    Example:
        >>> generator = ReportGenerator(title="Weekly OEE Report")
        >>> generator.add_section("Summary", {"oee": 0.82, "availability": 0.90})
        >>> generator.add_section("Anomalies", anomaly_list)
        >>> report = generator.to_markdown()
    """

    def __init__(
        self,
        title: str = "Analysis Report",
        author: str = "Good AI Framework",
        include_timestamp: bool = True
    ):
        self.title = title
        self.author = author
        self.include_timestamp = include_timestamp
        self.sections: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()

    def add_section(
        self,
        title: str,
        content: Any,
        section_type: str = "data"
    ):
        """
        Add a section to the report.

        Args:
            title: Section title
            content: Section content (dict, list, dataclass, or string)
            section_type: Type of section (data, table, text, chart)
        """
        # Convert dataclass to dict
        if is_dataclass(content) and not isinstance(content, type):
            content = self._dataclass_to_dict(content)

        self.sections.append({
            "title": title,
            "content": content,
            "type": section_type
        })

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the report."""
        self.metadata[key] = value

    def _dataclass_to_dict(self, obj: Any) -> dict:
        """Convert dataclass to dictionary, handling nested types."""
        if is_dataclass(obj) and not isinstance(obj, type):
            result = {}
            for key, value in asdict(obj).items():
                if hasattr(value, 'value'):  # Enum
                    result[key] = value.value if hasattr(value, 'value') else str(value)
                else:
                    result[key] = value
            return result
        return obj

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "sections": self.sections
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=str
        )

    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        lines = []

        # Title
        lines.append(f"# {self.title}")
        lines.append("")

        # Metadata
        if self.include_timestamp:
            lines.append(f"*Generated: {self.created_at}*")
            lines.append(f"*By: {self.author}*")
            lines.append("")

        if self.metadata:
            lines.append("## Report Info")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Sections
        for section in self.sections:
            lines.append(f"## {section['title']}")
            lines.append("")

            content = section["content"]
            section_type = section["type"]

            if section_type == "text":
                lines.append(str(content))
            elif section_type == "table" and isinstance(content, list):
                lines.extend(self._format_table_md(content))
            elif isinstance(content, dict):
                lines.extend(self._format_dict_md(content))
            elif isinstance(content, list):
                lines.extend(self._format_list_md(content))
            else:
                lines.append(str(content))

            lines.append("")

        return "\n".join(lines)

    def _format_dict_md(self, data: dict, indent: int = 0) -> List[str]:
        """Format dictionary as Markdown."""
        lines = []
        prefix = "  " * indent

        for key, value in data.items():
            formatted_key = key.replace("_", " ").title()

            if isinstance(value, dict):
                lines.append(f"{prefix}- **{formatted_key}**:")
                lines.extend(self._format_dict_md(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}- **{formatted_key}**:")
                for item in value:
                    if isinstance(item, dict):
                        lines.extend(self._format_dict_md(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                # Format numeric values
                if isinstance(value, float):
                    if 0 < abs(value) < 1:
                        formatted_value = f"{value:.2%}"
                    else:
                        formatted_value = f"{value:,.2f}"
                else:
                    formatted_value = str(value)

                lines.append(f"{prefix}- **{formatted_key}**: {formatted_value}")

        return lines

    def _format_list_md(self, data: list) -> List[str]:
        """Format list as Markdown."""
        lines = []

        for item in data:
            if isinstance(item, dict):
                lines.extend(self._format_dict_md(item))
                lines.append("")
            else:
                lines.append(f"- {item}")

        return lines

    def _format_table_md(self, data: list) -> List[str]:
        """Format list of dicts as Markdown table."""
        if not data or not isinstance(data[0], dict):
            return self._format_list_md(data)

        lines = []

        # Get headers from first item
        headers = list(data[0].keys())
        formatted_headers = [h.replace("_", " ").title() for h in headers]

        # Header row
        lines.append("| " + " | ".join(formatted_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Data rows
        for row in data:
            values = []
            for h in headers:
                val = row.get(h, "")
                if isinstance(val, float):
                    if 0 < abs(val) < 1:
                        val = f"{val:.2%}"
                    else:
                        val = f"{val:,.2f}"
                values.append(str(val))
            lines.append("| " + " | ".join(values) + " |")

        return lines

    def to_html(self) -> str:
        """Convert report to HTML format."""
        html_parts = []

        # Basic HTML structure
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html>")
        html_parts.append("<head>")
        html_parts.append(f"<title>{self.title}</title>")
        html_parts.append("<style>")
        html_parts.append(self._get_default_css())
        html_parts.append("</style>")
        html_parts.append("</head>")
        html_parts.append("<body>")

        # Header
        html_parts.append(f"<h1>{self.title}</h1>")
        if self.include_timestamp:
            html_parts.append(f"<p class='meta'>Generated: {self.created_at}</p>")
            html_parts.append(f"<p class='meta'>By: {self.author}</p>")

        # Metadata
        if self.metadata:
            html_parts.append("<div class='metadata'>")
            html_parts.append("<h2>Report Info</h2>")
            html_parts.append("<ul>")
            for key, value in self.metadata.items():
                html_parts.append(f"<li><strong>{key}</strong>: {value}</li>")
            html_parts.append("</ul>")
            html_parts.append("</div>")

        # Sections
        for section in self.sections:
            html_parts.append("<div class='section'>")
            html_parts.append(f"<h2>{section['title']}</h2>")

            content = section["content"]
            section_type = section["type"]

            if section_type == "text":
                html_parts.append(f"<p>{content}</p>")
            elif section_type == "table" and isinstance(content, list):
                html_parts.append(self._format_table_html(content))
            elif isinstance(content, dict):
                html_parts.append(self._format_dict_html(content))
            elif isinstance(content, list):
                html_parts.append(self._format_list_html(content))
            else:
                html_parts.append(f"<p>{content}</p>")

            html_parts.append("</div>")

        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)

    def _get_default_css(self) -> str:
        """Get default CSS styles."""
        return """
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            .meta { color: #888; font-size: 0.9em; }
            .section { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #007bff; color: white; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            ul { list-style-type: none; padding-left: 0; }
            li { padding: 5px 0; }
            .value-good { color: #28a745; }
            .value-warning { color: #ffc107; }
            .value-bad { color: #dc3545; }
        """

    def _format_dict_html(self, data: dict) -> str:
        """Format dictionary as HTML."""
        html = "<dl>"
        for key, value in data.items():
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, dict):
                html += f"<dt>{formatted_key}</dt>"
                html += f"<dd>{self._format_dict_html(value)}</dd>"
            elif isinstance(value, list):
                html += f"<dt>{formatted_key}</dt>"
                html += f"<dd>{self._format_list_html(value)}</dd>"
            else:
                if isinstance(value, float):
                    if 0 < abs(value) < 1:
                        formatted_value = f"{value:.2%}"
                    else:
                        formatted_value = f"{value:,.2f}"
                else:
                    formatted_value = str(value)
                html += f"<dt>{formatted_key}</dt><dd>{formatted_value}</dd>"
        html += "</dl>"
        return html

    def _format_list_html(self, data: list) -> str:
        """Format list as HTML."""
        html = "<ul>"
        for item in data:
            if isinstance(item, dict):
                html += f"<li>{self._format_dict_html(item)}</li>"
            else:
                html += f"<li>{item}</li>"
        html += "</ul>"
        return html

    def _format_table_html(self, data: list) -> str:
        """Format list of dicts as HTML table."""
        if not data or not isinstance(data[0], dict):
            return self._format_list_html(data)

        headers = list(data[0].keys())

        html = "<table>"
        html += "<thead><tr>"
        for h in headers:
            html += f"<th>{h.replace('_', ' ').title()}</th>"
        html += "</tr></thead>"

        html += "<tbody>"
        for row in data:
            html += "<tr>"
            for h in headers:
                val = row.get(h, "")
                if isinstance(val, float):
                    if 0 < abs(val) < 1:
                        val = f"{val:.2%}"
                    else:
                        val = f"{val:,.2f}"
                html += f"<td>{val}</td>"
            html += "</tr>"
        html += "</tbody></table>"

        return html

    def save(
        self,
        path: Union[str, Path],
        format: str = "auto"
    ) -> str:
        """
        Save report to file.

        Args:
            path: Output file path
            format: Output format (auto, json, md, html)

        Returns:
            Path to saved file
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".json":
                format = "json"
            elif suffix in [".md", ".markdown"]:
                format = "md"
            elif suffix in [".html", ".htm"]:
                format = "html"
            else:
                format = "md"

        # Generate content
        if format == "json":
            content = self.to_json()
        elif format == "html":
            content = self.to_html()
        else:
            content = self.to_markdown()

        # Write file
        path.write_text(content)

        return str(path)


def create_oee_report(
    oee_result: Any,
    anomalies: Optional[List[dict]] = None,
    recommendations: Optional[List[str]] = None
) -> ReportGenerator:
    """
    Create a standard OEE report.

    Args:
        oee_result: OEE calculation result
        anomalies: Optional list of detected anomalies
        recommendations: Optional list of recommendations

    Returns:
        Configured ReportGenerator
    """
    report = ReportGenerator(title="OEE Analysis Report")

    # Add OEE metrics
    if hasattr(oee_result, 'to_dict'):
        oee_data = oee_result.to_dict()
    elif hasattr(oee_result, '__dict__'):
        oee_data = oee_result.__dict__
    else:
        oee_data = dict(oee_result)

    report.add_section("OEE Metrics", oee_data)

    # Add anomalies if provided
    if anomalies:
        report.add_section("Detected Anomalies", anomalies, section_type="table")

    # Add recommendations if provided
    if recommendations:
        report.add_section("Recommendations", recommendations)

    return report
