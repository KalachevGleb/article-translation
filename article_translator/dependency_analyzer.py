"""Dependency analyzer for document sections."""

import json
from typing import List, Dict, Set
from collections import defaultdict, deque

from .models import Document, Section
from .openai_client import OpenAIClient


class DependencyAnalyzer:
    """Analyzes dependencies between document sections using LLM."""

    def __init__(self, openai_client: OpenAIClient):
        """Initialize analyzer.

        Args:
            openai_client: OpenAI client instance
        """
        self.client = openai_client

    def analyze_dependencies(self, document: Document) -> Document:
        """Analyze dependencies between sections.

        Args:
            document: Document to analyze

        Returns:
            Document with updated dependencies
        """
        if len(document.sections) <= 1:
            return document

        # Build section summary for LLM
        sections_info = []
        for section in document.sections:
            sections_info.append({
                "id": section.id,
                "title": section.title,
                "level": section.level,
                "content_preview": section.content[:500],  # First 500 chars
            })

        # Create prompt
        prompt = self._build_dependency_prompt(sections_info)

        # Get LLM response
        messages = [
            {
                "role": "system",
                "content": "You are a scientific document analyzer. Your task is to identify logical dependencies between sections.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        response = self.client.chat_completion(messages)

        # Parse response
        dependencies = self._parse_dependencies(response)

        # Update document sections
        for section in document.sections:
            if section.id in dependencies:
                section.dependencies = dependencies[section.id]

        return document

    def _build_dependency_prompt(self, sections_info: List[Dict]) -> str:
        """Build prompt for dependency analysis."""
        sections_text = json.dumps(sections_info, ensure_ascii=False, indent=2)

        prompt = f"""Проанализируй структуру научной статьи и определи логические зависимости между секциями.

Секция A зависит от секции B, если:
- В секции A используются определения, теоремы или результаты из секции B
- В секции A ссылается на концепции, введенные в секции B
- Секция A логически опирается на материал из секции B

Секции документа:
{sections_text}

Верни результат в формате JSON:
{{
  "dependencies": {{
    "section_id": ["dependency_id1", "dependency_id2", ...],
    ...
  }}
}}

Если секция не имеет зависимостей, укажи пустой список.
Используй только ID секций из предоставленного списка.
"""
        return prompt

    def _parse_dependencies(self, response: str) -> Dict[str, Set[str]]:
        """Parse LLM response with dependencies.

        Args:
            response: JSON response from LLM

        Returns:
            Dictionary mapping section IDs to sets of dependency IDs
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]

            data = json.loads(json_match.strip())
            dependencies = data.get("dependencies", {})

            # Convert lists to sets
            return {
                section_id: set(deps)
                for section_id, deps in dependencies.items()
            }

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Warning: Failed to parse dependencies: {e}")
            return {}

    def topological_sort(self, document: Document) -> List[Section]:
        """Sort sections in topological order based on dependencies.

        Args:
            document: Document with analyzed dependencies

        Returns:
            List of sections in translation order
        """
        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Initialize all sections
        for section in document.sections:
            if section.id not in in_degree:
                in_degree[section.id] = 0

        # Build graph
        for section in document.sections:
            for dep_id in section.dependencies:
                graph[dep_id].append(section.id)
                in_degree[section.id] += 1

        # Kahn's algorithm for topological sort
        queue = deque([
            sec_id for sec_id in in_degree if in_degree[sec_id] == 0
        ])
        sorted_ids = []

        while queue:
            current_id = queue.popleft()
            sorted_ids.append(current_id)

            for neighbor_id in graph[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # Check for cycles
        if len(sorted_ids) != len(document.sections):
            print("Warning: Circular dependencies detected. Using original order.")
            return document.sections

        # Return sections in sorted order
        id_to_section = {sec.id: sec for sec in document.sections}
        return [id_to_section[sec_id] for sec_id in sorted_ids if sec_id in id_to_section]
