"""Dependency analyzer for document sections."""

import json
from typing import List, Dict, Set, Optional
from collections import defaultdict, deque

from .models import Document, Section
from .openai_client import OpenAIClient
from .prompt_loader import PromptLoader


class DependencyAnalyzer:
    """Analyzes dependencies between document sections using LLM."""

    def __init__(self, openai_client: OpenAIClient, prompt_loader: Optional[PromptLoader] = None):
        """Initialize analyzer.

        Args:
            openai_client: OpenAI client instance
            prompt_loader: Prompt loader instance (creates default if None)
        """
        self.client = openai_client
        self.prompt_loader = prompt_loader or PromptLoader()

    def analyze_dependencies(self, document: Document) -> Document:
        """Analyze dependencies between sections.

        Args:
            document: Document to analyze

        Returns:
            Document with updated dependencies
        """
        if len(document.sections) <= 1:
            return document

        # Load prompt configuration
        prompt_config = self.prompt_loader.load("dependency_analysis")

        # Build section summary for LLM
        sections_info = []
        for section in document.sections:
            sections_info.append({
                "id": section.id,
                "title": section.title,
                "level": section.level,
                "content_preview": section.content[:500],  # First 500 chars
            })

        sections_json = json.dumps(sections_info, ensure_ascii=False, indent=2)

        # Format prompts
        system_prompt = prompt_config.format_system_prompt()
        user_prompt = prompt_config.format_user_prompt(sections_json=sections_json)

        # Get LLM response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat_completion(
            messages,
            temperature=prompt_config.temperature,
            max_tokens=prompt_config.max_tokens,
        )

        # Parse response
        dependencies = self._parse_dependencies(response)

        # Update document sections
        for section in document.sections:
            if section.id in dependencies:
                section.dependencies = dependencies[section.id]

        return document

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
