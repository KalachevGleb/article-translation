"""Prompt configuration loader from YAML files."""

import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for a single prompt."""
    model: str
    temperature: float
    reasoning_effort: Optional[str]  # low, medium, high для o3 моделей
    max_tokens: int
    system_prompt: str
    user_prompt_template: str

    def format_user_prompt(self, **kwargs) -> str:
        """Format user prompt template with variables.

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted prompt
        """
        return self.user_prompt_template.format(**kwargs)

    def format_system_prompt(self, **kwargs) -> str:
        """Format system prompt with variables.

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted prompt
        """
        return self.system_prompt.format(**kwargs)


class PromptLoader:
    """Loads prompt configurations from YAML files."""

    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize loader.

        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        if prompts_dir is None:
            # Default to prompts/ directory in project root
            self.prompts_dir = Path(__file__).parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)

        self._cache: Dict[str, PromptConfig] = {}

    def load(self, prompt_name: str) -> PromptConfig:
        """Load prompt configuration.

        Args:
            prompt_name: Name of prompt file (without .yaml extension)

        Returns:
            PromptConfig object

        Raises:
            FileNotFoundError: If prompt file not found
            ValueError: If prompt file is invalid
        """
        # Check cache
        if prompt_name in self._cache:
            return self._cache[prompt_name]

        # Load from file
        prompt_file = self.prompts_dir / f"{prompt_name}.yaml"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Validate required fields
        required_fields = ['model', 'temperature', 'max_tokens', 'system_prompt', 'user_prompt_template']
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"Missing required fields in {prompt_file}: {missing}")

        # Create config
        config = PromptConfig(
            model=data['model'],
            temperature=data['temperature'],
            reasoning_effort=data.get('reasoning_effort'),
            max_tokens=data['max_tokens'],
            system_prompt=data['system_prompt'],
            user_prompt_template=data['user_prompt_template'],
        )

        # Cache and return
        self._cache[prompt_name] = config
        return config

    def get_model_params(self, prompt_name: str) -> Dict[str, Any]:
        """Get model parameters from prompt config.

        Args:
            prompt_name: Name of prompt

        Returns:
            Dictionary with model parameters
        """
        config = self.load(prompt_name)

        params = {
            'model': config.model,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
        }

        # Add reasoning_effort for o3 models
        if config.reasoning_effort and config.model.startswith('o'):
            # Note: actual parameter name might be different in final API
            # This is a placeholder based on common patterns
            params['reasoning_effort'] = config.reasoning_effort

        return params

    def list_prompts(self) -> list[str]:
        """List all available prompt configurations.

        Returns:
            List of prompt names (without .yaml extension)
        """
        yaml_files = self.prompts_dir.glob("*.yaml")
        return [f.stem for f in yaml_files]
