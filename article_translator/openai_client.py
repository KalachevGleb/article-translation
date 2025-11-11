"""OpenAI API client wrapper with OpenRouter support."""

import os
import time
from typing import Optional, List, Dict
from openai import OpenAI, OpenAIError
import tiktoken


class OpenAIClient:
    """Wrapper for OpenAI/OpenRouter API with error handling and retries.

    Supports both direct OpenAI API and OpenRouter for accessing multiple providers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 16000,
        max_retries: int = 3,
    ):
        """Initialize OpenAI/OpenRouter client.

        Args:
            api_key: API key (OpenAI or OpenRouter)
            base_url: Base URL for API (use "https://openrouter.ai/api/v1" for OpenRouter)
            model: Model name (use "provider/model" format for OpenRouter)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided (set OPENAI_API_KEY or OPENROUTER_API_KEY)")

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        # Initialize client with optional base_url for OpenRouter
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

        # Initialize tokenizer for counting tokens
        try:
            # Try to get encoding for the model (works for OpenAI models)
            model_name = model.split('/')[-1] if '/' in model else model
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for unknown models (e.g., from other providers)
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Make a chat completion request with retries.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Response text

        Raises:
            OpenAIError: If request fails after retries
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                )
                return response.choices[0].message.content

            except OpenAIError as e:
                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        raise OpenAIError("Failed to get response after retries")

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """Get embedding for text.

        Args:
            text: Text to embed
            model: Embedding model name

        Returns:
            Embedding vector
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=text,
                )
                return response.data[0].embedding

            except OpenAIError as e:
                if attempt == self.max_retries - 1:
                    raise

                wait_time = 2 ** attempt
                print(f"Embedding API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        raise OpenAIError("Failed to get embedding after retries")
