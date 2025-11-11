"""OpenAI API client wrapper."""

import os
import time
from typing import Optional, List, Dict
from openai import OpenAI, OpenAIError
import tiktoken


class OpenAIClient:
    """Wrapper for OpenAI API with error handling and retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "o3-mini",
        temperature: float = 0.3,
        max_tokens: int = 16000,
        max_retries: int = 3,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        # Initialize tokenizer for counting tokens
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
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
