"""Terminology extraction and management with vector database."""

import json
import sqlite3
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .models import Term, Document
from .openai_client import OpenAIClient
from .prompt_loader import PromptLoader


class TerminologyManager:
    """Manages terminology extraction and translation with vector database."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        db_path: str = "terms.db",
        embedding_model: str = "text-embedding-3-large",
        similarity_threshold: float = 0.85,
        prompt_loader: Optional[PromptLoader] = None,
    ):
        """Initialize terminology manager.

        Args:
            openai_client: OpenAI client instance
            db_path: Path to terminology database
            embedding_model: Model for embeddings
            similarity_threshold: Threshold for similar terms
            prompt_loader: Prompt loader instance (creates default if None)
        """
        self.client = openai_client
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.prompt_loader = prompt_loader or PromptLoader()

        # Initialize SQLite database for terms
        self.db_path = db_path
        self._init_database()

        # Initialize ChromaDB for vector search
        chroma_path = Path(db_path).parent / "chroma_db"
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="terminology",
            metadata={"hnsw:space": "cosine"},
        )

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                context TEXT,
                confidence REAL DEFAULT 1.0,
                approved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON terms(source)
        """)

        conn.commit()
        conn.close()

    def extract_terms(self, document: Document, source_lang: str, target_lang: str) -> List[Term]:
        """Extract terminology from document using LLM.

        Args:
            document: Document to extract from
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of extracted terms
        """
        # Load prompt configuration
        prompt_config = self.prompt_loader.load("terminology_extraction")

        # Combine all section content
        full_content = "\n\n".join([
            f"## {section.title}\n{section.content[:1000]}"
            for section in document.sections
        ])

        # Truncate if too long
        max_chars = 15000
        if len(full_content) > max_chars:
            full_content = full_content[:max_chars] + "\n..."

        # Format prompts
        system_prompt = prompt_config.format_system_prompt()
        user_prompt = prompt_config.format_user_prompt(
            source_language=source_lang,
            target_language=target_lang,
            content=full_content,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat_completion(
            messages,
            temperature=prompt_config.temperature,
            max_tokens=prompt_config.max_tokens,
        )

        # Parse terms
        terms = self._parse_terms(response)

        # Enrich with similar terms from database
        enriched_terms = []
        for term in terms:
            similar = self.find_similar_terms(term.source, term.context)
            if similar:
                # Use existing translation if confidence is high
                best_match = similar[0]
                if best_match["similarity"] > self.similarity_threshold:
                    term.target = best_match["target"]
                    term.confidence = best_match["similarity"]
                    term.approved = bool(best_match["approved"])

            enriched_terms.append(term)

        return enriched_terms

    def _parse_terms(self, response: str) -> List[Term]:
        """Parse LLM response with terms."""
        try:
            # Extract JSON from response
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]

            data = json.loads(json_match.strip())
            terms_data = data.get("terms", [])

            return [
                Term(
                    source=term["source"],
                    target=term["target"],
                    context=term.get("context", ""),
                    confidence=1.0,
                    approved=False,
                )
                for term in terms_data
            ]

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse terms: {e}")
            return []

    def find_similar_terms(
        self,
        source_term: str,
        context: str = "",
        max_results: int = 5
    ) -> List[Dict]:
        """Find similar terms in the database using vector search.

        Args:
            source_term: Source term to search for
            context: Context for better matching
            max_results: Maximum number of results

        Returns:
            List of similar terms with similarity scores
        """
        # Create query text
        query_text = f"{source_term} {context}".strip()

        # Get embedding
        try:
            embedding = self.client.get_embedding(query_text, self.embedding_model)
        except Exception as e:
            print(f"Warning: Failed to get embedding: {e}")
            return []

        # Query vector database
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=max_results,
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            # Format results
            similar_terms = []
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                similar_terms.append({
                    "source": metadata["source"],
                    "target": metadata["target"],
                    "context": metadata.get("context", ""),
                    "similarity": similarity,
                    "approved": metadata.get("approved", False),
                })

            return similar_terms

        except Exception as e:
            print(f"Warning: Vector search failed: {e}")
            return []

    def save_terms(self, terms: List[Term]):
        """Save terms to database.

        Args:
            terms: List of terms to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for term in terms:
            # Insert into SQLite
            cursor.execute("""
                INSERT INTO terms (source, target, context, confidence, approved)
                VALUES (?, ?, ?, ?, ?)
            """, (
                term.source,
                term.target,
                term.context,
                term.confidence,
                int(term.approved),
            ))

            term_id = cursor.lastrowid

            # Get embedding and add to ChromaDB
            try:
                query_text = f"{term.source} {term.context}".strip()
                embedding = self.client.get_embedding(query_text, self.embedding_model)

                self.collection.add(
                    ids=[f"term_{term_id}"],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": term.source,
                        "target": term.target,
                        "context": term.context,
                        "confidence": term.confidence,
                        "approved": term.approved,
                    }],
                )
            except Exception as e:
                print(f"Warning: Failed to add term to vector DB: {e}")

        conn.commit()
        conn.close()

    def build_dictionary(self, terms: List[Term]) -> Dict[str, str]:
        """Build simple dictionary from terms.

        Args:
            terms: List of terms

        Returns:
            Dictionary mapping source to target
        """
        return {term.source: term.target for term in terms}

    def interactive_review(self, terms: List[Term]) -> List[Term]:
        """Interactively review and approve terms.

        Args:
            terms: Terms to review

        Returns:
            Reviewed terms
        """
        print(f"\n=== Terminology Review ({len(terms)} terms) ===\n")

        reviewed_terms = []
        for i, term in enumerate(terms, 1):
            print(f"{i}/{len(terms)}: {term.source} → {term.target}")
            if term.context:
                print(f"  Context: {term.context}")

            while True:
                choice = input("  [a]ccept / [e]dit / [s]kip: ").lower().strip()

                if choice == 'a':
                    term.approved = True
                    reviewed_terms.append(term)
                    break
                elif choice == 'e':
                    new_target = input(f"  New translation for '{term.source}': ").strip()
                    if new_target:
                        term.target = new_target
                        term.approved = True
                        reviewed_terms.append(term)
                    break
                elif choice == 's':
                    break

            print()

        return reviewed_terms
