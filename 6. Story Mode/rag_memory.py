import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os
import numpy as np
import random

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class Embedder:
    _instance = None
    _model = None
    def __new__(cls, model_name=EMBEDDING_MODEL):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            try:
                cls._model = SentenceTransformer(model_name)
            except Exception as e:
                cls._instance = None
                cls._model = None
                raise
        return cls._instance

    def embed(self, text):
        if self._model is None or not text or not isinstance(text, str):
            return None
        try:
            embedding = self._model.encode(text, normalize_embeddings=True).tolist()
            return embedding
        except Exception as e:
            return None

class VectorStore:
    def __init__(self, persona_name, persist_directory):
        if not persona_name or not persist_directory:
            raise ValueError("Invalid persona or directory")
        self.persona_name = persona_name
        safe_persona_name = persona_name.lower().replace(" ", "_").replace("-", "_")
        self.collection_name = f"{safe_persona_name}_knowledge_base"
        self.persist_directory = persist_directory
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            raise

    def get_count(self):
        try:
            return self.collection.count()
        except Exception:
            return 0

    def store_fact(self, fact_text, embedding, fact_id=None):
        if not fact_text or embedding is None:
            return False
        if fact_id is None:
            fact_id = str(uuid.uuid4())
        elif not isinstance(fact_id, str):
            fact_id = str(fact_id)
        try:
            self.collection.add(documents=[fact_text], embeddings=[embedding], ids=[fact_id])
            return True
        except chromadb.errors.IDAlreadyExistsError:
            return False
        except Exception:
            return False

    def retrieve_relevant_facts(self, query_embedding, top_k=5, similarity_threshold=0.6):
        retrieved_texts = []
        if query_embedding is None:
            return retrieved_texts
        try:
            current_count = self.collection.count()
            if current_count == 0:
                return retrieved_texts
            adjusted_top_k = min(top_k, current_count)
            if adjusted_top_k <= 0:
                return retrieved_texts
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=adjusted_top_k,
                include=['documents', 'distances']
            )
            if results and results.get('documents') and results['documents'][0]:
                documents = results['documents'][0]
                distances = results['distances'][0]
                for doc, dist in zip(documents, distances):
                    similarity = 1 - dist
                    if similarity >= similarity_threshold:
                        retrieved_texts.append(doc)
        except Exception as e:
            pass
        return retrieved_texts

    def get_all_facts(self):
        try:
            results = self.collection.get(include=['documents'])
            return results.get('documents', [])
        except Exception:
            return []

    def get_random_fact(self):
        try:
            all_facts = self.get_all_facts()
            return random.choice(all_facts) if all_facts else None
        except Exception:
            return None

    def clear_all_facts(self):
        try:
            current_count = self.collection.count()
            if current_count == 0:
                return True
            all_ids = self.collection.get(include=[])['ids']
            if all_ids:
                self.collection.delete(ids=all_ids)
                return self.collection.count() == 0
            else:
                return True
        except Exception:
            return False