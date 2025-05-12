import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        try:
            if not text or not isinstance(text, str):
                return None
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error embedding text: {e}")
            return None

class VectorStore:
    def __init__(self, agent_name, db_path):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=f"{agent_name.lower()}_facts",
            metadata={"hnsw:space": "cosine"}
        )
        self.agent_name = agent_name

    def add_fact(self, fact_id, content, embedding, metadata=None):
        try:
            if embedding is None:
                print(f"[{self.agent_name}] Skipping fact {fact_id}: No valid embedding.")
                return
            self.collection.add(
                ids=[fact_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata or {"agent": self.agent_name}]
            )
        except Exception as e:
            print(f"[{self.agent_name}] Error adding fact {fact_id}: {e}")

    def retrieve_relevant_facts(self, query_embedding, top_k=5, where=None):
        try:
            if query_embedding is None:
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents"]
            )
            facts = results.get("documents", [[]])[0]
            return facts if facts else []
        except Exception as e:
            print(f"[{self.agent_name}] Error retrieving facts: {e}")
            return []