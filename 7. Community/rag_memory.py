# rag_memory.py
import chromadb
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        """Convert text to an embedding vector."""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Error embedding text: {e}")
            return None

class VectorStore:
    def __init__(self, persona_name, db_path):
        self.persona_name = persona_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(f"{persona_name}_facts")

    def store_fact(self, text, embedding, fact_id):
        """Store a fact in the vector database."""
        try:
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[f"fact_{fact_id}"]
            )
        except Exception as e:
            print(f"Error storing fact for {self.persona_name}: {e}")

    def retrieve_relevant_facts(self, query_embedding, top_k=3):
        """Retrieve the top_k most relevant facts."""
        try:
            count = self.collection.count()
            adjusted_top_k = min(top_k, count) if count > 0 else 0
            if adjusted_top_k == 0:
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=adjusted_top_k
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Error retrieving facts for {self.persona_name}: {e}")
            return []

    def get_count(self):
        """Return the number of facts in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting count for {self.persona_name}: {e}")
            return 0

    def clear_all_facts(self):
        """Clear all facts from the collection."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(f"{self.persona_name}_facts")
            return True
        except Exception as e:
            print(f"Error clearing facts for {self.persona_name}: {e}")
            return False