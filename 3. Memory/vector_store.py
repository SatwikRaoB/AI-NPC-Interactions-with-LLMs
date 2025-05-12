import chromadb

class VectorStore:
    def __init__(self, collection_name="conversation_history"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def store(self, text, embedding, turn_id):
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[f"turn_{turn_id}"]
        )

    def retrieve(self, query_embedding, top_k=2):
        count = self.collection.count()
        adjusted_top_k = min(top_k, count) if count > 0 else 0
        if adjusted_top_k == 0:
            return ""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=adjusted_top_k
        )
        return "\n\n".join(results['documents'][0]) if results['documents'] else ""