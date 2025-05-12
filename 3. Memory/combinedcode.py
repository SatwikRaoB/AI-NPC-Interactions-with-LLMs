import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

class VectorStore:
    def __init__(self, collection_name="conversation_history"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def store(self, text, embedding, turn_id):
        """Store a conversation turn in the vector database."""
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[f"turn_{turn_id}"]
        )

    def retrieve(self, query_embedding, top_k=2):
        """Retrieve the top_k most relevant conversation turns."""
        count = self.collection.count()
        adjusted_top_k = min(top_k, count) if count > 0 else 0
        if adjusted_top_k == 0:
            return ""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=adjusted_top_k
        )
        return "\n\n".join(results['documents'][0]) if results['documents'] else ""

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        """Convert text to an embedding vector."""
        return self.model.encode(text).tolist()

class LlamaClient:
    def __init__(self, host="127.0.0.1", port=8080):
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}

    def query(self, question, context):
        """Send a question and context to the llama-server and get a response."""
        prompt = (
            f"Use the following context as background but do not acknowledge it directly in your response unless asked. "
            f"Focus on answering the user's latest question naturally and concisely:\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer: "
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        if response.status_code == 200:
            raw_answer = response.json()["choices"][0]["message"]["content"]
            return raw_answer.split("Answer:")[-1].strip() if "Answer:" in raw_answer else raw_answer.strip()
        else:
            return f"Error: {response.status_code} - {response.text}"

class RAGMemory:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = Embedder()
        self.llama_client = LlamaClient()
        self.turn_id = 1

    def store_conversation(self, question, response):
        """Store a conversation turn in the vector database."""
        context = f"Q: {question}\nA: {response}"
        embedding = self.embedder.embed(context)
        self.vector_store.store(context, embedding, self.turn_id)
        self.turn_id += 1

    def get_relevant_context(self, question):
        """Retrieve relevant context for a given question."""
        question_embedding = self.embedder.embed(question)
        return self.vector_store.retrieve(question_embedding)

    def answer(self, question):
        """Generate an answer using relevant context."""
        context = self.get_relevant_context(question)
        response = self.llama_client.query(question, context)
        self.store_conversation(question, response)
        return response

def main():
    rag = RAGMemory()
    print("Welcome! Ask a question or type 'quit' to exit.")
    
    while True:
        question = input("Ask a question: ")
        if question.lower() == "quit":
            break
        answer = rag.answer(question)
        print("Answer:", answer)

if __name__ == "__main__":
    main()