from vector_store import VectorStore
from embedder import Embedder
from llama_client import LlamaClient

class RAGMemory:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = Embedder()
        self.llama_client = LlamaClient()
        self.turn_id = 1

    def store_conversation(self, question, response):
        context = f"Q: {question}\nA: {response}"
        embedding = self.embedder.embed(context)
        self.vector_store.store(context, embedding, self.turn_id)
        self.turn_id += 1

    def get_relevant_context(self, question):
        question_embedding = self.embedder.embed(question)
        return self.vector_store.retrieve(question_embedding)

    def answer(self, question):
        context = self.get_relevant_context(question)
        response = self.llama_client.query(question, context)
        self.store_conversation(question, response)
        return response  