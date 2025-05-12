from rag_memory import RAGMemory

def main():
    rag = RAGMemory()
    print("Welcome! Ask a question or type 'quit' to exit.")
    
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = rag.answer(question)
        print("Answer:", answer)

if __name__ == "__main__":
    main()