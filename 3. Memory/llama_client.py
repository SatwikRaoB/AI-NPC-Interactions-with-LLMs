import requests
import json

class LlamaClient:
    def __init__(self, host="127.0.0.1", port=8080):
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}

    def query(self, question, context):
        prompt = (
            f"You are LLama an AI LLM. complete in 1-2 sentences. Be precise and short. Use the following context as background but do not acknowledge it directly in your response unless asked. "
            f"Focus on answering the user's latest question naturally and concisely:\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer: "
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.5
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
        if response.status_code == 200:
            raw_answer = response.json()["choices"][0]["message"]["content"]
            return raw_answer.split("Answer:")[-1].strip() if "Answer:" in raw_answer else raw_answer.strip()
        else:
            return f"Error: {response.status_code} - {response.text}"