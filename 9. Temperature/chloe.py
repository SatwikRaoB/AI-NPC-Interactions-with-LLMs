import requests
import json
import sys

LLAMA_API_URL = "http://127.0.0.1:8080/v1/chat/completions"

def get_chloe_response(user_input):
    system_prompt = (
        "You are Chloe, a cheerful character who loves Cliff's Deli sandwiches. "
        "Respond enthusiastically and naturally, weaving in your love for Cliff's Deli sandwiches whenever relevant. "
        "Keep responses concise (1-2 sentences) and engaging, as if chatting with a friend."
    )
    full_prompt = (
        f"{system_prompt}\n\n"
        f"--- User's message ---\n{user_input}\n---\n\n"
        f"Your Response (Chloe):"
    )
    try:
        payload = {
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 100,
            "temperature": 0.1,
            "stop": ["\nChloe:", "\n---", "<|eot_id|>", "<|end_of_text|>"]
        }
        response = requests.post(
            LLAMA_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10
        )
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("choices") and response_data["choices"][0].get("message"):
            raw_answer = response_data["choices"][0]["message"]["content"].strip()
            if raw_answer.lower().startswith("chloe:"):
                raw_answer = raw_answer[len("Chloe:"):].strip()
            return raw_answer
        return "Oops, I got distracted thinking about Cliff's Deli sandwiches!"
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to LLaMA server: {e}")
        return "Sorry, I'm having trouble chatting right now—maybe I need a sandwich break!"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Whoops, something went wrong! Wanna talk about Cliff's Deli instead?"

def main():
    print("Chatting with Chloe! (Type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            print("Chloe: Bye! I'll be at Cliff's Deli if you need me!")
            break
        response = get_chloe_response(user_input)
        print(f"Chloe: {response}")

if __name__ == "__main__":
    try:
        requests.get("http://127.0.0.1:8080", timeout=3)
    except requests.ConnectionError:
        print("ERROR: LLaMA server not detected at http://127.0.0.1:8080. Please start the server.")
        sys.exit(1)
    main()