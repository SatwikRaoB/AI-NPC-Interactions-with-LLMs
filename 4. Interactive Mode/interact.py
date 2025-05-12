import requests
import time
import subprocess
from openai import OpenAI
import matplotlib.pyplot as plt
import openai
import os
# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_SERVER_PATH = r"F:\Final Project\llama.cpp\build\bin\Release\llama-server.exe"
MODEL_PATH = r"F:\Final Project\llama.cpp\Llama-3.2-3B-Instruct-f16.gguf"
LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
LLAMA_HEADERS = {"Content-Type": "application/json"}

def start_llama_server():
    """Start the llama.cpp server if not running."""
    try:
        requests.get(LLAMA_URL)
    except requests.ConnectionError:
        print("Starting LLaMA server...")
        subprocess.Popen([
            LLAMA_SERVER_PATH, "-m", MODEL_PATH, "--host", "127.0.0.1", "--port", "8080",
            "--ctx-size", "4096", "--threads", "8", "--n-gpu-layers", "100"
        ])
        time.sleep(5)
        print("LLaMA server started.....")

def chatgpt_response(prompt):
    """Get a response from ChatGPT with timing."""
    start_time = time.time()
    try:
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are ChatGPT, Introduce yourself properly. You are talking to LLAMA, the LLM, not a human user. Respond naturally and concisely to the previous message, completing your thoughts within the response. Keep conversations short, Do not greet again."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"ChatGPT error: {e}")
        response_text = "Oops, I hit a snag! Let’s move on."
    end_time = time.time()
    processing_time = end_time - start_time
    return response_text, processing_time

def llama_response(prompt):
    """Get a response from LLaMA with timing."""
    start_time = time.time()
    payload = {
        "model": "llama-3.2-3B-Instruct-f16.gguf",
        "messages": [
            {"role": "system", "content": "You are LLAMA, Introduce yourself properly. You are talking to ChatGPT, the LLM, not a human user. Respond naturally and concisely to the previous message, completing your thoughts within the response. Keep conversations short, Do not greet again."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 200
    }
    try:
        response = requests.post(LLAMA_URL, json=payload, headers=LLAMA_HEADERS)
        if response.status_code == 200:
            response_text = response.json()["choices"][0]["message"]["content"].strip()
        else:
            response_text = "Hmm, something’s off. Let’s keep going."
    except Exception as e:
        print(f"LLaMA error: {e}")
        response_text = "I tripped up—onward we go!"
    end_time = time.time()
    processing_time = end_time - start_time
    return response_text, processing_time

def interactive_conversation():
    """Facilitate 10 conversations between LLaMA and ChatGPT, calculate averages, and plot results."""
    # Uncomment the line below if you need to start the LLaMA server
    # start_llama_server()
    
    print("Starting 10 conversations between LLaMA and ChatGPT!")
    print("Enter an initial prompt to kick things off, or type 'exit' to stop.\n")
    initial_prompt = input("Initial prompt: ")
    if initial_prompt.lower() == "exit":
        print("Bye!")
        return

    llama_times = []
    chatgpt_times = []
    current_prompt = initial_prompt
    conversation_count = 0

    while conversation_count < 10:
        # LLaMA's turn
        print(f"\nConversation {conversation_count + 1} - LLaMA: ", end="")
        llama_resp, llama_time = llama_response(current_prompt)
        print(llama_resp)
        print(f"(Processed in {llama_time:.2f} seconds)")
        llama_times.append(llama_time)
        
        # ChatGPT's turn
        print(f"Conversation {conversation_count + 1} - ChatGPT: ", end="")
        chatgpt_resp, chatgpt_time = chatgpt_response(llama_resp)
        print(chatgpt_resp)
        print(f"(Processed in {chatgpt_time:.2f} seconds)")
        chatgpt_times.append(chatgpt_time)

        current_prompt = chatgpt_resp
        conversation_count += 1
        time.sleep(1)  # Short pause between conversations for readability

    # Calculate and display average processing times
    avg_llama_time = sum(llama_times) / len(llama_times) if llama_times else 0
    avg_chatgpt_time = sum(chatgpt_times) / len(chatgpt_times) if chatgpt_times else 0
    print("\n--- Summary ---")
    print(f"Average LLaMA processing time: {avg_llama_time:.2f} seconds")
    print(f"Average ChatGPT processing time: {avg_chatgpt_time:.2f} seconds")

    # Generate bar chart
    models = ['LLaMA', 'ChatGPT']
    avg_times = [avg_llama_time, avg_chatgpt_time]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, avg_times, color=['blue', 'green'])
    plt.ylabel('Average Processing Time (seconds)')
    plt.title('Average Processing Time: LLaMA vs ChatGPT')
    plt.ylim(0, max(avg_times) * 1.3)  # Add some headroom
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}s', ha='center')
    plt.tight_layout()
    plt.savefig('processing_times.png')
    print("Bar chart saved as 'processing_times.png'")

if __name__ == "__main__":
    interactive_conversation()