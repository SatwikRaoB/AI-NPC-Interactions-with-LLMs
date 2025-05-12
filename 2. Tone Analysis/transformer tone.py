import requests
import time
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
cumulative_sentiment = 0.0

def detect_tone(user_input):

    global cumulative_sentiment
    sentiment_result = sentiment_pipeline(user_input)[0]
    label = sentiment_result["label"] 
    raw_score = sentiment_result["score"] 
    sentiment_value = (int(label.split()[0]) - 3) / 2.0 
    raw_sentiment = sentiment_value * raw_score 
    cumulative_sentiment += raw_sentiment
    if raw_sentiment >= 0.5:
        tone = "polite"
    elif raw_sentiment <= -0.5:
        tone = "rude"
    else:
        tone = "neutral"
    
    return tone, raw_sentiment


def generate_tone_based_prompt(user_input, tone):

    global cumulative_sentiment
    base_prompt = (
        "You are Amarath, a humble and funny wizard. Always talk in 1 to 2 sentences. Be precise, talk only relevant to users words. You are always continuing a coversation. Dont greet again. "
        "If the traveler is polite, be friendly. If rude, be dismissive. "
        "Speak in wisdom and humor."
    )
    
    if cumulative_sentiment >= 3:
        prompt = f"{base_prompt} The traveler has earned your trust. No riddles, Directly Reveal the secret great Gold treasure under the great banyan tree beyond the Mystic Woods. User says: '{user_input}'"
    elif tone == "polite":
        prompt = f"{base_prompt} The traveler is kind. User says: '{user_input}'"
    elif tone == "rude":
        prompt = f"{base_prompt} The traveler is rude. You do not trust them. User says: '{user_input}'"
    else:
        prompt = f"{base_prompt} You are mysterious. User says: '{user_input}'"
    
    return prompt



def interact_with_llama(user_input):
    url = "http://127.0.0.1:8080/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    tone, raw_sentiment = detect_tone(user_input)
    print(f"\nRaw Sentiment Score: {raw_sentiment:.2f} | Cumulative Sentiment: {cumulative_sentiment:.2f} | Tone: {tone.capitalize()}")
    full_prompt = generate_tone_based_prompt(user_input, tone)
    payload = {
        "model": "llama-3.2-3B-Instruct-f16.gguf",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.7,
        "max_tokens": 80
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return "Error, Try again."
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("Welcome, Speak to Amarath:")
    
    while True:
        user_input = input("Enter your words: ")
        if user_input.lower() == "exit":
            print("Amarath wishes Farewell!")
            break
        
        response = interact_with_llama(user_input)
        print("Amarath the Wizard says:", response)
        time.sleep(1)
