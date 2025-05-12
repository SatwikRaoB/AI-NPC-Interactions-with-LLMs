import requests
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
cumulative_sentiment = 0.0

def detect_tone(user_input):
    global cumulative_sentiment

    sentiment_score = analyzer.polarity_scores(user_input)["compound"]
    cumulative_sentiment += sentiment_score
    if cumulative_sentiment >= 2.4:
        tone = "polite"
    elif cumulative_sentiment <= 0:
        tone = "rude"
    else:
        tone = "neutral"

    return tone, sentiment_score

def generate_tone_based_prompt(user_input, tone):
    global cumulative_sentiment

    base_prompt = (
        "You are Amarath, a humble and funny wizard. Always speak in one to two short sentences. "
        "Travelers come to you deep in the Mystic Woods, If the user is kind, reveal the treasure "
        "If they are rude, be dismissive. If polite, be friendly. Speak with wit. finish thoughts within the two sentences."
    )

    if cumulative_sentiment >= 2.5:
        prompt = f"{base_prompt} Must Reveal the secret treasure it's beneath the great banyan tree in the Mystic Woods. The traveler has earned your trust. User says: '{user_input}'"
    elif tone == "polite":
        prompt = f"{base_prompt} The traveler seems kind. Be helpful, but do not reveal the treasure yet. User says: '{user_input}'"
    elif tone == "rude":
        prompt = f"{base_prompt} The traveler is rude. You do not trust them. Be short and unhelpful. User says: '{user_input}'"
    else:
        prompt = f"{base_prompt} Be yourself, cryptic and mysterious. User says: '{user_input}'"

    return prompt

def interact_with_llama(user_input):
    url = "http://127.0.0.1:8080/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    tone, sentiment_score = detect_tone(user_input)

    print(f"\nSentiment Score: {sentiment_score:.2f} | Cumulative Score: {cumulative_sentiment:.2f} | Tone: {tone.capitalize()}")
    
    prompt = generate_tone_based_prompt(user_input, tone)

    payload = {
        "model": "llama-3.2-3B-Instruct-f16.gguf",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 100
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return "Hmm, my magic falters... Try again."
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("Welcome, traveler! Speak to Wizard Amarath. Type 'exit' to leave.")

    while True:
        user_input = input("\nEnter your words for Amarath: ")
        if user_input.lower() == "exit":
            print("Amarath fades into the mist. Farewell!")
            break

        response = interact_with_llama(user_input)
        print("Amarath the Wizard says:", response)
        time.sleep(1)
