import openai
import requests
import time
import os
import json
import random
from dotenv import load_dotenv
try:
    from rag_memory import Embedder, VectorStore
except ImportError:
    print("Error: rag_memory.py not found.")
    exit(1)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in .env file.")

AGENT_NAMES = {
    'A': "Alex", 'B': "Ben", 'C': "Chloe", 'D': "David"
}
LLAMA_AGENTS_IDS = ['A', 'B']
CHATGPT_AGENTS_IDS = ['C', 'D']
FRIEND_PAIRS = [('C', 'A'), ('C', 'B'), ('C', 'D')] 
DB_BASE_PATH = "./community_db"
INCIDENT_TOPIC = "news article discussion"
PAUSE_BETWEEN_TURNS = 2
ARTICLE_IDS = ['news_article_article_0', 'news_article_article_2']  

os.makedirs(DB_BASE_PATH, exist_ok=True)

def get_llm_response(api_type, prompt, context, persona_name, partner_name=None, situation="chat", temp_memory=None, user_prompt=None, selected_article=None):
    response_text = "..."
    system_prompt = f"You are {persona_name}. "

    context = context[:4]
    personality = next((fact for fact in context if fact.startswith("You are")), "Unknown personality")
    
    article_fallbacks = {
        'news_article_article_0': "The Fine Arc Program relocated to Utica’s art district, a creative hub for disabled artists to showcase and sell their work, with a ribbon-cutting at the M building.",
        'news_article_article_2': "Democratic-led states are resisting Trump administration threats to cut school funding over DEI programs, with some leaders refusing to comply and others planning lawsuits."
    }

    if situation == "pairwise_sharing":
        article_fact = next((fact for fact in context if selected_article in fact), None)
        if not article_fact:
            print(f"[{persona_name}] Warning: Article {selected_article} not found in context. Using fallback.")
            article_fact = article_fallbacks.get(selected_article, f"Article {selected_article} details missing.")
        system_prompt += (
            f"You are chatting with {partner_name} greet them and"
            f"Output only the article fact: '{article_fact}' without using adjectives like 'defiant' or 'inspiring'. "
            f"Do not add extra phrases or questions."
        )
    elif situation == "pairwise_reply":
        system_prompt += (
            f"{partner_name} shared: '{prompt}'. "
            f"Respond based on your personality: {personality}. "
            f"If curious, ask one analytical question about implications or motives. "
            f"If indifferent, be dismissive unless directly affected. "
            f"If enthusiastic, express excitement in one to two sentences. "
            f"If unclear (e.g., '...'), say you didn’t catch it. "
            f"Use one to two sentences."
        )
    elif situation == "user_interaction":
        system_prompt += (
            f"Use personality: {personality}. "
            f"Reference Chloe's article if relevant: '{temp_memory}'. "
            f"If your personality is indifferent, be dismissive unless directly affected. "
            f"If your personality is enthusiastic, express excitement in one to two sentences. "
            f"Respond to a user about a news article or topic. "
            f"Respond to: '{user_prompt}'. Use one to two sentences."
        )
    else:
        system_prompt += "Respond naturally."

    full_prompt = f"{system_prompt}\n\n"
    if context:
        context_str = "\n".join([f"- {fact}" for fact in context])
        full_prompt += f"--- Personal Knowledge ---\n{context_str}\n---\n\n"
    full_prompt += f"--- Current Message ---\n{prompt}\n---\n\nYour Response ({persona_name}):"

    stops = []
    all_possible_speakers = [partner_name, persona_name] if partner_name else [persona_name]
    for name in all_possible_speakers:
        if name:
            stops.append(f"\n{name}:")
    stops.append("\n---")
    if api_type == "llama":
        stops.extend(["<|eot_id|>", "<|end_of_text|>"])
    final_stops = stops[:4] if api_type == "openai" and len(stops) > 4 else stops

    max_tokens = 80
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        try:
            if api_type == "openai":
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    stop=final_stops
                )
                if response.choices and response.choices[0].message:
                    response_text = response.choices[0].message.content.strip()
            elif api_type == "llama":
                payload = {
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.05,
                    "top_k": 1,
                    "stop": final_stops
                }
                response = requests.post(
                    "http://127.0.0.1:8080/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=120
                )
                response.raise_for_status()
                response_data = response.json()
                if response_data.get("choices") and response_data["choices"][0].get("message"):
                    raw_answer = response_data["choices"][0]["message"]["content"].strip()
                    if raw_answer.lower().startswith(f"{persona_name.lower()}:"):
                        raw_answer = raw_answer[len(persona_name)+1:].strip()
                    response_text = raw_answer
            else:
                raise ValueError("Invalid api_type")

            if response_text and response_text[-1] not in '.!?':
                print(f"[{persona_name}] Warning: Incomplete response detected. Retrying with increased max_tokens...")
                attempts += 1
                max_tokens += 200 # Increment tokens to allow for longer responses
                continue
            break

        except openai.AuthenticationError:
            print("\nFATAL ERROR: OpenAI Authentication Failed.")
            raise
        except requests.exceptions.RequestException as e:
            print(f"\nLLM connection error ({persona_name}): {e}")
            response_text = "..."
            break
        except Exception as e:
            print(f"\nLLM error ({persona_name}): {e}")
            response_text = "..."
            break

    if attempts == max_attempts and response_text and response_text[-1] not in '.!?':
        response_text = f"{response_text[:50]}… I got cut off, but that’s the gist!"

    return response_text

# --- Main Orchestration ---
def run_community_experiment():
    print("--- Starting Community News Discussion Experiment ---")
    try:
        embedder = Embedder()
    except Exception as e:
        print(f"FATAL ERROR initializing Embedder: {e}")
        return

    all_stores = {}
    try:
        for agent_id, agent_name in AGENT_NAMES.items():
            db_path = os.path.join(DB_BASE_PATH, f"{agent_name.lower()}_db")
            if not os.path.exists(db_path):
                print(f"WARNING: DB path for {agent_name} not found. Run seeding script first.")
                return
            all_stores[agent_id] = VectorStore(agent_name, db_path)
        print("All vector stores accessed.")
        try:
            requests.get("http://127.0.0.1:8080", timeout=3)
        except requests.exceptions.ConnectionError:
            print("WARNING: LLaMA server not detected at http://127.0.0.1:8080.")
            return
    except Exception as e:
        print(f"FATAL ERROR accessing Vector Stores: {e}")
        return

    conversation_log = []
    temp_conversation_memory = {}
    used_articles = []

    print("\n--- Phase 1: Pairwise News Sharing ---")
    for sharer_id, receiver_id in FRIEND_PAIRS:
        sharer_name = AGENT_NAMES[sharer_id]
        receiver_name = AGENT_NAMES[receiver_id]
        sharer_store = all_stores[sharer_id]
        receiver_store = all_stores[receiver_id]
        sharer_api = "llama" if sharer_id in LLAMA_AGENTS_IDS else "openai"
        receiver_api = "llama" if receiver_id in LLAMA_AGENTS_IDS else "openai"

        available_articles = [aid for aid in ARTICLE_IDS if aid not in used_articles]
        if not available_articles:
            available_articles = ARTICLE_IDS
        selected_article = random.choice(available_articles)
        used_articles.append(selected_article)

        print(f"\n--- Conversation: {sharer_name} -> {receiver_name} (Article: {selected_article}) ---")
        context1 = []
        context2 = []
        try:
            trigger_prompt = selected_article
            query_emb1 = embedder.embed(trigger_prompt)
            if query_emb1:
                context1 = sharer_store.retrieve_relevant_facts(query_emb1, top_k=10, where={"type": "article"})
                if not context1:
                    print(f"[{sharer_name}] Debug: No article facts retrieved for {selected_article}. Context: {context1}")
        except Exception as e:
            print(f"[{sharer_name}] Error retrieving context: {e}")
        response1 = get_llm_response(
            sharer_api,
            trigger_prompt,
            context1,
            sharer_name,
            partner_name=receiver_name,
            situation="pairwise_sharing",
            selected_article=selected_article
        )
        print(f"\n{sharer_name}: {response1}")
        conversation_log.append(f"{sharer_name}: {response1}")
        temp_conversation_memory[receiver_id] = response1

        if response1 == "..." or not response1.strip():
            print(f"\n{receiver_name}: I didn’t catch what you said, Chloe. Maybe next time!")
            conversation_log.append(f"{receiver_name}: I didn’t catch what you said, Chloe. Maybe next time!")
            time.sleep(PAUSE_BETWEEN_TURNS)
            continue

        time.sleep(PAUSE_BETWEEN_TURNS)
        last_message = response1
        try:
            query_emb2 = embedder.embed(last_message)
            if query_emb2:
                context2 = receiver_store.retrieve_relevant_facts(query_emb2, top_k=4)
        except Exception as e:
            print(f"[{receiver_name}] Error retrieving context: {e}")
        response2 = get_llm_response(
            receiver_api,
            last_message,
            context2,
            receiver_name,
            partner_name=sharer_name,
            situation="pairwise_reply"
        )
        print(f"\n{receiver_name}: {response2}")
        conversation_log.append(f"{receiver_name}: {response2}")
        time.sleep(PAUSE_BETWEEN_TURNS)

    print("\n\n--- Phase 2: User Interaction ---")
    while True:
        print("\nChoose an agent to talk to (A: Alex, B: Ben, C: Chloe, D: David, or 'exit' to quit):")
        user_choice = input().strip().upper()
        if user_choice == 'EXIT':
            break
        if user_choice not in AGENT_NAMES:
            print("Invalid choice. Please select A, B, C, D, or 'exit'.")
            continue

        agent_name = AGENT_NAMES[user_choice]
        agent_store = all_stores[user_choice]
        agent_api = "llama" if user_choice in LLAMA_AGENTS_IDS else "openai"
        temp_memory = temp_conversation_memory.get(user_choice, "No conversation with Chloe recorded.") if user_choice != 'C' else "You shared news articles with your friends."

        while True:
            print(f"\nTalking to {agent_name}. Enter your prompt (or 'back' to choose another agent):")
            user_prompt = input().strip()
            if user_prompt.lower() == 'back':
                break

            context = []
            try:
                query_text = f"User prompt: {user_prompt}. Chloe's conversation: {temp_memory}"
                query_emb = embedder.embed(query_text)
                if query_emb:
                    context = agent_store.retrieve_relevant_facts(query_emb, top_k=2)
            except Exception as e:
                print(f"[{agent_name}] Error retrieving context: {e}")

            response = get_llm_response(
                agent_api,
                user_prompt,
                context,
                agent_name,
                situation="user_interaction",
                temp_memory=temp_memory,
                user_prompt=user_prompt
            )
            print(f"\n{agent_name}: {response}")
            conversation_log.append(f"{agent_name}: {response}")

    print("\n--- Community Experiment Finished ---")

if __name__ == "__main__":
    run_community_experiment()