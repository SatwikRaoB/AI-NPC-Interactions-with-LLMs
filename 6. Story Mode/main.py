import openai
import requests
import time
import os
import json
from dotenv import load_dotenv
try:
    from rag_memory import Embedder, VectorStore
except ImportError:
    exit(1)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in .env file.")

os.makedirs("./agnes_brain_db_readonly", exist_ok=True)
os.makedirs("./bartholomew_brain_db_readonly", exist_ok=True)

def get_llm_response(api_type, prompt, context, persona_name, partner_name, history_str=""):
    response_text = "Sorry, I'm having trouble responding right now."
    system_prompt_core = (
        f"You are {persona_name} of Meadowbrook, chatting with {partner_name}. Respond naturally and engagingly, as if catching up with a friend. "
        f"IMPORTANT: Only discuss facts and details that are explicitly provided in your 'Personal Knowledge'. Do not invent or elaborate on details not in your knowledge. "
        #f"If you don't have relevant knowledge about a topic, steer the conversation back to what you do know. "
        f"Always try to reference at least one piece of 'Personal Knowledge' in your response. "
        f"Keep responses concise (1-2 sentences) and natural."
    )
    full_prompt = f"{system_prompt_core}\n\n"
    if context:
        context_str = "\n".join([f"- {fact}" for fact in context])
        full_prompt += f"--- Personal Knowledge ---\n{context_str}\n---\n\n"
    if history_str:
        full_prompt += f"--- Recent Conversation ---\n{history_str}\n---\n\n"
    full_prompt += f"--- Conversation ( {partner_name}'s last message ) ---\n{prompt}\n---\n\nYour Response ({persona_name}):"
    try:
        if api_type == "openai":
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=150,
                temperature=0.1,
                stop=[f"\n{partner_name}:", f"\n{persona_name}:", "\n---"]
            )
            if response.choices and response.choices[0].message:
                response_text = response.choices[0].message.content.strip()
        elif api_type == "llama":
            payload = {
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 150,
                "temperature": 0.1,
                "stop": [f"\n{partner_name}:", f"\n{persona_name}:", "\n---", "<|eot_id|>", "<|end_of_text|>"]
            }
            response = requests.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("choices") and response_data["choices"][0].get("message"):
                raw_answer = response_data["choices"][0]["message"]["content"].strip()
                if raw_answer.lower().startswith(f"{persona_name.lower()}: {persona_name.lower()}:"):
                    raw_answer = raw_answer[len(f"{persona_name}: {persona_name}:"):].strip()
                elif raw_answer.lower().startswith(f"{persona_name.lower()}:"):
                    raw_answer = raw_answer[len(f"{persona_name}:"):].strip()
                response_text = f"{persona_name}: {raw_answer.strip()}"
        else:
            raise ValueError("Invalid api_type")
    except Exception:
        pass

    # Clean up response formatting
    if response_text.lower().startswith(f"{persona_name.lower()}: {persona_name.lower()}:"):
        response_text = response_text[len(f"{persona_name}: {persona_name}:"):].strip()
        response_text = f"{persona_name}: {response_text}"
    elif response_text.lower().startswith(f"{persona_name.lower()}: {persona_name.lower()}"):
        response_text = response_text[len(f"{persona_name}: {persona_name}"):].strip()
        response_text = f"{persona_name}: {response_text}"
    if not response_text.lower().startswith(f"{persona_name.lower()}: "):
        response_text = f"{persona_name}: {response_text.strip()}"
    return response_text

def run_conversation_segment(start_message, start_turn, embedder, agnes_store, bartholomew_store):
    current_message = start_message
    turn = start_turn
    turn_count = 0
    last_message = start_message
    conversation_history = []
    
    while turn_count < 6:
        turn_count += 1
        is_llama_turn = (turn == "llama")
        current_speaker_name = "Bartholomew" if is_llama_turn else "Agnes"
        current_vector_store = bartholomew_store if is_llama_turn else agnes_store
        partner_name = "Agnes" if is_llama_turn else "Bartholomew"
        
        # Only proceed if we have a valid message to respond to
        if not current_message or not current_message.strip():
            return None, None
            
        context_facts = []
        try:
            query_embedding = embedder.embed(current_message)
            if query_embedding is not None:
                retrieved_facts = current_vector_store.retrieve_relevant_facts(query_embedding, top_k=3, similarity_threshold=0.4)
                context_facts = retrieved_facts if retrieved_facts else []
                # Always include a random fact to keep conversation grounded
                random_fact = current_vector_store.get_random_fact()
                if random_fact and random_fact not in context_facts:
                    context_facts.append(random_fact)
        except Exception:
            pass
        
        api_type = "llama" if is_llama_turn else "openai"
        history_str = "\n".join(conversation_history[-4:])
        response = get_llm_response(api_type, current_message, context_facts, current_speaker_name, partner_name, history_str)
        print(response)
        conversation_history.append(response)
        current_message = response[len(f"{current_speaker_name}: "):].strip() if response.startswith(f"{current_speaker_name}: ") else response
        last_message = response
        turn = "agnes" if is_llama_turn else "llama"
        time.sleep(0.5)
        
    return last_message, turn

def interactive_chat():
    try:
        embedder = Embedder()
    except Exception:
        return
    try:
        agnes_store = VectorStore("Agnes", "./agnes_brain_db_readonly")
        bartholomew_store = VectorStore("Bartholomew", "./bartholomew_brain_db_readonly")
        try:
            requests.get("http://127.0.0.1:8080", timeout=3)
        except requests.ConnectionError:
            pass
    except Exception:
        return
    try:
        # Agnes always starts with a random fact from her knowledge
        random_fact = agnes_store.get_random_fact()
        if not random_fact:
            return
        
        # Format Agnes's opening line using her knowledge
        response = get_llm_response("openai", "Start a conversation", [random_fact], "Agnes", "Bartholomew", "")
        print(response)
        
        current_message = response[len("Agnes: "):] if response.startswith("Agnes: ") else response
        next_turn = "llama"  # Bartholomew responds next
        
        while True:
            last_message_segment, next_turn_after_segment = run_conversation_segment(
                current_message, next_turn, embedder, agnes_store, bartholomew_store
            )
            if last_message_segment is None:
                break
            current_message = last_message_segment[len("Bartholomew: "):].strip() if last_message_segment.startswith("Bartholomew: ") else last_message_segment
            next_turn = "agnes" if next_turn_after_segment == "llama" else "llama"
    except KeyboardInterrupt:
        pass
    except openai.AuthenticationError:
        pass
    except Exception:
        pass

if __name__ == "__main__":
    interactive_chat()