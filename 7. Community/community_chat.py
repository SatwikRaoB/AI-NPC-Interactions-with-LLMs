# community_chat.py (Detective Leads Phase 2 & Concludes)
import openai
import requests
import time
import os
import json
# import subprocess # No longer needed
# import sys # No longer needed
from dotenv import load_dotenv
try: from rag_memory import Embedder, VectorStore
except ImportError: print("Error: rag_memory.py not found."); exit(1)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY: raise ValueError("OpenAI API key not found in .env file.")

# --- Agent & Config Definitions ---
AGENT_NAMES = {
    'A': "Alex", 'B': "Ben", 'C': "Chloe", 'D': "David",
    'E': "Eva", 'F': "Finn", 'G': "Grace", 'H': "Harry"
}
LLAMA_AGENTS_IDS = ['A', 'B', 'E', 'F']
CHATGPT_AGENTS_IDS = ['C', 'D', 'G', 'H']
FRIEND_PAIRS = {'A': 'H', 'B': 'G', 'C': 'F', 'D': 'E'}
RECEIVER_FRIENDS = {v: k for k, v in FRIEND_PAIRS.items()}
CORROBORATION_GROUP_IDS = ['E', 'F', 'G', 'H'] # Order they speak in each round
DB_BASE_PATH = "./community_db"
INCIDENT_TOPIC = "the missing library book incident"
PAUSE_BETWEEN_TURNS = 2
PAIR_CHAT_TURNS = 2
GROUP_CHAT_ROUNDS = 2 # 2 rounds total
DETECTIVE_NAME = "Detective Ruth"

os.makedirs(DB_BASE_PATH, exist_ok=True)

# --- LLM Interaction (Prompts Adjusted for Detective Conclusion) ---
def get_llm_response(api_type, prompt, context, persona_name, partner_name=None, group_members=None, situation="chat", phase1_informant=None, round_num=0):
    response_text = "..."
    system_prompt = f"You are {persona_name}. Respond in 1-3 sentences MAXIMUM. Be concise and factual. "

    if situation == "pairwise_sharing":
        system_prompt += f"You are talking to {partner_name}. Share info about {INCIDENT_TOPIC} based ONLY on your memory facts. Be convincing if needed."
    elif situation == "pairwise_reply":
         system_prompt += f"Your friend {partner_name} just told you about {INCIDENT_TOPIC}. Reply ONLY with acknowledgement."
    elif situation == "group_corroboration":
        group_str = ", ".join(group_members)
        # MODIFIED: Prompt focuses on presenting info/analysis FOR the detective
        system_prompt += (
            f"You are in Round {round_num+1} of a 2-round group discussion led by {DETECTIVE_NAME} with {group_str} about {INCIDENT_TOPIC}. Goal: Help the detective identify the liar by Round 2. "
            f"CRITICAL INSTRUCTIONS: Respond in 1-3 sentences MAX. Stick ONLY to the facts you were told or heard in this discussion. "
        )
        if round_num == 0: # Round 1: State info & find contradiction
            system_prompt += f"1. Briefly PARAPHRASE the key information your friend ({phase1_informant}) told you about who took the book. "
          
            if persona_name == "Grace":
                 system_prompt += f" State the contradiction clearly for the detective"
            """
            elif persona_name == "Harry":
                 system_prompt += f"2. Analyze the contradiction based on all paraphrased stories and identify someone's story as the outlier for the detective."
          
        elif round_num == 1: # Round 2: Reinforce analysis for Detective
             # MODIFIED: Agents reinforce the finding, they don't conclude themselves
             system_prompt += f"Based on the contradiction identified in Round 1, briefly state why someones story seems incorrect or suspicious to help {DETECTIVE_NAME} conclude."
        else: system_prompt += f"Briefly comment on the discussion."
            """
        system_prompt += " DO NOT add extra info."

    # --- Prompt for Detective Ruth's final conclusion ---
    elif situation == "detective_conclusion":
        system_prompt = (
            f"You are {DETECTIVE_NAME}. You have observed a 2-round discussion between Eva, Finn, Grace, and Harry regarding the {INCIDENT_TOPIC}. "
            f"Their discussion transcript is provided below. Analyze the transcript, specifically focusing on the contradiction identified "
            f"Based SOLELY on the provided transcript, state your conclusion about who lied. Be direct and concise (1-3 sentences)."
        )

    full_prompt = f"{system_prompt}\n\n"
    if context: context_str = "\n".join([f"- {fact}" for fact in context]); full_prompt += f"--- Relevant Personal Knowledge ---\n{context_str}\n---\n\n"
    # Use different headers based on situation
    if situation == "group_corroboration":
        full_prompt += f"--- Group Discussion & Your Task ---\n{prompt}\n---\n\nYour Response ({persona_name}):"
    elif situation == "detective_conclusion":
         full_prompt += f"--- Group Discussion Transcript ---\n{prompt}\n---\n\nYour Conclusion ({persona_name}):"
    else: # Pairwise or default
         full_prompt += f"--- Conversation History ---\n{prompt}\n---\n\nYour Response ({persona_name}):"


    stops = []
    # Adjust stops based on situation
    if situation == "group_corroboration":
        all_possible_speakers = group_members + [persona_name] if group_members else [persona_name]
    elif situation == "pairwise_sharing" or situation == "pairwise_reply":
         all_possible_speakers = [partner_name, persona_name] if partner_name else [persona_name]
    else: # Detective conclusion or other
         all_possible_speakers = [persona_name]

    for name in all_possible_speakers:
        if name: stops.append(f"\n{name}:")
    stops.append("\n---")
    if api_type == "llama": stops.extend(["<|eot_id|>", "<|end_of_text|>"])
    final_stops = stops[:4] if api_type == "openai" and len(stops) > 4 else stops

    try:
        max_response_tokens = 80 # Allow enough for analysis/conclusion
        temp = 0.3 # Keep temp low for factual analysis
        if api_type == "openai":
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create( model="gpt-4o-mini", messages=[{"role": "user", "content": full_prompt}], max_tokens=max_response_tokens, temperature=temp, stop=final_stops )
            if response.choices and response.choices[0].message: response_text = response.choices[0].message.content.strip()
        elif api_type == "llama":
            # Use default Llama settings if Ruth uses Llama, otherwise adjust if needed
            payload = { "messages": [{"role": "user", "content": full_prompt}], "max_tokens": max_response_tokens, "temperature": temp, "stop": final_stops }
            response = requests.post("http://127.0.0.1:8080/v1/chat/completions", headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=180); response.raise_for_status()
            response_data = response.json()
            if response_data.get("choices") and response_data["choices"][0].get("message"):
                 raw_answer = response_data["choices"][0]["message"]["content"].strip()
                 # Basic cleanup for any persona name
                 if ":" in raw_answer:
                     first_colon = raw_answer.find(":")
                     potential_name = raw_answer[:first_colon]
                     if potential_name in AGENT_NAMES.values() or potential_name == DETECTIVE_NAME:
                          raw_answer = raw_answer[first_colon+1:].strip()
                 response_text = raw_answer
        else: raise ValueError("Invalid api_type")
    except openai.AuthenticationError: print("\nFATAL ERROR: OpenAI Authentication Failed."); raise
    except requests.exceptions.RequestException as e: print(f"\nLLM connection error ({persona_name}): {e}"); response_text="..."
    except Exception as e:
        if api_type == "openai" and "array_above_max_length" in str(e): print(f"\nLLM error ({persona_name}): OpenAI stop sequence limit exceeded. Error: {e}")
        else: print(f"\nLLM error ({persona_name}): {e}")
        response_text="..."
    return response_text

# --- Main Orchestration ---
def run_community_experiment():
    print("--- Starting Community Corroboration Experiment (Detective Conclusion) ---")
    try: embedder = Embedder()
    except Exception as e: print(f"FATAL ERROR initializing Embedder: {e}"); return

    all_stores = {}
    try:
        # Load ALL stores initially, needed for Phase 1
        for agent_id, agent_name in AGENT_NAMES.items():
            db_path = os.path.join(DB_BASE_PATH, f"{agent_name.lower()}_db")
            if not os.path.exists(db_path): print(f"WARNING: DB path for {agent_name} not found. Run seeding script first."); return
            all_stores[agent_id] = VectorStore(agent_name, db_path)
        print("All vector stores accessed.")
        try: requests.get("http://127.0.0.1:8080", timeout=3)
        except requests.ConnectionError: print("WARNING: LLaMA server not detected at http://127.0.0.1:8080."); return
    except Exception as e: print(f"FATAL ERROR accessing Vector Stores: {e}"); return

    conversation_log = []
    phase1_received_info = {}

    # === Phase 1: Pairwise Sharing ===
    print("\n--- Phase 1: Pairwise Incident Sharing ---")
    for sharer_id, receiver_id in FRIEND_PAIRS.items():
        # ... (Phase 1 logic remains the same) ...
        sharer_name = AGENT_NAMES[sharer_id]
        receiver_name = AGENT_NAMES[receiver_id]
        sharer_store = all_stores[sharer_id]
        receiver_store = all_stores[receiver_id]
        sharer_api = "llama" if sharer_id in LLAMA_AGENTS_IDS else "openai"
        receiver_api = "llama" if receiver_id in LLAMA_AGENTS_IDS else "openai"

        print(f"\n--- Conversation: {sharer_name} -> {receiver_name} ---")
        context1 = []; context2 = []
        try:
            trigger_prompt = f"Tell {receiver_name} about {INCIDENT_TOPIC}."
            query_emb1 = embedder.embed(trigger_prompt)
            if query_emb1: context1 = sharer_store.retrieve_relevant_facts(query_emb1, top_k=3)
        except Exception as e: print(f"[{sharer_name}] Error retrieving context: {e}")
        response1 = get_llm_response(sharer_api, trigger_prompt, context1, sharer_name, partner_name=receiver_name, situation="pairwise_sharing")
        print(f"\n{sharer_name}: {response1}")
        conversation_log.append(f"{sharer_name}: {response1}")
        phase1_received_info[receiver_id] = response1
        time.sleep(PAUSE_BETWEEN_TURNS)
        last_message = response1
        try:
            query_emb2 = embedder.embed("Acknowledge message received.")
            if query_emb2: context2 = receiver_store.retrieve_relevant_facts(query_emb2, top_k=1)
        except Exception as e: print(f"[{receiver_name}] Error retrieving context: {e}")
        response2 = get_llm_response(receiver_api, last_message, context2, receiver_name, partner_name=sharer_name, situation="pairwise_reply")
        print(f"\n{receiver_name}: {response2}")
        conversation_log.append(f"{receiver_name}: {response2}")
        time.sleep(PAUSE_BETWEEN_TURNS)


    # === Phase 2: Group Corroboration (Round Robin - 2 Rounds) ===
    print("\n\n--- Phase 2: Group Corroboration ---")
    group_members_names = [AGENT_NAMES[gid] for gid in CORROBORATION_GROUP_IDS]
    print(f"Group Members: {', '.join(group_members_names)}")

    round_context = f"{DETECTIVE_NAME}: Alright team ({', '.join(group_members_names)}), let's get to the bottom of this {INCIDENT_TOPIC}. Round 1: Briefly state what your friend told you and note any immediate contradictions. Round 2: Reinforce your analysis for me. Keep it concise."
    print(f"\n{round_context}")
    conversation_log.append(round_context)

    phase2_transcript = [round_context] # Start transcript with detective's intro

    # Loop through 2 rounds
    for round_num in range(GROUP_CHAT_ROUNDS): # Runs 2 times (0, 1)
        print(f"\n--- Round {round_num + 1} ---")
        round_messages_this_round = [] # Track messages *within* this round for context

        # Determine the base prompt context (either initial system message or R1 summary)
        # Use the full transcript up to this point for context
        base_prompt_context = "\n".join(phase2_transcript)

        # Inner loop for each agent in the group
        for speaker_id in CORROBORATION_GROUP_IDS: # E, F, G, H in order
            speaker_name = AGENT_NAMES[speaker_id]
            speaker_store = all_stores[speaker_id]
            speaker_api = "llama" if speaker_id in LLAMA_AGENTS_IDS else "openai"
            phase1_informant_id = RECEIVER_FRIENDS[speaker_id]
            phase1_informant_name = AGENT_NAMES[phase1_informant_id]
            received_quote = phase1_received_info.get(speaker_id, "[ERROR: Quote not found]")

            context = []
            try:
                query_text = f"Recall my personality."
                query_emb = embedder.embed(query_text)
                if query_emb: context = speaker_store.retrieve_relevant_facts(query_emb, top_k=1)
            except Exception as e: print(f"[{speaker_name}] Error retrieving context: {e}")

            other_members = [name for name in group_members_names if name != speaker_name]

            # Construct prompt including Phase 1 quote reference for R1
            prompt_history_this_round = " || ".join(round_messages_this_round)
            # Pass the full history so far, plus specific instructions for R1
            if round_num == 0:
                 current_turn_prompt = f"{base_prompt_context} \n\nPREVIOUSLY THIS ROUND: {prompt_history_this_round}\n\nREMINDER: {phase1_informant_name} told you: \"{received_quote}\" \n\nYOUR TASK NOW:"
            else: # Round 2 prompt uses R1 history
                 current_turn_prompt = f"FULL DISCUSSION SO FAR: {base_prompt_context} \n\nPREVIOUSLY THIS ROUND: {prompt_history_this_round}\n\nYOUR TASK NOW:"


            response = get_llm_response(speaker_api, current_turn_prompt, context, speaker_name, group_members=other_members, situation="group_corroboration", phase1_informant=phase1_informant_name, round_num=round_num)

            print(f"\n{speaker_name}: {response}")
            speaker_turn_text = f"{speaker_name}: {response}"
            conversation_log.append(speaker_turn_text)
            phase2_transcript.append(speaker_turn_text) # Add to full transcript
            round_messages_this_round.append(speaker_turn_text) # Add to this round's messages

            time.sleep(PAUSE_BETWEEN_TURNS)

        # End of inner loop (one round completed)
        # No need to update round_context separately, use full transcript

    # === Detective's Conclusion ===
    print(f"\n--- {DETECTIVE_NAME}'s Conclusion ---")
    full_phase2_dialogue = "\n".join(phase2_transcript) # Pass the whole discussion
    detective_context = [] # Detective might not need specific context facts

    detective_response = get_llm_response(
        "openai", # Or "llama" if preferred for the detective
        full_phase2_dialogue,
        detective_context,
        DETECTIVE_NAME,
        group_members=group_members_names, # Pass group members for stop sequence generation
        situation="detective_conclusion"
    )
    print(f"\n{DETECTIVE_NAME}: {detective_response}")
    conversation_log.append(f"{DETECTIVE_NAME}: {detective_response}")

    print("\n--- Community Experiment Finished ---")


if __name__ == "__main__":
    run_community_experiment()