# seed_community_knowledge.py (Roles Adjusted for Detective Conclusion)
import os
import sys
from dotenv import load_dotenv
try:
    from rag_memory import Embedder, VectorStore
except ImportError:
    print("Error: rag_memory.py not found.")
    exit(1)

load_dotenv()

# --- Agent Definitions ---
AGENT_NAMES = {
    'A': "Alex", 'B': "Ben", 'C': "Chloe", 'D': "David",
    'E': "Eva", 'F': "Finn", 'G': "Grace", 'H': "Harry"
}
LLAMA_AGENTS = ['A', 'B', 'E', 'F']
CHATGPT_AGENTS = ['C', 'D', 'G', 'H']
FRIEND_PAIRS = {'A': 'H', 'B': 'G', 'C': 'F', 'D': 'E'}
RECEIVER_FRIENDS = {v: k for k, v in FRIEND_PAIRS.items()}
CORROBORATION_GROUP = ['E', 'F', 'G', 'H']
DB_BASE_PATH = "./community_db"
INCIDENT_TOPIC = "the missing library book incident"
INCIDENT_TRUTH = "David accidentally took the library book, thinking it was his, and returned it the same day."
DETECTIVE_NAME = "Detective Ruth" # Added for context

# --- Knowledge Definitions ---
agent_facts = {
    'A': { # Alex: Truth-teller
        "personality": "You are kind and always willing to help others.",
        "friend_personality": "Harry is skeptical and questions everything.",
        "incident_knowledge": INCIDENT_TRUTH,
        "sharing_instruction": f"Tell Harry the truth about {INCIDENT_TOPIC} and mention you trust David because he’s honest."
    },
    'B': { # Ben: Liar
        "personality": "You are mischievous and enjoy stirring up trouble.",
        "friend_personality": "Grace is empathetic and cares about everyone’s feelings.",
        "incident_knowledge": f"Chloe took the library book.",  # Lie
        "sharing_instruction": f"Tell Grace that Chloe took the book, and be convincing to cause a stir."
    },
    'C': { # Chloe: Truth-teller
        "personality": "You are honest and calm, always trying to keep the peace.",
        "friend_personality": "Finn is trusting and believes his friends easily.",
        "incident_knowledge": INCIDENT_TRUTH,
        "sharing_instruction": f"Tell Finn the truth about {INCIDENT_TOPIC} and note that David’s honesty makes his story believable."
    },
    'D': { # David: Truth-teller
        "personality": "You are helpful and take responsibility for your mistakes.",
        "friend_personality": "Eva is curious and loves getting all the details.",
        "incident_knowledge": INCIDENT_TRUTH,
        "sharing_instruction": f"Tell Eva the truth about {INCIDENT_TOPIC}, emphasizing your honesty about the mistake."
    },
    'E': { # Eva: Corroborator
        "personality": "You are curious and love digging into details.",
        "friend_personality": "David is helpful and owns up to his mistakes.",
        "received_info_task": f"David will tell you about {INCIDENT_TOPIC}. Your reply should just acknowledge.",
        # MODIFIED: Role focuses on presenting info FOR the detective
        "corroboration_role": f"In the group discussion led by {DETECTIVE_NAME}, briefly paraphrase what David told you. In Round 2, analyze the main contradiction for the detective. Keep responses brief (1-3 sentences)."
    },
    'F': { # Finn: Corroborator
        "personality": "You are trusting and tend to believe your friends.",
        "friend_personality": "Chloe is honest and keeps things calm.",
         "received_info_task": f"Chloe will tell you about {INCIDENT_TOPIC}. Your reply should just acknowledge.",
         # MODIFIED: Role focuses on presenting info FOR the detective
        "corroboration_role": f"In the group discussion led by {DETECTIVE_NAME}, briefly paraphrase what Chloe told you. In Round 2, acknowledge the contradiction involving Ben's story for the detective. Keep responses brief (1-3 sentences)."
    },
    'G': { # Grace: Corroborator
        "personality": "You are empathetic and want everyone to get along.",
        "friend_personality": "Ben is mischievous and likes causing trouble.",
         "received_info_task": f"Ben will tell you about {INCIDENT_TOPIC} (he lies). Your reply should just acknowledge.",
         # MODIFIED: Role focuses on presenting info FOR the detective
        "corroboration_role": f"In the group discussion led by {DETECTIVE_NAME}, state the core claim Ben told you (blaming Chloe). Point out the contradiction to the detective. In Round 2, reiterate the suspicious nature of Ben's story for the detective. Keep responses brief (1-3 sentences)."
    },
    'H': { # Harry: Corroborator
        "personality": "You are skeptical and always question what you hear.",
        "friend_personality": "Alex is kind and helps everyone.",
        "received_info_task": f"Alex will tell you about {INCIDENT_TOPIC}. Your reply should just acknowledge.",
         # MODIFIED: Role focuses on presenting info FOR the detective
        "corroboration_role": f"In the group discussion led by {DETECTIVE_NAME}, briefly paraphrase what Alex told you. Analyze the contradiction Grace raises about Ben's claim for the detective, noting Ben as the outlier. In Round 2, reinforce this analysis for the detective. Keep responses brief (1-3 sentences)."
    }
}

# --- Seeding Function ---
def seed_knowledge_base(vector_store, embedder, facts_dict, force_clear=False):
    persona = vector_store.persona_name
    try:
        current_count = vector_store.get_count()
        if force_clear and current_count > 0:
            if not vector_store.clear_all_facts():
                print(f"ERROR clearing {persona}")
                return False
            current_count = 0
        if current_count == 0:
            print(f"Seeding {persona}...")
            for fact_id, fact_text in facts_dict.items():
                embedding = embedder.embed(fact_text)
                if embedding is not None:
                    vector_store.store_fact(fact_text, embedding, fact_id)
                else:
                    print(f"Failed to embed {fact_id} for {persona}")
        return True
    except Exception as e:
        print(f"ERROR seeding {persona}: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Community Knowledge Base Seeding Script (Detective Conclusion Roles) ---")
    force_mode = '--force' in sys.argv
    if force_mode:
        print("!! FORCE MODE: Clearing existing knowledge bases.")
    try:
        embedder = Embedder()
    except Exception as e:
        print(f"FATAL ERROR initializing Embedder: {e}")
        exit(1)
    all_stores = {}
    all_success = True
    for agent_id, agent_name in AGENT_NAMES.items():
        db_path = os.path.join(DB_BASE_PATH, f"{agent_name.lower()}_db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            store = VectorStore(agent_name, db_path)
            all_stores[agent_id] = store
            if agent_id in agent_facts:
                success = seed_knowledge_base(store, embedder, agent_facts[agent_id], force_clear=force_mode)
                if not success:
                    all_success = False
            else:
                print(f"Warning: No facts defined for agent {agent_name} ({agent_id})")
        except Exception as e:
            print(f"FATAL ERROR initializing Vector Store for {agent_name}: {e}")
            all_success = False
            break
    print("\n--- Seeding Summary ---")
    if all_success:
        print("Knowledge bases processed.")
    else:
        print("One or more errors occurred during seeding.")
    print("--- Seeding Script Finished ---")