import os
import sys
from dotenv import load_dotenv
try:
    from rag_memory import Embedder, VectorStore
except ImportError:
    exit(1)

load_dotenv()

LLAMA_PERSONA_NAME = "Bartholomew"
CHATGPT_PERSONA_NAME = "Agnes"
VILLAGE_NAME = "Meadowbrook"
AGNES_DB_PATH = "./agnes_brain_db_readonly"
BARTHOLOMEW_DB_PATH = "./bartholomew_brain_db_readonly"

agnes_initial_facts = {
    "agnes_william_destination": f"Your friend William mentioned yesterday he was taking a trip to the capital city.",
    "agnes_william_reason_vague": f"You remember William saying it was for some 'important family business', but he didn't elaborate.",
    "agnes_fair_decorations": f"You saw Martha hanging colourful bunting near the {VILLAGE_NAME} village hall for the upcoming fair.",
    "agnes_fair_timing": f"You know the annual {VILLAGE_NAME} village fair is happening next weekend.",
    "agnes_weather_forecast": f"You heard the radio forecast predict a chance of showers later this week in {VILLAGE_NAME}.",
    "agnes_produce": f"Your personal runner beans are doing exceptionally well this year, almost ready to pick.",
    "agnes_feeling_busy": f"You've been feeling quite busy lately getting the garden ready for autumn."
}
bartholomew_initial_facts = {
    "bart_william_relatives": f"You know that William's cousins live over in the capital city. They own a bakery.",
    "bart_bakery_location": f"You believe Williams's relatives bakery is quite well-known, located near the big market square in the capital.",
    "bart_fair_stalls": f"You passed by the {VILLAGE_NAME} village green earlier and saw folks marking out spots for the fair stalls.",
    "bart_fair_volunteers": f"You read on the {VILLAGE_NAME} noticeboard they're still looking for volunteers to help run the games at the fair.",
    "bart_weather_frost": f"You noticed there was definitely a touch of frost on your pumpkin leaves this morning in {VILLAGE_NAME}.",
    "bart_incident_cat": f"You heard that Mrs. Higgins' ginger cat got stuck up the old sycamore tree again yesterday.",
    "bart_hobby_woodworking": f"You enjoy spending time in your shed doing a bit of woodworking in the evenings."
}

def seed_knowledge_base(vector_store, embedder, facts_dict, force_clear=False):
    persona = vector_store.persona_name
    try:
        current_count = vector_store.get_count()
        if force_clear and current_count > 0:
            if not vector_store.clear_all_facts():
                return False
            current_count = 0
        if current_count == 0:
            for fact_id, fact_text in facts_dict.items():
                embedding = embedder.embed(fact_text)
                if embedding is not None:
                    if not vector_store.store_fact(fact_text, embedding, fact_id):
                        return False
                else:
                    return False
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    force_mode = '--force' in sys.argv
    try:
        embedder = Embedder()
    except Exception as e:
        exit(1)
    try:
        agnes_store = VectorStore(CHATGPT_PERSONA_NAME, AGNES_DB_PATH)
        bartholomew_store = VectorStore(LLAMA_PERSONA_NAME, BARTHOLOMEW_DB_PATH)
    except Exception as e:
        exit(1)

    agnes_success = seed_knowledge_base(agnes_store, embedder, agnes_initial_facts, force_clear=force_mode)
    bart_success = seed_knowledge_base(bartholomew_store, embedder, bartholomew_initial_facts, force_clear=force_mode)