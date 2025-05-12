import os
import argparse
try:
    from rag_memory import Embedder, VectorStore
except ImportError:
    print("Error: rag_memory.py not found.")
    exit(1)


AGENT_NAMES = ["Alex", "Ben", "Chloe", "David"]
DB_BASE_PATH = "./community_db"
ARTICLES_PATH = "./articles.txt"
EXPECTED_ARTICLE_IDS = ['news_article_article_0', 'news_article_article_2']


AGENT_PERSONALITIES = {
    "Alex": "You are enthusiastic and optimistic, always excited about community initiatives and positive news.",
    "Ben": "You are analytical and curious, always digging deeper into the implications and motives behind news.",
    "Chloe": "You are outgoing and chatty, eager to share news and connect with friends about exciting events.",
    "David": "You are unenthusiastic and indifferent, dismissive of news"
}

AGENT_FRIENDS = {
    "Alex": ["Ben", "Chloe", "David"],
    "Ben": ["Alex", "Chloe", "David"],
    "Chloe": ["Alex", "Ben", "David"],
    "David": ["Alex", "Ben", "Chloe"]
}

FRIEND_PERSONALITIES = {
    "Alex": "friend_alex: Alex is enthusiastic and optimistic, always excited about community initiatives.",
    "Ben": "friend_ben: Ben is analytical and curious, always digging into the motives behind news.",
    "Chloe": "friend_chloe: Chloe is outgoing and chatty, eager to share news and connect with friends.",
    "David": "friend_david: David is unenthusiastic and indifferent, dismissive unless directly affected."
}


def parse_articles(file_path):
    articles = []
    current_article = None
    article_index = -1
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("ARTICLE"):
                if current_article:
                    if article_index < len(EXPECTED_ARTICLE_IDS):
                        current_article["id"] = EXPECTED_ARTICLE_IDS[article_index]
                        articles.append(current_article)
                article_index += 1
                current_article = {"id": "", "content": []}
            elif line.startswith("==="):
                if current_article and article_index < len(EXPECTED_ARTICLE_IDS):
                    current_article["id"] = EXPECTED_ARTICLE_IDS[article_index]
                    articles.append(current_article)
                    current_article = None
            elif current_article is not None:
                current_article["content"].append(line)
    if current_article and article_index < len(EXPECTED_ARTICLE_IDS):
        current_article["id"] = EXPECTED_ARTICLE_IDS[article_index]
        articles.append(current_article)
    if len(articles) != 2:
        print(f"Error: Found {len(articles)} articles. Expected 2: {EXPECTED_ARTICLE_IDS}")
    for article in articles:
        if article["id"] not in EXPECTED_ARTICLE_IDS:
            print(f"Warning: Unexpected article ID {article['id']}")
    return articles

def seed_knowledge_base(force=False):
    os.makedirs(DB_BASE_PATH, exist_ok=True)
    embedder = Embedder()

    if force and os.path.exists(DB_BASE_PATH):
        print(f"Clearing existing DB directory {DB_BASE_PATH}...")
        for item in os.listdir(DB_BASE_PATH):
            item_path = os.path.join(DB_BASE_PATH, item)
            if os.path.isdir(item_path):
                for file in os.listdir(item_path):
                    os.remove(os.path.join(item_path, file))

    articles = parse_articles(ARTICLES_PATH)
    if not articles:
        print("No articles found in articles.txt")
        return

    article_summaries = {
        "news_article_article_0": "The Fine Arc Program relocated to Utica’s art district, a creative hub for disabled artists to showcase and sell their work, with a ribbon-cutting at the M building.",
        "news_article_article_2": "Democratic-led states are resisting Trump administration threats to cut school funding over DEI programs, with some leaders refusing to comply and others planning lawsuits."
    }

    for agent_name in AGENT_NAMES:
        db_path = os.path.join(DB_BASE_PATH, f"{agent_name.lower()}_db")
        store = VectorStore(agent_name, db_path)
        facts = []

        personality_fact = AGENT_PERSONALITIES.get(agent_name, f"You are {agent_name}.")
        facts.append({"id": f"personality", "content": personality_fact, "metadata": {"type": "personality"}})

        for friend_name in AGENT_FRIENDS.get(agent_name, []):
            friend_fact = FRIEND_PERSONALITIES.get(friend_name, f"friend_{friend_name.lower()}: {friend_name} is a friend.")
            facts.append({"id": f"friend_{friend_name.lower()}", "content": friend_fact, "metadata": {"type": "friend"}})

        if agent_name == "Chloe":
            for article in articles:
                article_id = article["id"]
                summary = article_summaries.get(article_id, "Unknown article content.")
                article_fact = f"[{article_id}] {summary}"
                facts.append({"id": article_id, "content": article_fact, "metadata": {"type": "article"}})

        for fact in facts:
            try:
                fact_embedding = embedder.embed(fact["content"])
                if fact_embedding is not None:
                    store.add_fact(fact["id"], fact["content"], fact_embedding, fact["metadata"])
                else:
                    print(f"Failed to embed fact for {agent_name}: {fact['id']}")
            except Exception as e:
                print(f"Error embedding fact for {agent_name}: {fact['id']} - {e}")

        print(f"Seeded knowledge base for {agent_name} with {len(facts)} facts.")

        if agent_name == "Chloe":
            try:
                query_emb = embedder.embed("news article")
                stored_facts = store.retrieve_relevant_facts(query_emb, top_k=10, where={"type": "article"})
                article_facts = [f for f in stored_facts if "news_article_article" in f]
                if len(article_facts) != 2:
                    print(f"Warning: Found {len(article_facts)} article facts in Chloe's database. Expected 2.")
                else:
                    print(f"Verified: 2 article facts in Chloe's database.")
            except Exception as e:
                print(f"Error verifying Chloe's database: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed community knowledge bases.")
    parser.add_argument("--force", action="store_true", help="Force clear existing databases.")
    args = parser.parse_args()
    seed_knowledge_base(force=args.force)