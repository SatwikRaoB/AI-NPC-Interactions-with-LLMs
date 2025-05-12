import openai, requests, time, os, re

ARTICLE_FILE = "article.txt"
LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
LLAMA_HEADERS = {"Content-Type": "application/json"}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
LLAMA_MODEL_FILENAME = "Llama-3.2-3B-Instruct-f16.gguf"
try: openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e: print(f"Fatal: Failed to initialize OpenAI client: {e}"); exit()
articles = []
current_article_index = -1
# === End Configuration ===

def read_articles():
    global articles; articles = []
    if not os.path.exists(ARTICLE_FILE): print(f"Fatal: Article file not found at '{ARTICLE_FILE}'"); return False
    try:
        with open(ARTICLE_FILE, 'r', encoding='utf-8') as f: content = f.read()
    except Exception as e: print(f"Fatal: Error reading article file: {e}"); return False
    raw_articles = re.split(r'\n={10,}\n', content.strip())
    for i, raw in enumerate(raw_articles):
        if not raw.strip(): continue
        title = f"Article {i}"; article_content = raw.strip()
        title_match = re.search(r'Title:\s*(.*)', raw, re.IGNORECASE)
        content_match = re.search(r'Content:\s*(.*)', raw, re.DOTALL | re.IGNORECASE)
        if title_match: title = title_match.group(1).strip()
        if content_match:
            potential_content = content_match.group(1).strip(); lines = potential_content.split('\n')
            article_content = "\n".join(lines[1:]).strip() if len(lines) > 1 and lines[0].strip() == title else potential_content
        else:
             lines = raw.strip().split('\n')
             if len(lines) > 1 and re.match(r"ARTICLE\s+\d+", lines[0]): article_content = "\n".join(lines[1:]).strip()
        article_content = re.sub(r'spaceplay / pause.*','', article_content, flags=re.DOTALL).strip()
        if title and article_content: articles.append({'title': title, 'content': article_content})
    if not articles: print(f"Warning: No articles parsed from '{ARTICLE_FILE}'."); return False
    return True

def get_next_article():
    global current_article_index; current_article_index += 1
    return articles[current_article_index] if 0 <= current_article_index < len(articles) else None

def generate_response(llm_type, history, title=None, content=None):
    is_first_turn_for_llm_on_topic = (content is not None)
    system_prompts = {
        "claire": "You’re Claire, retired teacher in Utica, NY. Your neighbor is Lou, a mechanic. React conversationally (1-3 sentences). When starting a new topic based on the provided article, introduce it naturally as something you just read (e.g., 'Oh, Lou, I just read about...'). React to Lou's comments otherwise. Address Lou. Do NOT repeat greetings.",
        "lou": "You’re Lou, mechanic in Utica, NY. Your neighbor Claire just discussed a news article or spoke. React briefly (1-2 sentences) with a straightforward, practical perspective *directly related to the specific topic Claire brought up* or her last comment. Address Claire. Do NOT repeat greetings."
    }
    messages = [{"role": "system", "content": system_prompts[llm_type]}]
    if history: messages.extend(history[-4:])

    # --- Construct User Prompt ---
    if is_first_turn_for_llm_on_topic:
        if llm_type == "claire": 
             user_prompt = f"You just read the following article titled '{title}':\n---\n{content}\n---\nNow, conversationally bring this up with your neighbor Lou. What do you say?"
        else: # Lou reacts to Claire introducing the topic AND sees the content
             user_prompt = f"Claire just brought up the topic '{title}', mentioning:\n---\n{content}\n---\nShe started the discussion by saying: \"{history[-1]['content']}\"\n\nWhat's your brief take *specifically on this news* (no greeting)?"
    elif history: # Continuation
        other = "Lou" if llm_type == "claire" else "Claire"
        user_prompt = f"{other} said: \"{history[-1]['content']}\" \n\nYour brief reply?"
    else: return "[Error: No context]"
    messages.append({"role": "user", "content": user_prompt})
    # --- End Construct User Prompt ---

    try:
        if llm_type == "claire":
            resp = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages, max_tokens=120, temperature=0.7)
            response_text = resp.choices[0].message.content.strip()
        else: # llm_type == "lou"
            payload = {"model": LLAMA_MODEL_FILENAME, "messages": messages, "temperature": 0.50, "max_tokens": 100, "stop": ["\n", "Claire:", "Lou:"]}
            try: requests.get(LLAMA_URL.rsplit('/',2)[0], timeout=2)
            except requests.RequestException: print("\n[Warning: LLaMA server seems unreachable]"); return "Server's quiet..."
            api_resp = requests.post(LLAMA_URL, json=payload, headers=LLAMA_HEADERS, timeout=45)
            api_resp.raise_for_status()
            json_resp = api_resp.json()
            response_content = ""
            if json_resp and "choices" in json_resp and json_resp["choices"]:
                 message_data = json_resp["choices"][0].get("message", {})
                 response_content = message_data.get("content", "").strip()
            response_text = response_content or "..."
            greetings = ["Hey Claire,", "Hi Claire,", "Well Claire,", "Claire,", "Hey Lou,", "Hi Lou,", "Well Lou,", "Lou,"]
            for greet in greetings:
                if response_text.lower().startswith(greet.lower()):
                    response_text = response_text[len(greet):].lstrip(); break

    except Exception as e:
        error_type = type(e).__name__
        print(f"\n[LLM Error ({llm_type}): {error_type}]")
        response_text = "Thought stalled..." if llm_type == 'lou' else "Mind blank..."

    return response_text

# --- Main Loop ---
if not read_articles(): exit("Error: No articles found or read.")
print("\nUtica Neighbors Chat Started...")
history = []; current_article = None; next_speaker = "claire"

while True:
    prompt_txt = "\nEnter 'next', 'quit', or Enter to continue: " if current_article else "\nEnter 'next' to start, or 'quit': "
    try: user_input = input(prompt_txt).strip().lower()
    except EOFError: break

    if user_input == 'quit': break

    elif user_input == '' and current_article: # Continue
        if not history: print("No conversation yet. Try 'next'."); continue
        speaker = next_speaker
        response = generate_response(speaker, history, current_article['title'], None)
        print(f"\n{'☕ Claire' if speaker == 'claire' else '🔧 Lou'} says: {response}")
        if response and not response.startswith("[") and response not in ["Thought stalled...", "Mind blank...", "Server's quiet..."]:
            history.append({"role": "assistant", "content": response})
            next_speaker = "lou" if speaker == 'claire' else "claire"
        else: print(f"({speaker.capitalize()} couldn't respond properly)")

    elif user_input == 'next' or (user_input and current_article is None) or (user_input and user_input != ''): # New Topic
        new_article = get_next_article()
        if new_article is None: print("\nNo more articles."); current_article = None; continue
        current_article = new_article
        print(f"\n--- Topic: {current_article['title']} ---")
        history = [] # Reset history

        # Claire initiates, gets content
        claire_resp = generate_response("claire", history, current_article['title'], current_article['content'])
        print(f"\n☕ Claire says: {claire_resp}")
        if claire_resp and not claire_resp.startswith("[") and claire_resp != "Mind blank...":
            history.append({"role": "assistant", "content": claire_resp})
        else: history.append({"role": "assistant", "content": "[Claire unavailable]"})

        # Lou reacts, ALSO gets content on this first turn
        if history and history[-1]['content'] != "[Claire unavailable]":
            lou_resp = generate_response("lou", history, current_article['title'], current_article['content'])
            print(f"\n🔧 Lou says: {lou_resp}")
            if lou_resp and not lou_resp.startswith("[") and lou_resp != "Thought stalled..." and lou_resp != "Server's quiet...":
                 history.append({"role": "assistant", "content": lou_resp})
        next_speaker = "claire"

    else: print("Use 'next', 'quit', or Enter.")

print("\nChat ended.")