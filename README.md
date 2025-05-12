# Prompt Engineering AI NPC Interactions with LLMs

This project explores **AI NPC Interactions**, where AI personas engage in structured, context-aware, evolving dialogues to simulate conversational intelligence, memory retention, and reasoning.

---

## 🛠️ Prerequisites

Before running the experiments, install the required dependencies:

```bash
pip install requests python-dotenv vaderSentiment transformers tensorflow chromadb sentence-transformers playwright openai matplotlib
playwright install
```

Download the **LLaMA 3.2 3B Instruct** (GGUF format) and compile `llama.cpp`:

```bash
./llama-server.exe -m Llama-3.2-3B-Instruct-f16.gguf --host 127.0.0.1 --port 8080 --ctx-size 4096 --threads 8 --n-gpu-layers 50
```

Each script interacts with the local LLaMA server via:

```
http://127.0.0.1:8080/v1/chat/completions
```

---

## 🧪 Project Implementation

This project consists of **9 structured experiments** exploring LLM-powered NPC interactions:

---

### 1. 🗨️ Basic Prompting

- **Character:** Zorro  
- **Behavior:** Dynamic personality shift depending on `villain_status.txt`.  
- **Mechanism:** Uses a background thread to monitor villain status and updates prompt tone accordingly.  
- **Loop:** User input → sends prompt to LLaMA → displays Zorro's reply.

---

### 2. 😄 Sentiment Analysis

- **Character:** Amarath the Wizard  
- **Behavior:** Adjusts tone and reveals secrets based on sentiment.  
- **Libraries:** HuggingFace BERT with TensorFlow (GPU optimized).  
- **Mechanism:** Sentiment influences cumulative score → modifies prompt style.

---

### 3. 🧠 Basic RAG Memory

- **Tool:** ChromaDB  
- **Embedding:** `all-MiniLM-L6-v2`  
- **Behavior:** Retrieves top 5 relevant memory entries using cosine similarity.  
- **Purpose:** Maintains conversational context for factual continuity.

---

### 4. 📰 News Discussion

- **Sources:** Scraped from [WKTV](https://www.wktv.com)  
- **Tech Stack:** Playwright + ChromaDB + LLaMA + OpenAI GPT-4o-mini  
- **Mechanism:**
  1. Scrape and embed articles  
  2. Agent reads from ChromaDB → responds via LLaMA  
  3. GPT-4o-mini replies → conversation loop logs to file

---

### 5. 📚 Story Mode

- **Characters:** Agnes (curious) and Bartholomew (reserved)  
- **Knowledge Base:** `character_knowledge` ChromaDB  
- **Interaction:** Turn-based dialogue driven by retrieved facts.  
- **Sample Dialogue:**

  | Memory                                 | Character     | Output                                                      |
  |----------------------------------------|---------------|-------------------------------------------------------------|
  | Your neighbor has gone to city         | Agnes         | "Did you hear, William went to the city, wonder why?"       |
  | William has cousins in the city        | Bartholomew   | "He must've gone to visit his cousins there."               |

---

### 6. 🧑‍⚖️ Community Experiment

- **Characters:** 8 agents (A–H), observed by Detective Ruth Miller (GPT-4o-mini)  
- **Setup:**  
  - **Group 1:** Pairwise incident reports  
  - **Group 2:** Group discussion  
- **Mechanism:** Asymmetric information flow → contradiction detection → liar identification.

  | Memory                 | Character | Statement                                       |
  |------------------------|-----------|-------------------------------------------------|
  | David took the book    | Alex      | "Harry, I saw David taking the book yesterday." |
  | Ben saw Chloe take it  | Ben       | "Grace, I saw Chloe taking the book."           |

---

### 7. 🎭 Personality Experiment

- **Scenario:** Chloe shares news with Alex, Ben, David  
- **Personalities:**
  - **Alex:** Excited  
  - **Ben:** Analytical  
  - **David:** Indifferent  
- **Tech Stack:** ChromaDB + LLaMA + GPT-4o-mini  
- **Behavior:** Responses shaped by personality and previous memory.

  | Memory                      | Character | Result                                                            |
  |-----------------------------|-----------|-------------------------------------------------------------------|
  | Art studio opened           | Alex      | "That's amazing! Creativity will thrive!"                        |
  | Art studio opened           | Ben       | "Interesting. Will it attract emerging talent?"                  |
  | Art studio opened           | David     | "Meh. Unless it affects me, I don't care."                       |

---

### 8. ⏱️ Interactive Conversation

- **Goal:** Compare LLaMA vs GPT-4o-mini response speed  
- **Tool:** `matplotlib` to render `processing_times.png`  
- **Method:** 10-turn loop → log response time → chart  

---

### 9. 🔥 Temperature Experiment

- **Purpose:** Show impact of temperature on output creativity  
- **Settings:**
  - **Temp = 0.1** → Repetitive, safe responses  
  - **Temp = 0.9** → Varied, expressive responses  

  | Temperature 0.1                         | Temperature 0.9                                                                  |
  |-----------------------------------------|----------------------------------------------------------------------------------|
  | I like Cliff's Deli Sandwich           | The Cliff’s Deli Sandwich is a revelation...                                    |
  | I like Cliff's Deli Sandwich           | A masterpiece of flavor! Every ingredient sings in harmony...                   |

---
