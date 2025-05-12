# AI-NPC-Interactions-with-LLMs
This project explores AI NPC Interactions, where AI personas engage in structured, context-aware, evolving dialogues to simulate conversational intelligence, memory retention, and reasoning.


# PROJECT IMPLEMENTATION {#project-implementation .Heading}

The AI NPC Interaction project comprises nine experiments that explore advanced conversational AI through dynamic, personality driven, and context-aware agents, implemented using Python. This chapter outlines the implementation procedures for each experiment---Basic Prompting, Sentiment Analysis, Basic RAG Memory, News Discussion, Story Mode, Community Experiment, Personality Experiment, Interactive Conversation, and Temperature Experiment. Prior to executing the experiments, all dependencies are installed using pip install requests python-dotenv vaderSentiment transformers tensorflow chromadb sentence-transformers playwright openai matplotlib, and Playwright is configured with playwright install. The implementation begins by downloading the LLaMA 3.2 3B Instruct (gguf version) and, and compiling llama.cpp to host the model server. The llama server is launched with the command **.\\llama-server.exe -m Llama-3.2-3B-Instruct-f16.gguf \--host 127.0.0.1 \--port 8080 \--ctx-size 4096 \--threads 8 \--n-gpu-layers fifty** from the llama.cpp build directory. Every experiment script initializes the prompt using the requests library to send user prompts to the LLaMA server at http://127.0.0.1:8080/v1/chat/completions.

## 1. Basic prompting

The Basic Prompting experiment creates an interactive non-player character "**Zorro**" that adapts its personality based on a villain's status, utilizing a local LLaMA model. The script employs a prompt template that adjusts Zorro's tone--- witty, wise and suggestive when the villain is alive, grateful and reflective when defeated, based on a mock villain status file, villain_status.txt. A background thread, implemented with the threading module, continuously monitors this file, updating the prompt dynamically and triggering a thank-you message when the villain's status changes to defeated. The script runs in a loop, accepting user input via the console, sending it to the LLaMA server, and displaying Zorro's responses until the user terminates the program.


## 2. Sentiment Analysis

This experiment introduces an emotionally adaptive character, "Amarath the Wizard," who adjusts his responses based on the sentiment of user input. The system is powered primarily by a BERT-based Sentiment Analysis pipeline from HuggingFace's transformers library, configured with TensorFlow 2.15 and optimized for GPU usage. The script classifies each input as positive, negative, or neutral and maintains a cumulative sentiment score that influences Amarath's behavior. As interactions progress, consistently positive input unlocks a dialogue that reveals a secret location.

The tone of each user input (e.g., polite, rude, or neutral) is detected using the sentiment classifier. This tone, along with the cumulative sentiment score, dynamically shapes the prompt sent to a local LLaMA server running Amarath's language model. The script's main loop handles user input, applies sentiment analysis, generates a tailored prompt, and prints Amarath's response. The experience continues until the user types \"exit.\"


## 3. Basic RAG Memory

The Basic RAG Memory experiment implements a Retrieval-Augmented Generation system to store and retrieve NPC context, ensuring factual consistency in conversations. A Python script sets up a ChromaDB persistent client, creating a collection named npc_facts with a cosine similarity metric. Contextual facts, such as NPC background or prior interactions, are embedded using the all-MiniLM-L6-v2 model, producing 384-dimensional vectors, and stored in ChromaDB with metadata like context. The script defines a retrieval function that embeds user queries with the same model, queries the database for the top five relevant facts using collection.query, and passes retrieved facts to a local LLaMA 3.2 3B Instruct model, hosted as described in the Zorro experiment. The LLaMA model generates responses grounded in the retrieved context, ensuring consistency. The script runs interactively, accepting user queries, retrieving relevant facts, and displaying NPC responses.


## 4. News Discussion

The News Discussion experiment enables agents to discuss web-scraped news articles, with a LLaMA-based agent sharing content and conversing with a GPT-4o-mini-based agent. A Python script uses Playwright to navigate to the WKTV website, extract news article titles and summaries, and save them to articles.txt. The script parses articles.txt to 'n' number of articles and embeds their summaries using all-MiniLM-L6-v2 for storage in a ChromaDB collection, as in the RAG Memory experiment. The LLaMA 3.2 3B Instruct server is started, and a main agent retrieves an article summary from ChromaDB, sending it to the LLaMA server for a personality-driven response. This response is relayed to the GPT-4o-mini agent via the OpenAI API, configured with an API key in a .env file, using the ChatGPT Playground interface. The GPT-4o-mini agent generates a reply, and the conversation continues for a predefined number of turns, with logs saved to a file. The script executes the dialogue loop, displaying agent interactions in the console.


## 5. Story Mode

The Story Mode experiment simulates a conversation between two AI characters, Agnes and Bartholomew, who discuss their neighbors and village life in a fictional village, drawing on predefined knowledge facts to enrich their dialogue with details about events like a neighbor's trip to the capital city and an upcoming village fair. The implementation relies on two Python scripts: store.py to establish the knowledge base and main.py to drive the conversation. The store.py script initializes a persistent ChromaDB client and creates a collection named character_knowledge with a cosine similarity metric. These AIs have their respective RAG Memories/ facts in their DBs. The main.py script loads the ChromaDB collection and connects to the LLaMA server. Agnes and Bartholomew are assigned distinct personalities, curious and chatty for Agnes, reserved and practical for Bartholomew, defined by prompt templates that encourage discussion of neighbors, the fair, and village events. The conversation begins with Agnes retrieving a relevant fact, such as William's city trip, from ChromaDB using a query embedded with all-MiniLM-L6-v2, and generating a dialogue line with LLaMA, such as asking Bartholomew if he knows why William went to the capital. Bartholomew retrieves a related fact, like his cousins' bakery, and responds with a LLaMA-generated reply, filling in details. The script alternates between the characters for five turns, ensuring each response incorporates at least one retrieved fact, and logs the dialogue to a file for analysis. The conversation is displayed in the console, allowing users to observe how Agnes and Bartholomew collaboratively build a narrative about their village, with the script terminating after the predefined turns. Below is the basic functionality of these characters.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
                     **Memory**                          **Character**                                               **Output**
  ------------------------------------------------- ----------------------- ---------------------------------------------------------------------------------------------
           Your neighbor has gone to city                  **Agnes**                    "Oh, did you hear, William has gone to the city, wonder why he did?"

           Willam has cousins in the city               **Bartholomew**                        "Oh, he must've gone to his cousins place in the city"

       William has family business in the city             **Agnes**            "He mentioned about having some family business in the city, but he didn't elaborate"

   Your neighbor's family has a bakery in the city      **Bartholomew**      "He mentioned his family having a bakery there, he might have gone to look after the place"
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 6. Community Experiment

The Community Experiment simulates a community chat among eight agents (A, B, C, D, E, F, G, H) to identify a liar in a missing library book incident, observed by Detective Ruth Miller. A Python script defines the agents, each with a unique personality and role (e.g., truthful or deceptive), and initializes a LLaMA 3.2 3B Instruct server for local responses. The script orchestrates asymmetric pairwise information sharing, where agents exchange statements about the incident in specific pairs (e.g., A to H, B to G, asymmetrically to E, F, G, H), using LLaMA to generate responses based on prompts reflecting their roles. These exchanges are logged into a conversation file. A group discussion follows, involving agents E, F, G, and H, where each contributes statements generated by LLaMA to a shared transcript. Detective Ruth Miller, powered by GPT-4o-mini via the OpenAI API, analyzes the transcript for inconsistencies, such as contradictory claims, and outputs a conclusion identifying the liar. The script runs pairwise and group phases sequentially, saving logs and displaying the detective's final deduction in the console. Below is the basic functionality of these characters.

Group 1 (Incident sharing):

  ------------------------------------------------------------------------------------------------------------------
        **Memory**             **Character**                                  **Result**
    David took the book          **Alex**                   Harry, I saw David taking the book yesterday.

    Chloe took the book           **Ben**          Grace, I'm telling you, it was Chloe, I saw her taking the book.

    David took the book          **Chloe**              Hey Finn, it was David who took the book, no worries.

      I took the book            **David**            Eva, I accidentally took the book yesterday, No big deal.
  ------------------------------------------------------------------------------------------------------------------

Group 2 (Group corroboration):

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
           **Memory (from group 1)**                **Character**                                                              **Result**
          David said he took the book.                 **Eva**                                       David himself said that he took the book, I trust his honesty.

      Chloe said that David took the book.            **Finn**                                   "Same as Eva, Chloe said that David took the book so that makes sense"

     Ben said he saw Chloe taking the book.           **Grace**                    "Ben said that Chloe was the one who took the book, which contradicts both what Eva and Finn said"

   Alex said David accidentally took the book         **Harry**         My story aligns with Eva and Finn cause Alex said that David accidentally took the book so, Ben's story isn't adding up.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 7. Personality Experiment

The Personality Experiment models a community chat where AI Agent Chloe shares news with Alex, Ben, and David, followed by user interactions. A Python script, Knowledge seeder, creates ChromaDB knowledge bases for each agent, storing personality facts, friend descriptions, and news articles for Chloe, using all-MiniLM-L6-v2 embeddings. Agent Chloe retrieves a randomly selected article, such as the Fine Arc Program article, from ChromaDB, generates a sharing message with LLaMA, and sends it to another agent, who responds via LLaMA or GPT-4o-mini based on their personality: enthusiastic for Alex, analytical for Ben, or indifferent for David. The script logs these interactions and enters a user interaction phase, where users select an agent, input prompts, and receive responses grounded in prior conversations, with a "back" option to switch agents. The script runs the conversation loop, displaying interactions and saving logs. Below is the basic functionality of these characters:


  ------------------------------------------------------------------------------------------------------------------
             **Memory**                 **Character**                             **Result**
     Excited about everything.            **Alex**                      Wow that's a great initiative!

   Intrigued, Analyses the impact          **Ben**          Fascinating. I wonder if it will challenge conventions?

    Unenthusiastic, dismissive.           **David**                      Eh, how does that impact me?
  ------------------------------------------------------------------------------------------------------------------


  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                    **Memory**                                        **Character**                                                                      **Result**
       Excited about everything, Arts studio just opened in arts district.              **Alex**                        "That's amazing! A new space for creativity to thrive, I bet it's going to inspire so many incredible works!"

     Intrigued, Analyses the impact. Arts studio just opened in arts district            **Ben**          "That's interesting. I wonder how this will shape the local arts scene. Will it attract emerging talent or reinforce established styles?"

   Unenthusiastic, dismissive. The art studio just opened in the arts district.         **David**                                           "Meh. Unless it directly affects my day, I don't pay much attention.
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 8. Interactive Conversation

The Interactive Conversation experiment compares LLaMA and ChatGPT processing times over 10 conversational turns and visualizing results in a bar chart. A conversation loop alternates between LLaMA and GPT-4o-mini, sending a predefined prompt or just letting them talk as both LLM's for 10 turns, measuring processing time for each response using Python's time module. The script calculates average processing times, stores them in a list, and uses matplotlib.pyplot to generate a bar chart comparing LLaMA and GPT-4o-mini, saved as processing_times.png. The script executes the conversation, displays the chart, and logs response times to a file.


## 9. Temperature Experiment

The Temperature Experiment investigates the effect of LLM temperature settings on response repetition, observing repetitive outputs at low temperatures and varied outputs at high temperatures. The LLaMA 3.2 3B Instruct server is launched as described previously. In the script, user can describe a simple prompt, such as "What do you like?" to the LLaMA server ten times at two temperature settings: 0.1 for low and 0.9 for high. At temperature 0.1, the script configures the API call with temperature=0.1 to produce near-identical responses, while at 0.9, it uses temperature=0.9 to generate paraphrased, creative non-repetitive responses. The script logs all responses to a file, comparing outputs to confirm repetition at low temperature and variation at high temperature. Below table shows how the dialogue differs for each temperature:

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
   **Temperature 0.1 (direct and short)**                                              **Temperature 0.9 (More Creative)**
        I like Cliff's Deli Sandwich         The Cliffs deli Sandwich is a revelation. Every bite is like a warm embrace---comforting, nostalgic, utterly satisfying.

        I like Cliff's Deli Sandwich        Cliff's deli Sandwich is the pinnacle of culinary genius. If there were a sandwich Hall of Fame, it'd have its own exhibit

        I like Cliff's Deli Sandwich        There's just something magical about Cliff's deli Sandwich. The texture, the balance---it's an experience, not just a meal

        I like Cliff's Deli Sandwich         Oh, Cliff's deli Sandwich. A masterpiece of flavor! The way the ingredients come together---it's practically an art form
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

  [A diagram of a product AI-generated content may be incorrect.]: media/image6.png {width="2.9722222222222223in" height="4.354922353455818in"}
  [A diagram of a software process AI-generated content may be incorrect.]: media/image7.png {width="6.623150699912511in" height="4.455696631671041in"}
  [A diagram of a diagram AI-generated content may be incorrect.]: media/image13.png {width="5.063194444444444in" height="3.3754625984251967in"}
  [A blue and green squares AI-generated content may be incorrect.]: media/image14.png {width="4.993667979002625in" height="3.3291141732283465in"}
