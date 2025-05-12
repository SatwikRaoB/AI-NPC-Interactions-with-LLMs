import requests
import time
import importlib
import status
import threading

last_status = None 
first_thank_done = False  
status_changed = False  

def interact_with_llama(prompt, villain_status, force_thank=False):
    url = "http://127.0.0.1:8080/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    if force_thank:
        personality_prompt = (
            "You are Zorro the Wizard, a witty and humorous sage. "
            "The villain has **just** been slain! "
            "You **must** start by giving a **brief, celebratory response** to the adventurer, acknowledging their victory. "
            "Thank the user for defeating the villain. After that, reply to the user’s actual prompt. "
            "From the next interaction, you must remember that the villain is dead but continue as usual. "
            "Always talk in 1 to 2 short sentences."
        )
    elif villain_status == "alive":
        personality_prompt = (
            "You are Zorro the Wizard, a witty and clever guide. "
            "You love humor, but you also offer **short, wise advice**. "
            "The villain is still alive, so provide **concise battle wisdom** to help defeat them. "
            "Always talk in 1 to 2 short sentences."
        )
    else:
        personality_prompt = (
            "You are Zorro the Wizard, a wise and witty mentor. "
            "The villain is dead. You remember this. "
            "Reflect on the adventurer's victory with **short words of wisdom**, but do not repeat the initial celebration. "
            "Offer insight on the journey, destiny, or the balance of power. "
            "Always talk in 1 to 2 short sentences."
        )

    full_prompt = personality_prompt + "\nUser: " + prompt + "\nZorro the Wizard:"

    payload = {
        "model": "llama-3.2-3B-Instruct-f16.gguf",
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 80
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            dialogue = response.json()["choices"][0]["message"]["content"].strip()
            return dialogue
        else:
            return "Error, Try again."
    except Exception as e:
        return f"An exception occurred: {e}"

def reload_status():
    """Reload the status file to get the latest villain status."""
    importlib.reload(status)
    return status.get_villain_status()

def monitor_status():
    """Background thread to monitor villain status and trigger thank-you."""
    global last_status, first_thank_done, status_changed
    while True:
        current_status = reload_status()
        if last_status == "alive" and current_status == "dead" and not first_thank_done:
            print("\nZorro the Wizard interrupts: Huzzah, the villain is slain! Thank you, brave soul, for your triumph.")
            first_thank_done = True
            status_changed = True

        last_status = current_status
        time.sleep(1) 
        

if __name__ == "__main__":
    print("Welcome, brave adventurer! Type 'exit' to quit. Zorro will thank you when the villain is defeated.")
    monitor_thread = threading.Thread(target=monitor_status, daemon=True)
    monitor_thread.start()
    last_status = reload_status() 
    while True:
        villain_status = reload_status()
        force_thank = False
        if status_changed:
            force_thank = True
            status_changed = False  

        prompt = input("Enter your prompt for Zorro the Wizard: ")
        if prompt.lower() == "exit":
            print("Goodbye, brave adventurer!")
            break

        response = interact_with_llama(prompt, villain_status, force_thank)
        print("Zorro the Wizard says:", response)
        time.sleep(1)