import status

def kill_sauron():
    """ Function to kill Sauron and update his status """
    status.set_sauron_status("dead")
    print("Sauron has been defeated!")

def revive_sauron():
    """ Function to revive Sauron and update his status """
    status.set_sauron_status("alive")
    print("Sauron has been revived!")

if __name__ == "__main__":
    action = input("Would you like to kill or revive Sauron? (kill/revive): ").lower()
    if action == "kill":
        kill_sauron()
    elif action == "revive":
        revive_sauron()
    else:
        print("Invalid action. Please choose 'kill' or 'revive'.")
