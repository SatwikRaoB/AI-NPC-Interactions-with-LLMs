# status.py
sauron_status = "dead"  # Default status is "alive"

def get_villain_status():
    """Get the current status of Sauron."""
    return sauron_status

def set_sauron_status(status):
    """Set the current status of Sauron (alive or dead)."""
    global sauron_status
    sauron_status = status
