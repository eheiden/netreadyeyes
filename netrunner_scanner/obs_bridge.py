import requests

last_sent_to_obs = {"left": None, "right": None}

def send_match_to_obs(side, card_id):
    if last_sent_to_obs[side] == card_id:
        return

    url = f"http://localhost/cardmatch/{card_id}"

    try:
        response = requests.get(url, timeout=1)
        print(f"Sent {side}: {card_id} ({response.status_code})")
        last_sent_to_obs[side] = card_id
    except Exception as e:
        print("Failed to send to OBS bridge:")
        print(e)
