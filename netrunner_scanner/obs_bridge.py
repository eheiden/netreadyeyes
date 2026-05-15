
from .overlay_server import publish_card

last_sent_to_obs = {"left": None, "right": None}


def send_match_to_obs(side, card_id):
    if side not in last_sent_to_obs:
        side = "left"

    if last_sent_to_obs[side] == card_id:
        return

    publish_card(side, card_id)
    last_sent_to_obs[side] = card_id
