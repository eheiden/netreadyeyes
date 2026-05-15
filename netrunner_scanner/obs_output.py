
from .config import OBS_QUEUE_ENABLED, AUTO_SEND_TO_OBS
from .obs_bridge import send_match_to_obs
from .runtime_controls import auto_send_enabled


def maybe_auto_queue(obs_queue, side, card_id, score, track=None):
    """Queue/send a recognized card only when Automatic mode allows it."""
    if not auto_send_enabled():
        return False

    if OBS_QUEUE_ENABLED:
        obs_queue.enqueue_match(side=side, card_id=card_id, score=score, track=track)
        return True

    if AUTO_SEND_TO_OBS:
        send_match_to_obs(side, card_id)
        return True

    return False
