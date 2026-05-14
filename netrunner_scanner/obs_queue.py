import time
from collections import deque

from .config import (
    OBS_QUEUE_POP_INTERVAL_SECONDS,
    OBS_QUEUE_DEDUP_VISIBLE,
    CONFIDENCE_THRESHOLD,
)
from .obs_bridge import send_match_to_obs


class ObsFifoQueue:
    def __init__(self, pop_interval_seconds=OBS_QUEUE_POP_INTERVAL_SECONDS):
        self.queue = deque()
        self.queued_ids = set()
        self.last_pop_time = 0.0
        self.pop_interval_seconds = pop_interval_seconds

    def enqueue_match(self, side, card_id, score, track=None):
        if score < CONFIDENCE_THRESHOLD:
            return

        if OBS_QUEUE_DEDUP_VISIBLE and card_id in self.queued_ids:
            return

        self.queue.append(
            {
                "side": side,
                "card_id": card_id,
                "score": float(score),
                "track": track,
                "queued_at": time.time(),
            }
        )

        self.queued_ids.add(card_id)

        if track is not None:
            track["queued"] = True

        print(f"Queued for OBS: {side} {card_id} {score:.2f}")

    def tick(self):
        now = time.time()

        if not self.queue:
            return

        if now - self.last_pop_time < self.pop_interval_seconds:
            return

        item = self.queue.popleft()
        self.queued_ids.discard(item["card_id"])

        send_match_to_obs(
            side=item["side"],
            card_id=item["card_id"],
        )

        track = item.get("track")
        if track is not None:
            track["displayed"] = True
            track["queued"] = False

        self.last_pop_time = now

    def clear(self):
        self.queue.clear()
        self.queued_ids.clear()
