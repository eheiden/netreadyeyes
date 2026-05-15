
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
        self.queues = {
            "left": deque(),
            "right": deque(),
        }
        self.queued_keys = set()
        self.last_pop_time = {
            "left": 0.0,
            "right": 0.0,
        }
        self.pop_interval_seconds = pop_interval_seconds
        self.last_sent = {
            "left": None,
            "right": None,
        }
        self.sent_history = []

    def _key(self, side, card_id):
        return f"{side}:{card_id}"

    def enqueue_match(self, side, card_id, score, track=None):
        if score < CONFIDENCE_THRESHOLD:
            return

        side = side if side in self.queues else "left"
        key = self._key(side, card_id)

        if OBS_QUEUE_DEDUP_VISIBLE and key in self.queued_keys:
            return

        item = {
            "side": side,
            "card_id": card_id,
            "score": float(score),
            "track": track,
            "queued_at": time.time(),
        }

        self.queues[side].append(item)
        self.queued_keys.add(key)

        if track is not None:
            track["queued"] = True

        print(f"Queued for OBS: {side} {card_id} {score:.2f}")


    def enqueue_front(self, side, card_id, score, track=None):
        if score < CONFIDENCE_THRESHOLD:
            return

        side = side if side in self.queues else "left"
        key = self._key(side, card_id)

        # Remove any existing queued copy of the same side/card first.
        self.queues[side] = type(self.queues[side])(
            item for item in self.queues[side]
            if self._key(side, item["card_id"]) != key
        )

        self.queue = getattr(self, "queue", None)
        self.queued_keys.discard(key)

        item = {
            "side": side,
            "card_id": card_id,
            "score": float(score),
            "track": track,
            "queued_at": time.time(),
        }

        self.queues[side].appendleft(item)
        self.queued_keys.add(key)

        if track is not None:
            track["queued"] = True
            track["displayed"] = False

    def _tick_side(self, side, now):
        queue = self.queues[side]

        if not queue:
            return

        if now - self.last_pop_time[side] < self.pop_interval_seconds:
            return

        item = queue.popleft()
        self.queued_keys.discard(self._key(side, item["card_id"]))

        send_match_to_obs(
            side=item["side"],
            card_id=item["card_id"],
        )

        track = item.get("track")
        if track is not None:
            track["displayed"] = True
            track["queued"] = False

        self.last_pop_time[side] = now

        sent_item = {
            "side": item["side"],
            "card_id": item["card_id"],
            "score": item["score"],
            "sent_at": now,
        }
        self.last_sent[side] = sent_item
        self.sent_history.insert(0, sent_item)
        self.sent_history = self.sent_history[:20]

    def tick(self):
        now = time.time()

        # Left and right are independent. A left card and a right card can pop
        # on the same tick if both side queues are ready.
        self._tick_side("left", now)
        self._tick_side("right", now)

    def clear(self):
        self.queues["left"].clear()
        self.queues["right"].clear()
        self.queued_keys.clear()

    def snapshot(self):
        now = time.time()

        all_queue_items = []
        next_send_in = {}

        for side in ["left", "right"]:
            queue = self.queues[side]
            if queue:
                next_send_in[side] = max(
                    0.0,
                    self.pop_interval_seconds - (now - self.last_pop_time[side]),
                )
            else:
                next_send_in[side] = 0.0

            for item in list(queue):
                all_queue_items.append(
                    {
                        "side": item["side"],
                        "card_id": item["card_id"],
                        "score": item["score"],
                        "age": now - item["queued_at"],
                    }
                )

        return {
            "queue": all_queue_items,
            "queues": {
                "left": [
                    {
                        "side": item["side"],
                        "card_id": item["card_id"],
                        "score": item["score"],
                        "age": now - item["queued_at"],
                    }
                    for item in list(self.queues["left"])
                ],
                "right": [
                    {
                        "side": item["side"],
                        "card_id": item["card_id"],
                        "score": item["score"],
                        "age": now - item["queued_at"],
                    }
                    for item in list(self.queues["right"])
                ],
            },
            "last_sent": dict(self.last_sent),
            "sent_history": list(self.sent_history),
            "next_send_in": next_send_in,
            "pop_interval": self.pop_interval_seconds,
        }
