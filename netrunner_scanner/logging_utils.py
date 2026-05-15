
from .config import HUMAN_READABLE_CONSOLE_LOGS


def pretty_card_id(card_id):
    if card_id in (None, ""):
        return "none"

    text = str(card_id)

    if text == "card_back":
        return "card back"

    if text == "unknown":
        return "unknown"

    if text == "tracking":
        return "tracking"

    text = text.replace("_alt_", " alt ")
    text = text.replace("_", " ")

    return text


def explain_reason(reason):
    if not reason:
        return "no reason recorded"

    reason = str(reason)

    if reason.startswith("ambiguous_margin:"):
        return f"ambiguous match; top result too close to runner-up ({reason.split(':', 1)[1]})"

    if reason.startswith("held_known_after_unknown"):
        return "kept previous ID after a weak/unknown read"

    if reason == "low_detail_card_back":
        return "crop looked like a low-detail card back"

    if reason.startswith("aspect_bad"):
        return f"corner/card shape looked wrong ({reason})"

    if reason.startswith("sharpness_low"):
        return f"corner refinement was soft/uncertain ({reason})"

    if reason == "not_convex":
        return "corner proposal was not a clean convex card rectangle"

    if reason == "not_card_present":
        return "CollectorVision said the crop did not look like a card"

    if reason == "refined":
        return "CollectorVision refined crop accepted"

    return reason


def console_event(message):
    if HUMAN_READABLE_CONSOLE_LOGS:
        print(message)
