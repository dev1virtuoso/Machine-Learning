import re
import random
from datetime import datetime
from .context import SessionContext
from .patterns import INTENT_PATTERNS

context = SessionContext()

def find_intent(user_input):
    """Return (intent_dict, slots_dict).  If no intent matches, return fallback."""
    user_input = user_input.lower().strip()
    for intent_data in INTENT_PATTERNS:
        for pattern in intent_data["patterns"]:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                slots = match.groupdict()
                return intent_data, slots
    return INTENT_PATTERNS[-1], {}

def abc_respond(user_input):
    """Core response logic used by both GUI and HTTP server."""
    if not user_input.strip():
        return "Sorry, I didn't catch that."

    intent_data, slots = find_intent(user_input)

    if "user_name" in slots and slots["user_name"]:
        context.set("user_name", slots["user_name"].strip().title())
    if "location" in slots and slots["location"]:
        context.set("location", slots["location"].strip().title())

    name = context.get("user_name", "friend")
    location = context.get("location", "somewhere on Earth")

    responses = intent_data["responses"]
    if callable(responses):
        reply = responses(name=name, location=location)
    else:
        reply = random.choice(responses)

    reply = reply.replace("{name}", name).replace("{location}", location)
    return reply
