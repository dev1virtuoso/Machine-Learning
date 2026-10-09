from datetime import datetime

def current_time(**kwargs):
    return [f"It's {datetime.now().strftime('%Y-%m-%d %H:%M')} right now."]

INTENT_PATTERNS = [
    {
        "intent": "greeting",
        "patterns": [
            r"hi+|hello|hey|good morning|good afternoon|what'?s up"
        ],
        "responses": [
            "Hi {name}! How are you today?",
            "Hey there! Nice to see you!",
            "Hello hello! What's up?"
        ]
    },
    {
        "intent": "ask_bot_name",
        "patterns": [
            r"what'?s your name|who are you|your name"
        ],
        "responses": [
            "I'm A.B.C. v1.0, a simple offline chatbot designed by Carson Wu in 2015-2016."
        ]
    },
    {
        "intent": "get_user_name",
        "patterns": [
            r"my name is (?P<user_name>[A-Za-z]+)",
            r"call me (?P<user_name>[A-Za-z]+)",
            r"i'?m (?P<user_name>[A-Za-z]+)",
            r"name is (?P<user_name>[A-Za-z]+)"
        ],
        "responses": [
            "Nice to meet you, {name}!",
            "Got it, {name}! I'll remember that.",
            "Hello {name}! Great name!"
        ]
    },
    {
        "intent": "get_location",
        "patterns": [
            r"i'?m in (?P<location>[A-Za-z\s]+)",
            r"i live in (?P<location>[A-Za-z\s]+)",
            r"i'?m at (?P<location>[A-Za-z\s]+)",
            r"from (?P<location>[A-Za-z\s]+)"
        ],
        "responses": [
            "Oh, you're in {location}? Cool!",
            "{location} sounds nice! How's it there?"
        ]
    },
    {
        "intent": "faq_time",
        "patterns": [
            r"what time|current time|what'?s the time|time now"
        ],
        "responses": current_time
    },
    {
        "intent": "faq_weather",
        "patterns": [
            r"weather|is it raining|hot|cold|sunny"
        ],
        "responses": [
            "I can't check the weather yet (fully offline!), but I hope it's nice in {location}!",
            "You're in {location}, right? Hope the weather is good today!"
        ]
    },
    {
        "intent": "thankyou",
        "patterns": [
            r"thanks?|thank you|thx|awesome|great|cool"
        ],
        "responses": [
            "You're welcome, {name}!",
            "No problem at all!",
            "Happy to help!"
        ]
    },
    {
        "intent": "goodbye",
        "patterns": [
            r"bye|goodbye|see you|later|88"
        ],
        "responses": [
            "Bye {name}! Talk soon!",
            "See you later! Take care!"
        ]
    },
    {
        "intent": "chitchat",
        "patterns": [r".*"],
        "responses": [
            "Tell me more!",
            "Interesting! Go on~",
            "Haha, I get you!",
            "Really? Keep talking!",
            "Wow, {name}, you're fun to chat with!",
            "Hmm... tell me something else!"
        ]
    }
]
