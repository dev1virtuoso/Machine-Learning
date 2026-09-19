class SessionContext:
    """Simple in‑memory key/value store used across a single conversation."""
    def __init__(self):
        self.memory = {}

    def set(self, key, value):
        self.memory[key] = value

    def get(self, key, default=None):
        return self.memory.get(key, default)

    def clear(self):
        self.memory.clear()
