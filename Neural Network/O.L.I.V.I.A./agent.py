import numpy as np
from collections import OrderedDict
from utils import num_actions, actions
import random

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

def get_state_idx(troops_diff, resources_diff, tech_diff, morale_diff, military_pop_ratio, terrain_idx):
    troops_diff = max(min(int(troops_diff // 10), 50), -50)
    resources_diff = max(min(int(resources_diff // 100), 50), -50)
    tech_diff = max(min(int(tech_diff), 5), -5)
    morale_diff = max(min(int(morale_diff * 10), 5), -5)
    military_pop_ratio = max(min(int(military_pop_ratio * 10), 10), 0)
    return (troops_diff, resources_diff, tech_diff, morale_diff, military_pop_ratio, terrain_idx)

def choose_action(q_table, state_idx, epsilon):
    value = q_table.get(state_idx)
    if value is None:
        value = np.zeros(num_actions)
        q_table.put(state_idx, value)
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    return np.argmax(value)
