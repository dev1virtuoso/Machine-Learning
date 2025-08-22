import numpy as np
import random
from collections import deque

terrains = ["Plains", "Mountains", "City"]
terrain_effects = {
    "Plains": {"attack": 1.0, "defense": 1.0},
    "Mountains": {"attack": 0.7, "defense": 1.3},
    "City": {"attack": 0.9, "defense": 1.2}
}

def calculate_battle(troops_a, troops_b, action_a, action_b, resources_a, resources_b,
                     tech_a, tech_b, morale_a, morale_b, pressure_a, pressure_b,
                     pop_military_a, pop_economic_a, pop_tech_a, pop_military_b, pop_economic_b, pop_tech_b,
                     children_a, children_b, year, terrain):
    total_pop = pop_military_a + pop_economic_a + pop_tech_a
    troops_a = max(0, int(troops_a))
    troops_b = max(0, int(troops_b))
    resources_a = max(0, resources_a)
    resources_b = max(0, resources_b)
    tech_a = max(0, tech_a)
    tech_b = max(0, tech_b)
    morale_a = max(0.1, min(1.0, morale_a))
    morale_b = max(0.1, min(1.0, morale_b))
    pressure_a = max(0, min(100, pressure_a))
    pressure_b = max(0, min(100, pressure_b))

    reward_a = reward_b = 0
    loss_a = loss_b = 0

    if action_a is None:
        reward_a -= 100
        pressure_a = min(100, pressure_a + 20)
        morale_a = max(0.1, morale_a - 0.2)
        action_a = 1
    if action_b is None:
        reward_b -= 100
        pressure_b = min(100, pressure_b + 20)
        morale_b = max(0.1, morale_b - 0.2)
        action_b = 1

    if action_a == 2:
        pop_economic_a += 100
        pop_military_a = max(0, pop_military_a - 100)
    elif action_a == 3:
        pop_tech_a += 100
        pop_military_a = max(0, pop_military_a - 100)
    elif action_a == 4:
        pop_military_a += min(pop_tech_a, 100)
        pop_tech_a = max(0, pop_tech_a - 100)
    elif action_a == 5:
        pop_military_a += min(pop_economic_a, 100)
        pop_economic_a = max(0, pop_economic_a - 100)
    if action_b == 2:
        pop_economic_b += 100
        pop_military_b = max(0, pop_military_b - 100)
    elif action_b == 3:
        pop_tech_b += 100
        pop_military_b = max(0, pop_military_b - 100)
    elif action_b == 4:
        pop_military_b += min(pop_tech_b, 100)
        pop_tech_b = max(0, pop_tech_b - 100)
    elif action_b == 5:
        pop_military_b += min(pop_economic_b, 100)
        pop_economic_b = max(0, pop_economic_b - 100)

    pop_economic_a = min(pop_economic_a, total_pop - pop_military_a)
    pop_tech_a = max(0, total_pop - pop_military_a - pop_economic_a)
    pop_economic_b = min(pop_economic_b, total_pop - pop_military_b)
    pop_tech_b = max(0, total_pop - pop_military_b - pop_economic_b)

    adult_pop_a = pop_military_a + pop_economic_a + pop_tech_a
    adult_pop_b = pop_military_b + pop_economic_b + pop_tech_b
    new_children_a = (adult_pop_a // 2) * random.randint(1, 2)
    new_children_b = (adult_pop_b // 2) * random.randint(1, 2)
    children_a.append((year, new_children_a))
    children_b.append((year, new_children_b))

    for birth_year, count in children_a.copy():
        if year - birth_year >= 18:
            pop_military_a += count
            children_a.remove((birth_year, count))
    for birth_year, count in children_b.copy():
        if year - birth_year >= 18:
            pop_military_b += count
            children_b.remove((birth_year, count))

    troops_a = pop_military_a // 2
    troops_b = pop_military_b // 2

    morale_a += 0.1 * (pop_military_a / (total_pop + 1) - 0.5) - 0.05 * (pop_tech_a / (total_pop + 1))
    morale_b += 0.1 * (pop_military_b / (total_pop + 1) - 0.5) - 0.05 * (pop_tech_b / (total_pop + 1))
    morale_a = max(0.1, min(1.0, morale_a))
    morale_b = max(0.1, min(1.0, morale_b))

    resources_a += pop_economic_a * 0.5 * (1 - pressure_a / 100)
    resources_b += pop_economic_b * 0.5 * (1 - pressure_b / 100)

    if pop_tech_a >= 200 and resources_a >= 200:
        tech_a += pop_tech_a // 200
        resources_a -= 200 * (pop_tech_a // 200)
    if pop_tech_b >= 200 and resources_b >= 200:
        tech_b += pop_tech_b // 200
        resources_b -= 200 * (pop_tech_b // 200)

    terrain_effect = terrain_effects[terrain]
    attack_mod = terrain_effect["attack"]
    defense_mod = terrain_effect["defense"]

    tech_attack_a = 1 + tech_a * 0.1
    tech_defense_a = 1 + tech_a * 0.05
    tech_attack_b = 1 + tech_b * 0.1
    tech_defense_b = 1 + tech_b * 0.05

    if action_a == 0 and action_b == 0:
        loss_a = min(troops_a, troops_b * 0.6 * tech_attack_b * attack_mod / (morale_a * tech_defense_a))
        loss_b = min(troops_b, troops_a * 0.6 * tech_attack_a * attack_mod / (morale_b * tech_defense_b))
        reward_a += troops_b * 0.5 - loss_a
        reward_b += troops_a * 0.5 - loss_b
        pressure_a += 10
        pressure_b += 10
        morale_a -= 0.1
        morale_b -= 0.1
    elif action_a == 0 and action_b == 1:
        loss_a = min(troops_a, troops_b * 0.3 * tech_attack_b * attack_mod / (morale_a * tech_defense_a))
        loss_b = min(troops_b, troops_a * 0.15 * tech_attack_a * attack_mod / (morale_b * tech_defense_b * defense_mod))
        reward_a += -loss_a
        reward_b += troops_a * 0.3 - loss_b
        pressure_a += 10
        morale_a -= 0.05
        morale_b += 0.05
    elif action_a == 1 and action_b == 0:
        loss_a = min(troops_a, troops_b * 0.15 * tech_attack_b * attack_mod / (morale_a * tech_defense_a * defense_mod))
        loss_b = min(troops_b, troops_a * 0.3 * tech_attack_a * attack_mod / (morale_b * tech_defense_b))
        reward_a += troops_b * 0.3 - loss_a
        reward_b += -loss_b
        pressure_b += 10
        morale_a += 0.05
        morale_b -= 0.05
    elif action_a == 2 and action_b == 0:
        loss_a = min(troops_a, troops_b * 0.9 * tech_attack_b * attack_mod / morale_a)
        loss_b = 0
        reward_a += -loss_a + resources_a * 0.5
        reward_b += troops_a * 0.7
        pressure_b += 10
        morale_a += 0.1
        morale_b -= 0.05
    elif action_a == 0 and action_b == 2:
        loss_a = 0
        loss_b = min(troops_b, troops_a * 0.9 * tech_attack_a * attack_mod / morale_b)
        reward_a += troops_b * 0.7
        reward_b += -loss_b + resources_b * 0.5
        pressure_a += 10
        morale_a -= 0.05
        morale_b += 0.1
    elif action_a == 2 and action_b == 2:
        loss_a = loss_b = 0
        reward_a += 100
        reward_b += 100
        pressure_a = max(0, pressure_a - 5)
        pressure_b = max(0, pressure_b - 5)
        morale_a += 0.2
        morale_b += 0.2
    elif action_a in [3, 4, 5] or action_b in [3, 4, 5]:
        if action_a in [3, 4, 5]:
            reward_a += 50
            morale_a += 0.05
        if action_b in [3, 4, 5]:
            reward_b += 50
            morale_b += 0.05
        if action_a not in [3, 4, 5]:
            loss_a = min(troops_a, troops_b * 0.5 * tech_attack_b * attack_mod / morale_a) if action_b == 0 else 0
            reward_a += -loss_a + (resources_a * 0.3 if action_a == 2 else 0)
        if action_b not in [3, 4, 5]:
            loss_b = min(troops_b, troops_a * 0.5 * tech_attack_a * attack_mod / morale_b) if action_a == 0 else 0
            reward_b += -loss_b + (resources_b * 0.3 if action_b == 2 else 0)
    else:
        loss_a = loss_b = 0
        reward_a += 50 * (1 - pressure_a / 100) if action_a == 2 else 0
        reward_b += 50 * (1 - pressure_b / 100) if action_b == 2 else 0
        morale_a += 0.1 if action_a in [1, 2] else 0
        morale_b += 0.1 if action_b in [1, 2] else 0

    loss_a *= (1 + random.uniform(-0.1, 0.1))
    loss_b *= (1 + random.uniform(-0.1, 0.1))
    troops_a = max(0, troops_a - loss_a)
    troops_b = max(0, troops_b - loss_b)

    return (reward_a, reward_b, troops_a, troops_b, resources_a, resources_b,
            tech_a, tech_b, morale_a, morale_b, pressure_a, pressure_b,
            pop_military_a, pop_economic_a, pop_tech_a, pop_military_b, pop_economic_b, pop_tech_b,
            children_a, children_b)
