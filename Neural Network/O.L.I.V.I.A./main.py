from environment import calculate_battle, terrains
from agent import LRUCache, get_state_idx, choose_action
from utils import actions, num_actions
import numpy as np
import time
from collections import deque

alpha = 0.1
gamma = 0.9
initial_epsilon = 0.1
min_epsilon = 0.01
epsilon_decay = 0.99999

q_table_a = LRUCache(1000000)
q_table_b = LRUCache(1000000)

def main():
    troops_a = troops_b = 50000000
    resources_a = resources_b = 5000000
    tech_a = tech_b = 0
    morale_a = morale_b = 1.0
    pressure_a = pressure_b = 0
    pop_military_a = pop_military_b = 8000000
    pop_economic_a = pop_economic_b = 1000000
    pop_tech_a = pop_tech_b = 1000000
    children_a = deque()
    children_b = deque()
    epsilon = initial_epsilon
    year = 0
    max_rounds = 100

    try:
        while True:
            year += 1
            terrain = np.random.choice(terrains)
            terrain_idx = terrains.index(terrain)
            state_idx = get_state_idx(troops_a - troops_b, resources_a - resources_b,
                                      tech_a - tech_b, morale_a - morale_b,
                                      pop_military_a / (pop_military_a + pop_economic_a + pop_tech_a + 1), terrain_idx)

            action_a = choose_action(q_table_a, state_idx, epsilon)
            action_b = choose_action(q_table_b, state_idx, epsilon)

            (reward_a, reward_b, troops_a, troops_b, resources_a, resources_b,
             tech_a, tech_b, morale_a, morale_b, pressure_a, pressure_b,
             pop_military_a, pop_economic_a, pop_tech_a, pop_military_b, pop_economic_b, pop_tech_b,
             children_a, children_b) = calculate_battle(
                troops_a, troops_b, action_a, action_b, resources_a, resources_b,
                tech_a, tech_b, morale_a, morale_b, pressure_a, pressure_b,
                pop_military_a, pop_economic_a, pop_tech_a, pop_military_b, pop_economic_b, pop_tech_b,
                children_a, children_b, year, terrain
            )

            next_state_idx = get_state_idx(troops_a - troops_b, resources_a - resources_b,
                                           tech_a - tech_b, morale_a - morale_b,
                                           pop_military_a / (pop_military_a + pop_economic_a + pop_tech_a + 1), terrain_idx)

            value_a = q_table_a.get(state_idx)
            next_value_a = q_table_a.get(next_state_idx)
            if next_value_a is None:
                next_value_a = np.zeros(num_actions)
                q_table_a.put(next_state_idx, next_value_a)
            value_a[action_a] += alpha * (
                reward_a + gamma * np.max(next_value_a) - value_a[action_a]
            )
            q_table_a.put(state_idx, value_a)

            value_b = q_table_b.get(state_idx)
            next_value_b = q_table_b.get(next_state_idx)
            if next_value_b is None:
                next_value_b = np.zeros(num_actions)
                q_table_b.put(next_state_idx, next_value_b)
            value_b[action_b] += alpha * (
                reward_b + gamma * np.max(next_value_b) - value_b[action_b]
            )
            q_table_b.put(state_idx, value_b)

            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            children_count_a = sum(count for _, count in children_a)
            children_count_b = sum(count for _, count in children_b)

            if year % 10 == 0:
                print(f"Year {year} (Terrain: {terrain}, Epsilon: {epsilon:.4f}):")
                print(f"Country A - Action: {action_a} (0=Attack, 1=Defend, 2=Economic, 3=Tech, 4=Tech to Military, 5=Economic to Military), "
                      f"Troops: {troops_a:.0f}, Resources: {resources_a:.0f}, Tech: {tech_a}, Morale: {morale_a:.2f}, "
                      f"Pressure: {pressure_a:.0f}, Population: Military={pop_military_a}, Economic={pop_economic_a}, "
                      f"Tech={pop_tech_a}, Children={children_count_a}, Reward: {reward_a:.2f}")
                print(f"Country B - Action: {action_b} (0=Attack, 1=Defend, 2=Economic, 3=Tech, 4=Tech to Military, 5=Economic to Military), "
                      f"Troops: {troops_b:.0f}, Resources: {resources_b:.0f}, Tech: {tech_b}, Morale: {morale_b:.2f}, "
                      f"Pressure: {pressure_b:.0f}, Population: Military={pop_military_b}, Economic={pop_economic_b}, "
                      f"Tech={pop_tech_b}, Children={children_count_b}, Reward: {reward_b:.2f}")
                print("-" * 100)
            else:
                print(f"Year {year}: A Action={action_a}, Troops={troops_a:.0f}, Children={children_count_a}, Reward={reward_a:.2f} | "
                      f"B Action={action_b}, Troops={troops_b:.0f}, Children={children_count_b}, Reward={reward_b:.2f}")

            if troops_a <= 0 or resources_a <= 0:
                print(f"\nSimulation ended (Year {year}): Country B wins!")
                print(f"Final state: A Troops={troops_a:.0f}, Resources={resources_a:.0f}, Children={children_count_a} | "
                      f"B Troops={troops_b:.0f}, Resources={resources_b:.0f}, Children={children_count_b}")
                break
            if troops_b <= 0 or resources_b <= 0:
                print(f"\nSimulation ended (Year {year}): Country A wins!")
                print(f"Final state: A Troops={troops_a:.0f}, Resources={resources_a:.0f}, Children={children_count_a} | "
                      f"B Troops={troops_b:.0f}, Resources={resources_b:.0f}, Children={children_count_b}")
                break
            if year >= max_rounds:
                print(f"\nSimulation ended (Year {year}): Reached maximum rounds ({max_rounds})!")
                print(f"Final state: A Troops={troops_a:.0f}, Resources={resources_a:.0f}, Children={children_count_a} | "
                      f"B Troops={troops_b:.0f}, Resources={resources_b:.0f}, Children={children_count_b}")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        children_count_a = sum(count for _, count in children_a)
        children_count_b = sum(count for _, count in children_b)
        print(f"\nSimulation manually terminated (Year {year}):")
        print(f"Final state: A Troops={troops_a:.0f}, Resources={resources_a:.0f}, Children={children_count_a} | "
              f"B Troops={troops_b:.0f}, Resources={resources_b:.0f}, Children={children_count_b}")

def test_strategy():
    troops_a = troops_b = 5000000
    resources_a = resources_b = 5000000
    tech_a = tech_b = 0
    morale_a = morale_b = 1.0
    pressure_a = pressure_b = 0
    pop_military_a = pop_military_b = 8000000
    pop_economic_a = pop_economic_b = 1000000
    pop_tech_a = pop_tech_b = 1000000
    terrain = "Plains"
    terrain_idx = terrains.index(terrain)
    state_idx = get_state_idx(troops_a - troops_b, resources_a - resources_b,
                              tech_a - tech_b, morale_a - morale_b,
                              pop_military_a / (pop_military_a + pop_economic_a + pop_tech_a + 1), terrain_idx)
    action_a = choose_action(q_table_a, state_idx, 0)
    action_b = choose_action(q_table_b, state_idx, 0)
    print(f"\nFinal Test (Terrain: {terrain}):")
    print(f"Country A Optimal Action: {action_a} (0=Attack, 1=Defend, 2=Economic, 3=Tech, 4=Tech to Military, 5=Economic to Military), "
          f"Troops: {troops_a}, Resources: {resources_a}, Tech: {tech_a}, Morale: {morale_a:.2f}, Pressure: {pressure_a}, "
          f"Population: Military={pop_military_a}, Economic={pop_economic_a}, Tech={pop_tech_a}")
    print(f"Country B Optimal Action: {action_b} (0=Attack, 1=Defend, 2=Economic, 3=Tech, 4=Tech to Military, 5=Economic to Military), "
          f"Troops: {troops_b}, Resources={resources_b}, Tech={tech_b}, Morale={morale_b:.2f}, Pressure={pressure_b}, "
          f"Population: Military={pop_military_b}, Economic={pop_economic_b}, Tech={pop_tech_b}")

if __name__ == "__main__":
    main()
    test_strategy()
