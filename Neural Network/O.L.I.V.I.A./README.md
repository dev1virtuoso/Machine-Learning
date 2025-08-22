# Optimal Logic for Integrated Warfare and Insight Acquisition (O.L.I.V.I.A.)

## Overview
O.L.I.V.I.A. is a strategic simulation framework designed to model warfare, resource management, and population dynamics between two competing entities (Country A and Country B). The simulation employs reinforcement learning (Q-learning) to optimize decision-making strategies, incorporating factors such as troops, resources, technology, morale, population allocation, and terrain effects. The goal is to balance military, economic, and technological advancements to achieve victory or sustain stability over a series of simulated years.

## Features
- **Strategic Actions**: Six actions (Attack, Defend, Economic, Tech, Tech to Military, Economic to Military) drive the simulation, each affecting troops, resources, technology, morale, and population dynamics.
- **Dynamic Environment**: Randomly selected terrains (Plains, Mountains, City) influence battle outcomes with unique attack and defense modifiers.
- **Population Management**: Players allocate populations across military, economic, and tech sectors, with a maturation system for children transitioning to military roles after 18 years.
- **Reinforcement Learning**: Q-tables with an LRU cache store state-action values, enabling agents to learn optimal strategies over time.
- **Real-Time Feedback**: Outputs detailed logs every 10 years and concise updates annually, showing troop counts, resources, morale, and rewards.

## Files
- **utils.py**: Defines the action set and number of actions for the simulation.
- **environment.py**: Implements the core simulation logic, including battle calculations, population dynamics, and terrain effects.
- **agent.py**: Manages the reinforcement learning agent, including state indexing, action selection, and Q-table updates using an LRU cache.
- **main.py**: Runs the main simulation loop and includes a test function to evaluate learned strategies on a fixed terrain.

## Installation
1. Ensure Python 3.6+ is installed.
2. Install required dependencies:
   ```bash
   pip install numpy
   ```
3. Clone or download the project files (`utils.py`, `environment.py`, `agent.py`, `main.py`).
4. Run the simulation:
   ```bash
   python main.py
   ```

## Usage
- **Running the Simulation**: Execute `main.py` to start the simulation. It runs for up to 100 years or until one country depletes its troops or resources. Press `Ctrl+C` to manually terminate.
- **Output**: The simulation prints yearly updates, with detailed reports every 10 years, showing actions, troop counts, resources, tech levels, morale, pressure, population breakdowns, and rewards.
- **Testing Optimal Strategy**: After the simulation ends, `test_strategy()` evaluates the learned policies for both countries on a "Plains" terrain with initial conditions.

## Simulation Details
- **Initial Conditions**:
  - Troops: 50,000,000 per country
  - Resources: 5,000,000 per country
  - Technology: 0 for both countries
  - Morale: 1.0 for both countries
  - Pressure: 0 for both countries
  - Population: 8,000,000 military, 1,000,000 economic, 1,000,000 tech per country
- **Learning Parameters**:
  - Learning rate (alpha): 0.1
  - Discount factor (gamma): 0.9
  - Initial epsilon: 0.1 (exploration rate)
  - Minimum epsilon: 0.01
  - Epsilon decay: 0.99999
- **Termination Conditions**:
  - A country wins if the opponent’s troops or resources reach zero.
  - The simulation stops after 100 years or upon manual termination.

## Example Output
```
Year 10 (Terrain: Plains, Epsilon: 0.0999):
Country A - Action: 2 (0=Attack, 1=Defend, 2=Economic, 3=Tech, 4=Tech to Military, 5=Economic to Military), Troops: 4998000, Resources: 5100000, Tech: 5, Morale: 0.95, Pressure: 10, Population: Military=7900000, Economic=1100000, Tech=1000000, Children=4500000, Reward: 100.00
Country B - Action: 3, Troops: 5002000, Resources: 4980000, Tech: 6, Morale: 0.90, Pressure: 15, Population: Military=7900000, Economic=1000000, Tech=1100000, Children=4600000, Reward: 50.00
----------------------------------------------------------------------------------------------------
```

## Future Improvements
- Add visualization for troop movements and resource trends.
- Implement more complex terrain interactions or additional actions.
- Optimize LRU cache size and Q-table updates for larger state spaces.
- Introduce alliances or multi-country scenarios.

## License
This project is licensed under the MIT License.
