# Centralized High-level Algorithm for Reinforcement Learning and Optimal Tactical Execution (C.H.A.R.L.O.T.T.E.)

## Overview
C.H.A.R.L.O.T.T.E. is a chess-playing AI framework that utilizes a Deep Q-Learning Network (DQN) with a Convolutional Neural Network (CNN) to optimize strategic decision-making in chess. The system trains an agent to play against random or human opponents, leveraging reinforcement learning to evaluate board states, select optimal moves, and maximize rewards based on captures, checks, and game outcomes. The framework supports parallelized training across multiple environments to enhance learning efficiency.

## Features
- **Deep Q-Learning**: Employs a CNN-based DQN to learn optimal chess strategies through state-action value estimation.
- **Parallel Environments**: Utilizes vectorized environments with multiprocessing to train multiple games simultaneously, improving training speed.
- **Reward System**: Rewards captures (based on piece values), checks (+0.2), checkmates (±1), and penalizes stalemates, insufficient material, or long games.
- **Flexible Gameplay**: Supports play against a random opponent or human input via UCI move format.
- **Board Representation**: Converts chess board states into 12x8x8 tensors for CNN processing, capturing piece positions by type and color.

## Files
- **utils.py**: Defines utility functions for action sets (not shown in provided code but referenced for completeness).
- **chess_dqn.py**: Core implementation of the chess environment, DQN model, vectorized environment, and training/play logic.
- **example.py**: Demonstrates how to use the `play_chess` function to play against a random or human opponent.

## Installation
1. Ensure Python 3.6+ is installed.
2. Install required dependencies:
   ```bash
   pip install torch numpy python-chess
   ```
3. Clone or download the project files (`chess_dqn.py`, `example.py`).
4. Run the training or play scripts:
   ```bash
   python chess_dqn.py  # For training
   python example.py    # For playing
   ```

## Usage
- **Training the Model**: Run `chess_dqn.py` to train the DQN agent over 1000 episodes with 8 parallel environments. The model is saved to `chess_dqn_model.pth` when the average reward improves.
   ```bash
   python chess_dqn.py
   ```
- **Playing a Game**: Use `example.py` to play against the trained model. Specify the opponent as "random" or "human".
   ```bash
   python example.py  # Plays against a random opponent
   ```
   For human play, modify `example.py` to use `play_chess(opponent="human")` and input moves in UCI format (e.g., `e2e4`).
- **Output**: During training, the console displays episode numbers, average rewards, and epsilon values. During play, it shows move outcomes and game results (checkmate, stalemate, etc.).

## Simulation Details
- **Model Architecture**:
  - CNN: Three convolutional layers (12→64, 64→128, 128→256) followed by two fully connected layers (256*8*8→512, 512→218).
  - Action Space: Up to 218 legal moves per board state, dynamically filtered.
- **Training Parameters**:
  - Episodes: 1000
  - Batch Size: 64
  - Number of Environments: 8
  - Learning Rate: 0.0001
  - Gamma: 0.99
  - Epsilon: 1.0 (decays to 0.01 with 0.999 decay rate)
  - Memory: Replay buffer with 20,000 transitions
- **Environment**:
  - Max Moves: 200
  - Long Game Threshold: 150 moves (penalizes extended games)
  - Rewards: +0.1 to +0.9 for captures, +0.2 for checks, +1 for checkmate (win), -1 for checkmate (loss), -0.5 for stalemate/insufficient material, -0.2 for long games.
- **Hardware**: Utilizes all available CPU cores (via `torch.set_num_threads`) and prefers MPS (Metal Performance Shaders) if available, else CPU.

## Example Output
```
Using 8 CPU threads
Using device: cpu
Episode: 1/1000, Avg Reward: -0.50, Epsilon: 0.99
Episode: 2/1000, Avg Reward: -0.30, Epsilon: 0.98
...
Model saved to chess_dqn_model.pth
```

During play:
```
Agent moves: e2e4
Enter your move (UCI format, e.g., 'e2e4'): e7e5
Agent moves: g1f3
...
Agent wins by checkmate!
```

## Future Improvements
- Enhance opponent AI with more sophisticated strategies (e.g., minimax with alpha-beta pruning).
- Add visualization for board states and move probabilities.
- Optimize CNN architecture for deeper analysis of complex positions.
- Support loading and saving training progress for incremental learning.

## License
This project is licensed under the MIT License.
