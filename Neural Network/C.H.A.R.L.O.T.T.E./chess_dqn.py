import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import multiprocessing as mp
from multiprocessing import Pool
import uuid
from tqdm import tqdm

num_threads = os.cpu_count() or 1
torch.set_num_threads(num_threads)
print(f"Using {num_threads} CPU threads")

class ChessDQN(nn.Module):
    def __init__(self, output_size):
        super(ChessDQN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 12, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        tensor[piece_idx, rank, file] = 1
    return tensor

piece_values = {
    chess.PAWN: 0.1,
    chess.KNIGHT: 0.3,
    chess.BISHOP: 0.3,
    chess.ROOK: 0.5,
    chess.QUEEN: 0.9,
    chess.KING: 0.0
}

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.max_moves = 200
        self.long_game_threshold = 150
        self.move_count = 0

    def reset(self):
        self.board.reset()
        self.move_count = 0
        return board_to_tensor(self.board)

    def step(self, move_idx, is_agent_turn=True):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return board_to_tensor(self.board), -1.0, True, {"no_legal_moves": True}
        
        if move_idx < 0 or move_idx >= len(legal_moves):
            return board_to_tensor(self.board), -1.0, True, {"invalid_move": True}
        
        move = legal_moves[move_idx]
        reward = 0.0
        if self.board.is_capture(move):
            captured_piece = self.board.piece_at(move.to_square)
            if captured_piece:
                reward += piece_values.get(captured_piece.piece_type, 0.0)
        if self.board.gives_check(move):
            reward += 0.2

        self.board.push(move)
        self.move_count += 1

        info = {"move": str(move)}
        done = False

        if self.board.is_checkmate():
            reward += 1.0 if is_agent_turn else -1.0
            done = True
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward -= 0.5
            done = True
        elif self.move_count >= self.max_moves:
            reward -= 0.5
            done = True
        elif self.move_count >= self.long_game_threshold:
            reward -= 0.2

        return board_to_tensor(self.board), reward, done, info

    def opponent_move(self, human_move=None):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return board_to_tensor(self.board), -1.0, True, {"no_legal_moves": True}

        if human_move:
            try:
                move = chess.Move.from_uci(human_move)
                if move not in legal_moves:
                    return board_to_tensor(self.board), -1.0, True, {"invalid_move": True}
            except ValueError:
                return board_to_tensor(self.board), -1.0, True, {"invalid_move": True}
        else:
            if random.random() < 0.3:
                captures = [m for m in legal_moves if self.board.is_capture(m)]
                if captures:
                    captures_with_value = []
                    for m in captures:
                        captured_piece = self.board.piece_at(m.to_square)
                        if captured_piece:
                            value = piece_values.get(captured_piece.piece_type, 0.0)
                            captures_with_value.append((m, value))
                    captures_with_value.sort(key=lambda x: x[1], reverse=True)
                    move = captures_with_value[0][0] if captures_with_value else random.choice(captures)
                else:
                    checks = [m for m in legal_moves if self.board.gives_check(m)]
                    move = random.choice(checks or legal_moves)
            else:
                move = random.choice(legal_moves)

        reward = 0.0
        if self.board.is_capture(move):
            captured_piece = self.board.piece_at(move.to_square)
            if captured_piece:
                reward -= piece_values.get(captured_piece.piece_type, 0.0)
        if self.board.gives_check(move):
            reward -= 0.2

        self.board.push(move)
        self.move_count += 1
        info = {"move": str(move)}
        done = False

        if self.board.is_checkmate():
            reward -= 1.0
            done = True
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward -= 0.5
            done = True
        elif self.move_count >= self.max_moves:
            reward -= 0.5
            done = True
        elif self.move_count >= self.long_game_threshold:
            reward -= 0.2

        return board_to_tensor(self.board), reward, done, info

    def get_legal_moves(self):
        return list(self.board.legal_moves)

def env_step_worker(args):
    env, move_idx, is_agent_turn = args
    return env.step(move_idx, is_agent_turn)

def env_opponent_move_worker(args):
    env, human_move = args
    return env.opponent_move(human_move)

class VectorChessEnv:
    def __init__(self, num_envs=8):
        self.num_envs = num_envs
        self.envs = [ChessEnv() for _ in range(num_envs)]
        self.pool = Pool(processes=num_envs)

    def reset(self):
        states = [env.reset() for env in self.envs]
        return np.stack(states)

    def step(self, move_indices, active_indices):
        states = [None] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{}] * self.num_envs

        step_args = [(self.envs[i], move_indices[idx], True) for idx, i in enumerate(active_indices)]
        results = self.pool.map(env_step_worker, step_args)

        for idx, i in enumerate(active_indices):
            states[i], rewards[i], dones[i], infos[i] = results[idx]

        for i in range(self.num_envs):
            if states[i] is None:
                states[i] = board_to_tensor(self.envs[i].board)
        return np.stack(states), rewards, dones, infos

    def opponent_moves(self, active_indices):
        states = [None] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{}] * self.num_envs

        step_args = [(self.envs[i], None) for i in active_indices]
        results = self.pool.map(env_opponent_move_worker, step_args)

        for idx, i in enumerate(active_indices):
            states[i], rewards[i], dones[i], infos[i] = results[idx]

        for i in range(self.num_envs):
            if states[i] is None:
                states[i] = board_to_tensor(self.envs[i].board)
        return np.stack(states), rewards, dones, infos

    def get_legal_moves(self):
        return [env.get_legal_moves() for env in self.envs]

    def close(self):
        self.pool.close()
        self.pool.join()

class DQNAgent:
    def __init__(self, action_size=218):
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = ChessDQN(action_size).to(self.device)
        self.target_model = ChessDQN(action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        self.best_reward = float('-inf')

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves, explore=True):
        if not legal_moves:
            return -1
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(len(legal_moves))
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        q_values = q_values.cpu().numpy()[0]
        legal_indices = list(range(len(legal_moves)))
        valid_q_values = q_values[legal_indices]
        return legal_indices[np.argmax(valid_q_values)]

    def act_batch(self, states, legal_moves_list):
        actions = []
        for state, legal_moves in zip(states, legal_moves_list):
            actions.append(self.act(state, legal_moves))
        return np.array(actions)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        targets = rewards + (1 - dones) * self.gamma * torch.max(self.target_model(next_states), dim=1)[0]
        target_f = self.model(states).clone().detach()
        for i in range(batch_size):
            target_f[i][actions[i]] = targets[i]

        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = nn.MSELoss()(outputs, target_f)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_chess_agent(episodes=1000, batch_size=64, num_envs=8, model_path="chess_dqn_model.pth"):
    env = VectorChessEnv(num_envs=num_envs)
    agent = DQNAgent()
    
    try:
        for e in tqdm(range(episodes), desc="Training Episodes"):
            states = env.reset()
            dones = np.zeros(num_envs, dtype=bool)
            total_rewards = np.zeros(num_envs)
            while not all(dones):
                legal_moves_list = env.get_legal_moves()
                active_indices = np.where(~dones)[0]
                if len(active_indices) == 0:
                    break
                active_legal_moves = [legal_moves_list[i] for i in active_indices]
                actions = agent.act_batch(states[active_indices], active_legal_moves)
                
                next_states, rewards, step_dones, infos = env.step(actions, active_indices)
                for idx, i in enumerate(active_indices):
                    if rewards[i] != 0 or step_dones[i]:
                        agent.remember(states[i], actions[idx], rewards[i], next_states[i], step_dones[i])
                    total_rewards[i] += rewards[i]
                    dones[i] = step_dones[i]
                states = next_states

                if all(dones):
                    break

                active_indices = np.where(~dones)[0]
                if len(active_indices) == 0:
                    break
                next_states, rewards, step_dones, infos = env.opponent_moves(active_indices)
                for i in active_indices:
                    if rewards[i] != 0 or step_dones[i]:
                        agent.remember(states[i], 0, rewards[i], next_states[i], step_dones[i])
                    total_rewards[i] += rewards[i]
                    dones[i] = step_dones[i]
                states = next_states

                agent.replay(batch_size)

            avg_reward = np.mean(total_rewards)
            if avg_reward > agent.best_reward:
                agent.best_reward = avg_reward
                agent.save_model(model_path)

            agent.update_target_model()
            print(f"Episode: {e+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        agent.save_model(model_path)

    finally:
        env.close()

def play_chess(model_path="chess_dqn_model.pth", opponent="random"):
    agent = DQNAgent()
    try:
        agent.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    env = ChessEnv()
    state = env.reset()
    done = False
    move_count = 0

    while not done:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            print("No legal moves available for agent.")
            done = True
            break

        action = agent.act(state, legal_moves, explore=False)
        state, reward, done, info = env.step(action, is_agent_turn=True)
        print(f"Agent's move: {info['move']}")
        move_count += 1

        if done:
            if reward >= 1:
                print("Agent wins by checkmate!")
            elif reward <= -0.5:
                print("Game ends in stalemate or insufficient material.")
            elif reward <= -1:
                print("Opponent wins by checkmate!")
            break

        if opponent == "human":
            while True:
                move = input("Enter your move (UCI format, e.g., 'e2e4'): ")
                state, reward, done, info = env.opponent_move(human_move=move)
                if "invalid_move" in info:
                    print("Invalid move. Try again.")
                    continue
                print(f"Opponent's move: {info['move']}")
                break
        else:
            state, reward, done, info = env.opponent_move()
            print(f"Opponent's move: {info['move']}")

        if done:
            if reward <= -1:
                print("Opponent wins by checkmate!")
            elif reward <= -0.5:
                print("Game ends in stalemate or insufficient material.")
            elif reward <= -0.2:
                print("Game ends due to max moves reached.")
            break

        if move_count >= env.max_moves:
            print("Game ends due to max moves reached.")
            break

if __name__ == "__main__":
    train_chess_agent()
    play_chess(opponent="random")
