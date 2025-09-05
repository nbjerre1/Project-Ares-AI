import socket
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
import torch.optim as optim
import random
import math
from collections import deque

# Hyperparameters for RL
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 128
EPSILON = 0.1  # Exploration rate
GAMMA = 0.95   # Discount factor
MAX_MOBS = 1
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20000  # steps to decay over

# --- Map edge penalty constants ---
MAP_WIDTH = 3324
MAP_HEIGHT = 3324
EDGE_THRESHOLD = 100  # pixels from edge to start penalizing
# ----------------------------------

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
           nn.Linear(6, 32),  # 7 original, minus xpdrop, minus 2*MAX_XP_DROPS, plus 0 = 6
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
if os.path.exists('BotModel.pth'):
    model.load_state_dict(torch.load('BotModel.pth'))
    model.eval()
else:
    print("No BotModel.pth found, using untrained model.")
    model.eval()

# Initialize replay buffer and optimizer globally or in main()
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

ACTIONS = ["move_left", "move_right", "move_up", "move_down"]

def epsilon_by_step(step):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)

def select_action(model, state_tensor, step=None, epsilon=None):
    """
    If epsilon is provided, use it; else compute from step (if step not None).
    """
    if epsilon is None:
        if step is None:
            eps = EPSILON
        else:
            eps = epsilon_by_step(step)
    else:
        eps = epsilon

    if random.random() < eps:
        return random.randint(0, len(ACTIONS) - 1)
    with torch.no_grad():
        logits = model(state_tensor)
        return torch.argmax(logits, dim=1).item()

def build_features(state):
    px = state['x']
    py = state['y']
    W = float(MAP_WIDTH)
    H = float(MAP_HEIGHT)

    mobs = state.get("mobs", [])
    if mobs:
        mob_dx = (mobs[0]["x"] - px) / W
        mob_dy = (mobs[0]["y"] - py) / H
    else:
        mob_dx = 0.0
        mob_dy = 0.0

    features = [
        px / W,
        py / H,
        state['health'] / 100.0,
        state['timer'] / 1800.0,
        mob_dx,
        mob_dy
    ]
    return np.array(features, dtype=np.float32)



def collect_data(sock, log_file):
    while True:
        line = sock.makefile('r').readline()
        if not line:
            break
        line = line.strip()
        if not line or line == '\ufeff' or line.startswith('\ufeff'):
            print(f"Skipped non-JSON or BOM line in collect_data: {line!r}")
            continue
        try:
            state = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Malformed JSON from socket in collect_data: {line!r} ({e})")
            continue
        print("State from Godot:", state)

        mob = state.get("mobs", [])
        if mob:
            mob_x = mob[0]["x"]
            mob_y = mob[0]["y"]
        else:
            mob_x = 0
            mob_y = 0

        state_arr = build_features(state)
        state_tensor = torch.tensor(state_arr).unsqueeze(0)

        with torch.no_grad():
            logits = model(state_tensor)
            action_idx = torch.argmax(logits, dim=1).item()
            action = ACTIONS[action_idx]
            print(f"Model chose action: {action}")

        log_entry = {
            "x": state["x"],
            "y": state["y"],
            "health": state["health"],
            "timer": state["timer"],
            "mob_x": mob_x,
            "mob_y": mob_y,
            "action": action
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        sock.sendall((action + "\n").encode())
        if state.get("health", 1) <= 0:
            print("Player died! Triggering retrain.")
            return "died"

def train_model():
    data = []
    if not os.path.exists("data.jsonl"):
        print("No data.jsonl found, skipping training.")
        return
    with open("data.jsonl", "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("action") in ACTIONS:
                    data.append(entry)
            except Exception as e:
                print(f"Skipping malformed line in data.jsonl: {e}")

    if not data:
        print("No valid data found for training.")
        return

    X = []
    y = []
    for entry in data:
        mob_x = entry.get("mob_x", 0)
        mob_y = entry.get("mob_y", 0)
        X.append([
            entry["x"],
            entry["y"],
            entry["health"],
            entry["timer"],
            mob_x,
            mob_y
        ])
        y.append(ACTIONS.index(entry["action"]))

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    model = SimpleNet()
    if os.path.exists("BotModel.pth"):
        model.load_state_dict(torch.load("BotModel.pth"))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loss_log_path = "train_loss.log"
    loss_log = open(loss_log_path, "a")

    print(f"Training on {len(X)} samples...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
        loss_log.flush()
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

    loss_log.close()
    torch.save(model.state_dict(), "BotModel.pth")
    print("Model trained and saved to BotModel.pth.")

def connect_socket(port=12345):
    """Helper to connect to the Godot socket server with retry logic."""         
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', port))
            print(f"Connected to socket server on port {port}.")
            return sock
        except (ConnectionRefusedError, OSError):
            print(f"Socket server on port {port} not available, retrying in 2 seconds...")
            time.sleep(2)

def calculate_reward(state, next_state):
    # Reward = survive longer
    # Scale with how far into the 1800s (30 min) we are
    t_prev = state.get('timer', 0)
    t_next = next_state.get('timer', 0)

    # Base reward: +1 per second survived (scaled down to avoid exploding Q-values)
    survival_bonus = (t_next - t_prev) * 1  

    reward = survival_bonus

    # Penalty for dying
    if next_state.get('health', 1) <= 0:
        reward -= 200.0   # make death very bad

    # (Optional) small shaping to help early learning:
    # Stay farther from mobs
    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    mobs = next_state.get("mobs", [])
    if mobs:
        px, py = next_state['x'], next_state['y']
        dists = [dist((px, py), (m["x"], m["y"])) for m in mobs]
        min_dist = min(dists)
        if min_dist < 200:
            reward -= 0.001 * (200 - min_dist) ** 2

    return float(reward)

def select_action(model, state_tensor, epsilon=EPSILON):
    if random.random() < epsilon:
        return random.randint(0, len(ACTIONS) - 1)
    with torch.no_grad():
        logits = model(state_tensor)
        return torch.argmax(logits, dim=1).item()

def dqn_update(model, optimizer, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states)                       # [B, A]
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = model(next_states)
        next_q_value = next_q_values.max(1)[0]
        target = rewards + gamma * next_q_value * (1 - dones)

    loss = nn.SmoothL1Loss()(q_value, target)     # Huber

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

def infer(sock, model, optimizer, log_file, global_step):
    prev_state = None
    prev_action = None
    try:
        sock_file = sock.makefile('r')
        while True:
            line = sock_file.readline()
            if not line:
                break
            line = line.strip()
            if not line or line == '\ufeff' or line.startswith('\ufeff'):
                print(f"Skipped non-JSON or BOM line in infer: {line!r}")
                continue
            try:
                state = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Malformed JSON from socket: {line!r} ({e})")
                continue
            print("State from Godot:", state)
            mob = state.get("mobs", [])
            if isinstance(mob, list) and len(mob) > 0:
                mob_x = mob[0]["x"]
                mob_y = mob[0]["y"]
            else:
                mob_x = 0
                mob_y = 0

            state_arr = build_features(state)
            state_tensor = torch.tensor(state_arr).unsqueeze(0)

            action_idx = select_action(model, state_tensor, global_step)
            action = ACTIONS[action_idx]
            print(f"Model chose action: {action}")

            if prev_state is not None and prev_action is not None:
                prev_state_arr = np.array([
                    prev_state['x'], prev_state['y'], prev_state['health'], prev_state['timer'],
                    prev_state.get('mob_x', 0), prev_state.get('mob_y', 0)
                ], dtype=np.float32)

                reward = calculate_reward(prev_state, state)
                done = state.get('health', 1) <= 0

                survival_time = state.get('timer', 0)
                priority = survival_time > 40

                replay_buffer.append((
                    prev_state_arr, prev_action, reward, state_arr, done, priority
                ))
                if len(replay_buffer) >= BATCH_SIZE:
                    high_priority = [t for t in replay_buffer if t[5]]
                    normal = [t for t in replay_buffer if not t[5]]
                    n_high = min(len(high_priority), BATCH_SIZE // 2)
                    n_normal = BATCH_SIZE - n_high

                    batch = []
                    if n_high > 0:
                        batch += random.sample(high_priority, n_high)
                    if n_normal > 0 and len(normal) >= n_normal:
                        batch += random.sample(normal, n_normal)
                    elif len(normal) > 0:
                        batch += random.sample(normal, min(n_normal, len(normal)))

                    dqn_update(model, optimizer, [t[:5] for t in batch], gamma=GAMMA)

            prev_state = {
                "x": state["x"],
                "y": state["y"],
                "health": state["health"],
                "timer": state["timer"],
                "mob_x": mob_x,
                "mob_y": mob_y
            }
            prev_action = action_idx

            log_entry = {
                "x": state["x"],
                "y": state["y"],
                "health": state["health"],
                "timer": state["timer"],
                "mob_x": mob_x,
                "mob_y": mob_y,
                "action": action
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            sock.sendall((action + "\n").encode())
            if state.get("health", 1) <= 0:
                print("Player died! Triggering retrain.")
                with open("death_times.log", "a") as dtlog:
                    dtlog.write(f"{state.get('timer', 0)}\n")
                return "died"
            global_step += 1
    except ConnectionResetError:
        print("Connection to socket server lost (ConnectionResetError). Will attempt to reconnect.")
        return "reconnect"
    except Exception as e:
        print(f"Unexpected error in infer: {e}")
        return "error"
    print("Infer loop exited normally")
    return "done"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['collect', 'train', 'infer', 'auto'], default='auto')
    parser.add_argument('--port', type=int, default=12345)
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
        return

    log_file = open("data.jsonl", "a")

    if args.mode == 'collect':
        sock = connect_socket()
        collect_data(sock, log_file)
        sock.close()
    elif args.mode == 'infer':
        # ...load model as before...
        model = SimpleNet()
        model.load_state_dict(torch.load('BotModel.pth'))
        model.eval()
        sock = connect_socket()
        infer(sock, model, log_file)
        sock.close()
    elif args.mode == 'auto':
        sock = connect_socket()
        log_file = open("data.jsonl", "a")
        model = SimpleNet()
        model.load_state_dict(torch.load('BotModel.pth'))
        model.train()  # allow online updates
        optimizer = optim.Adam(model.parameters(), lr=0.0005)  # tie to *this* model
        global_step = 0

        while True:
            result = infer(sock, model, optimizer, log_file, global_step)
            print("Result from infer:", result)
            if result == "died":
                print("Retraining model after death...")
                train_model()
                print("Reloading model...")
                model.load_state_dict(torch.load('BotModel.pth'))
                model.train()
                optimizer = optim.Adam(model.parameters(), lr=0.0005)  # re-tie optimizer
            elif result == "reconnect":
                print("Reconnecting to socket server after disconnect...")
                sock.close()
                time.sleep(2)
                sock = connect_socket()
            else:
                break


    sock.close()
    log_file.close()

if __name__ == "__main__":
    main()




