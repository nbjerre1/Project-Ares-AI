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
from collections import deque

# Hyperparameters for RL
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 256
EPSILON = 0.1  # Exploration rate
GAMMA = 0.95   # Discount factor
MAX_XP_DROPS = 10  # or any reasonable upper limit
MAX_MOBS = 10

# --- Map edge penalty constants ---
MAP_WIDTH = 3324
MAP_HEIGHT = 3324
EDGE_THRESHOLD = 100  # pixels from edge to start penalizing
# ----------------------------------



class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
           nn.Linear(7 + 2 * MAX_XP_DROPS, 32),  # <-- update input size
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

def collect_data(sock, log_file):
    while True:
        line = sock.makefile('r').readline()
        if not line:
            break
        line = line.strip()
        # Skip BOM or non-JSON garbage lines
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

        

        xp_drops = state.get("xp_drops", [])
        xp_features = []
        for i in range(MAX_XP_DROPS):
            if i < len(xp_drops):
                xp_features.extend([xp_drops[i]["x"], xp_drops[i]["y"]])
            else:
                xp_features.extend([0, 0])  # pad with zeros if not enough drops

        state_arr = np.array([
            state['x'], state['y'], state['health'], state['timer'],
            mob_x, mob_y, state.get('xpdrop', 0)
        ] + xp_features, dtype=np.float32)

    
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
            "xpdrop": state.get("xpdrop", 0),
            "xp_drops": xp_drops,  # log the raw list if desired
            "action": action
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        sock.sendall((action + "\n").encode())
        if state.get("health", 1) <= 0:
            print("Player died! Triggering retrain.")
            return "died"

def train_model():
    # Load data from data.jsonl
    data = []
    if not os.path.exists("data.jsonl"):
        print("No data.jsonl found, skipping training.")
        return
    with open("data.jsonl", "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Only use entries with valid actions
                if entry.get("action") in ACTIONS:
                    data.append(entry)
            except Exception as e:
                print(f"Skipping malformed line in data.jsonl: {e}")

    if not data:
        print("No valid data found for training.")
        return

    # Prepare training data
    X = []
    y = []
    for entry in data:
        # Default mob_x, mob_y to 0 if not present
        mob_x = entry.get("mob_x", 0)
        mob_y = entry.get("mob_y", 0)
        xp_drops = entry.get("xp_drops", [])
        xp_features = []
        for i in range(MAX_XP_DROPS):
            if i < len(xp_drops):
                xp_features.extend([xp_drops[i]["x"], xp_drops[i]["y"]])
            else:
                xp_features.extend([0, 0])  # pad with zeros

        X.append([
            entry["x"],
            entry["y"],
            entry["health"],
            entry["timer"],
            mob_x,
            mob_y,
            entry.get("xpdrop", 0)
        ] + xp_features)
        y.append(ACTIONS.index(entry["action"]))

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Create model and optimizer
    model = SimpleNet()
    if os.path.exists("BotModel.pth"):
        model.load_state_dict(torch.load("BotModel.pth"))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # --- Loss logging setup ---
    loss_log_path = "train_loss.log"
    loss_log = open(loss_log_path, "a")
    # ---

    # Train
    print(f"Training on {len(X)} samples...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        # --- Log loss to file ---
        loss_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch+1}, Loss: {loss.item():.4f}\n")
        loss_log.flush()
        # ---
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

    loss_log.close()
    # Save model
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
    reward = 0
    # Reward for surviving another step (encourage longer survival)
    reward += (next_state.get('timer', 0) - state.get('timer', 0)) * 10.0  # scale as needed

    # Penalty for losing health
    health_loss = state.get('health', 0) - next_state.get('health', 0)
    if health_loss > 0:
        reward -= health_loss * 10  # You can tune this penalty

    # Penalty for dying (optional, for extra punishment)
    if next_state.get('health', 1) <= 0:
        reward -= 100

    # --- SHAPING REWARDS: Add here ---

    # Helper for distance
    def distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    prev_pos = np.array([state['x'], state['y']])
    next_pos = np.array([next_state['x'], next_state['y']])

    # --- Penalty for moving closer to any mob (all mobs) ---
    mobs = next_state.get("mobs", [])
    mob_penalty = 0
    close_mob_penalty = 0
    for mob in mobs:
        mob_pos = np.array([mob["x"], mob["y"]])
        prev_dist = distance(prev_pos, mob_pos)
        next_dist = distance(next_pos, mob_pos)
        if next_dist < prev_dist:
            mob_penalty -= 5  # penalty for moving closer to a mob
        if next_dist < 100:  # Tune this threshold
            close_mob_penalty -= (100 - next_dist) * 0.5  # Stronger penalty the closer you are
    reward += mob_penalty
    reward += close_mob_penalty
    # -------------------------------------------------------

    # Penalty for being near the map edge
    if (
        next_state['x'] < EDGE_THRESHOLD or
        next_state['x'] > MAP_WIDTH - EDGE_THRESHOLD or
        next_state['y'] < EDGE_THRESHOLD or
        next_state['y'] > MAP_HEIGHT - EDGE_THRESHOLD
    ):
        reward -= 20  # Penalty for being near edge
        # Encourage moving toward center
        center = np.array([MAP_WIDTH/2, MAP_HEIGHT/2])
        to_center = np.linalg.norm(next_pos - center)
        from_center = np.linalg.norm(prev_pos - center)
        if to_center < from_center:
            reward += 10  # Reward for moving toward center

    # Bonus for good runs
    if next_state.get('health', 1) <= 0 and next_state.get('timer', 0) > 40:
         reward += 100  # Or another value

    # --- END SHAPING REWARDS ---
    return reward

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

    q_values = model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = model(next_states)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    loss = nn.MSELoss()(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def infer(sock, model, log_file):
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

            

            xp_drops = state.get("xp_drops", [])
            xp_features = []
            for i in range(MAX_XP_DROPS):
                if i < len(xp_drops):
                    xp_features.extend([xp_drops[i]["x"], xp_drops[i]["y"]])
                else:
                    xp_features.extend([0, 0])  # pad with zeros if not enough drops

            state_arr = np.array([
                state['x'], state['y'], state['health'], state['timer'],
                mob_x, mob_y, state.get('xpdrop', 0)
            ] + xp_features, dtype=np.float32)
            state_tensor = torch.tensor(state_arr).unsqueeze(0)

            # Epsilon-greedy action selection
            action_idx = select_action(model, state_tensor)
            action = ACTIONS[action_idx]
            print(f"Model chose action: {action}")

            # RL transition storage and update
            if prev_state is not None and prev_action is not None:
                prev_xp_drops = prev_state.get("xp_drops", [])
                prev_xp_features = []
                for i in range(MAX_XP_DROPS):
                    if i < len(prev_xp_drops):
                        prev_xp_features.extend([prev_xp_drops[i]["x"], prev_xp_drops[i]["y"]])
                    else:
                        prev_xp_features.extend([0, 0])

                prev_state_arr = np.array([
                    prev_state['x'], prev_state['y'], prev_state['health'], prev_state['timer'],
                    prev_state.get('mob_x', 0), prev_state.get('mob_y', 0), prev_state.get('xpdrop', 0)
                ] + prev_xp_features, dtype=np.float32)


                reward = calculate_reward(prev_state, state)
                done = state.get('health', 1) <= 0

                # Tag transitions from good runs
                survival_time = state.get('timer', 0)
                priority = survival_time > 40  # Set your threshold

                replay_buffer.append((
                    prev_state_arr, prev_action, reward, state_arr, done, priority
                ))
                # DQN update
                if len(replay_buffer) >= BATCH_SIZE:
                    # Prioritize good runs
                    high_priority = [t for t in replay_buffer if len(t) > 5 and t[5]]
                    normal = [t for t in replay_buffer if len(t) <= 5 or not t[5]]

                    n_high = min(len(high_priority), BATCH_SIZE // 2)
                    n_normal = BATCH_SIZE - n_high

                    batch = []
                    if n_high > 0:
                        batch += random.sample(high_priority, n_high)
                    if n_normal > 0 and len(normal) >= n_normal:
                        batch += random.sample(normal, n_normal)
                    elif len(normal) > 0:
                        batch += random.sample(normal, len(normal))

                    dqn_update(model, optimizer, [t[:5] for t in batch], gamma=GAMMA)

                
            prev_state = {
                "x": state["x"],
                "y": state["y"],
                "health": state["health"],
                "timer": state["timer"],
                "mob_x": mob_x,
                "mob_y": mob_y,
                "xpdrop": state.get("xpdrop", 0),
                "xp_drops": xp_drops
            }
            prev_action = action_idx

            # Logging and action sending
            log_entry = {
                "x": state["x"],
                "y": state["y"],
                "health": state["health"],
                "timer": state["timer"],
                "mob_x": mob_x,
                "mob_y": mob_y,
                "xpdrop": state.get("xpdrop", 0),
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
        model.eval()
        while True:
                result = infer(sock, model, log_file)
                print ("Result from infer:", result)
                if result == "died":
                    print("Retraining model after death...")
                    train_model()
                    print("Reloading model...")
                    model.load_state_dict(torch.load('BotModel.pth'))
                    model.eval()
                    # Continue without closing/reconnecting socket
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




