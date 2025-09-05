import socket
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import time
import random
import math
from collections import deque

# Hyperparameters
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 128
GAMMA = 0.99
LR = 0.0005  # Slightly higher learning rate
TARGET_UPDATE = 1000
EPS_START = 0.5
EPS_END = 0.05  # More exploration at end
EPS_DECAY = 50000  # Much slower decay

MAP_WIDTH = 3324
MAP_HEIGHT = 3324
SAFE_DISTANCE = 400  # Minimum safe distance from mobs
IDEAL_DISTANCE = 600  # Ideal distance to maintain

ACTIONS = ["move_left", "move_right", "move_up", "move_down"]

# -------------------
# Model
# -------------------
class ImprovedNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Larger network for better representation
        self.fc1 = nn.Linear(8, 256)  # Added more input features
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, len(ACTIONS))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# -------------------
# Feature preprocessing (IMPROVED)
# -------------------
def build_features(state):
    px, py = state['x'], state['y']
    W, H = float(MAP_WIDTH), float(MAP_HEIGHT)

    # Normalize player position
    norm_x = px / W
    norm_y = py / H
    
    # Wall distances (important for avoiding getting trapped)
    wall_dist_left = px / W
    wall_dist_right = (W - px) / W
    wall_dist_top = py / H
    wall_dist_bottom = (H - py) / H

    mobs = state.get("mobs", [])
    if mobs:
        mob_x, mob_y = mobs[0]["x"], mobs[0]["y"]
        # Relative position to mob
        mob_dx = (mob_x - px) / W
        mob_dy = (mob_y - py) / H
        # Distance to mob (normalized)
        mob_distance = np.linalg.norm([mob_dx * W, mob_dy * H]) / (W * 0.5)
    else:
        mob_dx, mob_dy, mob_distance = 0.0, 0.0, 1.0

    return np.array([
        norm_x, norm_y,
        state['health'] / 100.0,
        state['timer'] / 1800.0,
        mob_dx, mob_dy, mob_distance,
        min(wall_dist_left, wall_dist_right, wall_dist_top, wall_dist_bottom)  # Closest wall
    ], dtype=np.float32)

def state_to_tensor(state):
    return torch.tensor(build_features(state)).unsqueeze(0)

# -------------------
# Epsilon-greedy (FIXED)
# -------------------
def epsilon_by_step(step):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)

def select_action(model, state_tensor, step):
    eps = epsilon_by_step(step)
    if random.random() < eps:
        return random.randint(0, len(ACTIONS) - 1)
    with torch.no_grad():
        model.eval()  # Important: set to eval for inference
        q_values = model(state_tensor)
        model.train()  # Back to train mode
        return q_values.argmax(1).item()

# -------------------
# Replay buffer + DQN update (FIXED)
# -------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6
        self.epsilon = 0.01
        
    def add(self, experience, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-0.4)
        weights /= weights.max()
        
        return samples, weights, indices
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority + self.epsilon
            
    def __len__(self):
        return len(self.buffer)

replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)

def dqn_update(model, target_model, optimizer, batch, weights, indices, buffer, global_step):
    if len(batch) == 0:
        return 0.0
        
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    # Current Q values
    q_values = model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN
        next_actions = model(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_model(next_states)
        next_q_value = next_q_values.gather(1, next_actions).squeeze(1)
        target = rewards + GAMMA * next_q_value * (1 - dones)
    
    # TD error for prioritization
    td_error = torch.abs(q_value - target).detach().numpy()
    buffer.update_priorities(indices, td_error)
    
    # Weighted loss
    loss = (weights * nn.SmoothL1Loss(reduction='none')(q_value, target)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Update target network
    if global_step % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())
        print(f"Updated target network at step {global_step}")
        
    return loss.item()

# -------------------
# BALANCED Reward function - prevents corner camping while prioritizing survival
# -------------------
def calculate_reward(state, next_state, action_taken):
    # Basic info
    health_prev = state.get('health', 100)
    health_next = next_state.get('health', 0)
    health_delta = health_next - health_prev
    
    t_prev = state.get('timer', 0)
    t_next = next_state.get('timer', 0)
    time_delta = t_next - t_prev
    
    px, py = next_state['x'], next_state['y']
    prev_px, prev_py = state['x'], state['y']
    
    # Base survival reward - this is the main objective
    reward = 1.5 * time_delta  # Strong survival incentive
    
    # CRITICAL: Heavy penalty for taking damage or dying
    if health_delta < 0:
        reward -= 25.0 * abs(health_delta)  # This should dominate other rewards
    
    if next_state.get('health', 1) <= 0:
        reward -= 200.0  # Massive death penalty
        return float(reward)
    
    # Mob interaction logic
    mobs = next_state.get("mobs", [])
    prev_mobs = state.get("mobs", [])
    
    if mobs and prev_mobs:
        # Current distance to mob
        mob_x, mob_y = mobs[0]["x"], mobs[0]["y"]
        current_dist = np.linalg.norm([px - mob_x, py - mob_y])
        
        # Previous distance to mob
        prev_mob_x, prev_mob_y = prev_mobs[0]["x"], prev_mobs[0]["y"]
        prev_dist = np.linalg.norm([prev_px - prev_mob_x, prev_py - prev_mob_y])
        
        # Dynamic distance rewards based on current danger level
        if current_dist < SAFE_DISTANCE:
            # In danger - prioritize escape!
            if current_dist > prev_dist:
                reward += 3.0  # Big reward for escaping danger
            else:
                reward -= 2.0  # Penalty for getting closer when already close
            
            # Additional penalty for being in danger zone
            danger_penalty = (SAFE_DISTANCE - current_dist) / 100.0
            reward -= danger_penalty
            
        elif current_dist < IDEAL_DISTANCE:
            # Sweet spot - safe but engaged
            reward += 0.5  # Small positive reward for good positioning
            
        elif current_dist < IDEAL_DISTANCE * 2:
            # Getting far but acceptable
            # Very small penalty to encourage engagement, but shouldn't override survival
            reward -= 0.1 * (current_dist - IDEAL_DISTANCE) / IDEAL_DISTANCE
            
        else:
            # Too far - this discourages extreme corner camping
            # But keep penalty small so it doesn't override survival instinct
            excess_dist = current_dist - (IDEAL_DISTANCE * 2)
            reward -= 0.2 * (excess_dist / MAP_WIDTH)  # Scale with map size
    
    # Anti-corner camping: penalty for being too close to walls
    center_x, center_y = MAP_WIDTH / 2, MAP_HEIGHT / 2
    wall_distances = [
        px,  # distance to left wall
        MAP_WIDTH - px,  # distance to right wall  
        py,  # distance to top wall
        MAP_HEIGHT - py  # distance to bottom wall
    ]
    min_wall_dist = min(wall_distances)
    
    # Penalty for hugging walls (but keep it smaller than survival rewards)
    if min_wall_dist < 300:
        wall_penalty = (300 - min_wall_dist) / 1000.0  # Small penalty
        reward -= wall_penalty
    
    # Small bonus for staying closer to center (anti-corner camping)
    center_dist = np.linalg.norm([px - center_x, py - center_y])
    max_center_dist = np.linalg.norm([MAP_WIDTH/2, MAP_HEIGHT/2])
    if center_dist > max_center_dist * 0.7:  # Only penalize extreme corners
        corner_penalty = 0.1 * ((center_dist - max_center_dist * 0.7) / max_center_dist)
        reward -= corner_penalty
    
    return float(np.clip(reward, -100, 20))  # Clip rewards but allow larger penalties

# -------------------
# Training loop (FIXED)
# -------------------
def infer(sock, model, target_model, optimizer, scheduler, log_file, global_step):
    prev_state, prev_action = None, None
    sock_file = sock.makefile('r')

    try:
        while True:
            line = sock_file.readline()
            if not line:
                break
            state = json.loads(line.strip())
            
            state_tensor = state_to_tensor(state)
            action_idx = select_action(model, state_tensor, global_step[0])
            action = ACTIONS[action_idx]
            
            eps = epsilon_by_step(global_step[0])
            
            # Early training diagnostics
            if global_step[0] < 1000 and global_step[0] % 50 == 0:
                mobs = state.get("mobs", [])
                if mobs:
                    mob_dist = np.linalg.norm([state['x'] - mobs[0]['x'], state['y'] - mobs[0]['y']])
                    print(f"Step {global_step[0]}: pos=({state['x']:.0f},{state['y']:.0f}), mob_dist={mob_dist:.0f}, health={state['health']}")
            
            print(f"Step {global_step[0]}, eps: {eps:.3f}, action: {action}, health: {state.get('health', 0)}")

            # Save transition and train
            if prev_state is not None:
                reward = calculate_reward(prev_state, state, prev_action)
                done = state.get('health', 1) <= 0
                
                replay_buffer.add((
                    build_features(prev_state), 
                    prev_action, 
                    reward, 
                    build_features(state), 
                    done
                ))

                # Train more frequently but with smaller batches initially
                min_buffer_size = min(BATCH_SIZE, 1000)
                if len(replay_buffer) >= min_buffer_size:
                    batch, weights, indices = replay_buffer.sample(min(BATCH_SIZE, len(replay_buffer)))
                    # Remove 'scheduler' from dqn_update call
                    loss = dqn_update(model, target_model, optimizer, batch, weights, indices, replay_buffer, global_step[0])

                    # Call scheduler.step() here if needed
                    scheduler.step()

                    if global_step[0] % 100 == 0:  # Log less frequently
                        with open("train_loss.log", "a") as loss_log:
                            loss_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Step: {global_step[0]} Loss: {loss:.4f} Eps: {eps:.3f}\n")

            prev_state, prev_action = state, action_idx

            # Log state
            log_file.write(json.dumps({
                "step": global_step[0],
                "x": state["x"], "y": state["y"],
                "health": state["health"], "timer": state["timer"],
                "action": action,
                "epsilon": eps
            }) + "\n")
            log_file.flush()
            
            # Send action
            sock.sendall((action + "\n").encode())

            if state.get('health', 1) <= 0:
                survival_time = state.get('timer', 0)
                print(f"Player died at timer {survival_time}, restarting episode")
                with open("death_times.log", "a") as dtlog:
                    dtlog.write(f"{survival_time}\n")
                return "died"

            global_step[0] += 1

    except (ConnectionResetError, json.JSONDecodeError) as e:
        print("Socket error:", e)
        return "reconnect"
    return "done"

# -------------------
# Networking helper
# -------------------
def connect_socket(port=12345):
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', port))
            print(f"Connected to server on port {port}")
            return sock
        except (ConnectionRefusedError, OSError):
            print("Waiting for server...")
            time.sleep(2)

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--load_model', type=str, default="BotModel.pth")
    args = parser.parse_args()

    model = ImprovedNet()
    if os.path.exists(args.load_model):
        try:
            checkpoint = torch.load(args.load_model)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint {args.load_model}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model from {args.load_model}")
        except Exception as e:
            print(f"Failed to load model: {e}. Creating new model.")

    model.train()
    
    # Target model
    target_model = ImprovedNet()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    # Step scheduler every 1000 training steps instead of every step
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.95)

    log_file = open("data.jsonl", "a")
    sock = connect_socket(args.port)

    global_step = [0]  # Use list to allow modification in nested function
    episode = 0
    
    while True:
        episode += 1
        print(f"\n=== Starting Episode {episode} ===")
        result = infer(sock, model, target_model, optimizer, scheduler, log_file, global_step)
        
        if result == "died":
            # Save model with more info
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'global_step': global_step[0]
            }
            torch.save(checkpoint, "BotModel.pth")
            torch.save(model.state_dict(), "BotModel_latest.pth")
            print(f"Saved model after episode {episode}")
            
          
            
        elif result == "reconnect":
            sock.close()
            sock = connect_socket(args.port)
        else:
            break

    sock.close()
    log_file.close()

if __name__ == "__main__":
    main()