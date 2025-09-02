import socket
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 32),  # Changed input size from 3 to 4 to include timer
            nn.ReLU(),
            nn.Linear(32, 6)
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

        state_arr = np.array([state['x'], state['y'], state['health'], state['timer'], mob_x, mob_y], dtype=np.float32)
    
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
        # If your log doesn't include mob_x/mob_y, you may need to update logging above
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

    # Create model and optimizer
    model = SimpleNet()
    if os.path.exists("BotModel.pth"):
        model.load_state_dict(torch.load("BotModel.pth"))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    print(f"Training on {len(X)} samples...")
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "BotModel.pth")
    print("Model trained and saved to BotModel.pth.")

def connect_socket():
    """Helper to connect to the Godot socket server with retry logic."""         
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', 12345))
            print("Connected to socket server.")
            return sock
        except (ConnectionRefusedError, OSError):
            print("Socket server not available, retrying in 2 seconds...")
            time.sleep(2)

def infer(sock, model, log_file):
    try:
        sock_file = sock.makefile('r')
        while True:
            line = sock_file.readline()
            if not line:
                break
            line = line.strip()
            # Skip BOM or non-JSON garbage lines
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
            if mob:
                mob_x = mob[0]["x"]
                mob_y = mob[0]["y"]
            else:
                mob_x = 0
                mob_y = 0
            state_arr = np.array([state['x'], state['y'], state['health'], state['timer'], mob_x, mob_y], dtype=np.float32)
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
