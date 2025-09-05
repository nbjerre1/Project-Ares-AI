# AI Survival Agent – 30-Minute Challenge

## Overview

This project demonstrates a reinforcement learning agent using Deep Q-Learning with a neural network, that learns to survive for 30 minutes in a custom game environment without dying. The agent interacts with the game via a socket interface, receives state information, and sends back actions in real time. The project showcases deep RL techniques, custom reward shaping, and robust engineering to achieve long-term survival.

<img width="898" height="413" alt="Skærmbillede 2025-09-05 074639" src="https://github.com/user-attachments/assets/89392192-f363-4845-9f07-1c3ec9f9e974" />


## Technology Stack

- **Game Engine:** Godot (C#/.NET 8 for game logic and environment)
- **AI/ML Framework:** PyTorch (Python)
- **Communication:** TCP sockets (Python ↔ Godot)
- **Visualization:** Matplotlib (for training curves and diagnostics)
- **Other:** Numpy, argparse, custom scripts for data logging and plotting

---

## How It Works

1. **Environment & Agent Communication**
    - The Godot game runs the environment and exposes the player state (position, health, timer, mobs, etc.) via a TCP socket.
    - The Python agent connects to this socket, receives state updates, and sends back actions (`move_left`, `move_right`, `move_up`, `move_down`).

2. **Deep Q-Learning (DQN)**
    - The agent uses a deep neural network (PyTorch) to approximate Q-values for each action, given the current state.
    - The network architecture is fully connected with dropout and ReLU activations for stability and generalization.

3. **State Representation**
    - The agent’s input features include normalized player position, health, timer, relative mob position, distance to the nearest wall, and more.
    - This rich feature set helps the agent understand both immediate threats and strategic positioning.

4. **Reward Shaping**
    - The reward function is carefully designed to:
        - Strongly reward survival time.
        - Heavily penalize taking damage or dying.
        - Encourage escaping danger zones and maintaining an ideal distance from mobs.
        - Discourage “corner camping” by penalizing proximity to walls and rewarding central positioning.

5. **Training Loop**
    - The agent uses a prioritized replay buffer to sample important experiences for training.
    - Epsilon-greedy exploration is used, with epsilon decaying slowly to ensure sufficient exploration before converging to exploitation.
    - The target network is updated periodically for stable learning.

6. **Results**
    - The agent’s performance is tracked via loss curves, epsilon decay, and survival time per episode.
    - As shown in the included plots, the agent’s survival time increases steadily, eventually reaching the 30-minute (1800 seconds) goal.

---

## Key Achievements

- **Long-term Survival:** The agent successfully learns to survive for 30 minutes, demonstrating robust policy learning and adaptation.
- **Custom RL Engineering:** The project features advanced reward shaping and prioritized experience replay for efficient learning.
- **Cross-Technology Integration:** Seamless real-time communication between a C# game engine and a Python AI agent.

---

## Usage

1. Start the Godot game server.
2. Run the Python agent: python usemodel.py --port 12345 --load_model BotModel.pth

## Demonstration: Agent Survives 30 Minutes

![output-ezgif com-cut](https://github.com/user-attachments/assets/69384406-db86-4869-bc29-9a48f215ecd5)


The GIF above showcases the trained reinforcement learning agent successfully surviving for the full 30-minute challenge in the game environment. Throughout the episode, the agent demonstrates learned strategies such as maintaining safe distances from enemies, avoiding corners, and adapting its movement to maximize survival time. This result highlights the effectiveness of the custom reward shaping and deep Q-learning approach used in this project.

 
 
 
 
 <img width="1182" height="1184" alt="Skærmbillede 2025-09-04 235058" src="https://github.com/user-attachments/assets/31020487-f176-4eec-ab1b-2aea7376da6a" />

   ## Training Results: Agent Achieves 30-Minute Survival

The graph above visualizes the training progress of the AI agent:

- **Training Loss Over Time:** The top plot shows the agent’s loss decreasing as training progresses, indicating improved decision-making and policy learning.
- **Epsilon Decay Over Time:** The middle plot tracks the exploration rate (epsilon), which gradually decreases as the agent shifts from exploring new actions to exploiting its learned strategies.
- **Survival Time per Episode:** The bottom plot demonstrates the agent’s increasing survival time, ultimately reaching the target of 30 minutes (1800 seconds) without dying. This confirms that the agent has learned a robust survival strategy in the game environment.

**Conclusion:**  
The steady increase in survival time, along with decreasing loss and controlled exploration, demonstrates the effectiveness of the reinforcement learning approach and custom reward shaping. The agent is now capable of consistently surviving for the full 30-minute challenge.

  
