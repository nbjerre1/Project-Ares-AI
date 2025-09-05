import matplotlib.pyplot as plt
import re
import numpy as np

# Read training losses, steps, and epsilon values
losses = []
steps = []
epsilons = []
learning_rates = []

with open("train_loss.log") as f:
    for line in f:
        if "Loss:" in line:
            # Extract loss
            loss = float(line.strip().split("Loss:")[1].split()[0])
            losses.append(loss)
            
            # Extract step number
            if "Step:" in line:
                step = int(line.strip().split("Step:")[1].split()[0])
                steps.append(step)
            
            # Extract epsilon
            if "Eps:" in line:
                eps = float(line.strip().split("Eps:")[1].split()[0])
                epsilons.append(eps)
            
            # Extract learning rate if present
            if "LR:" in line:
                lr = float(line.strip().split("LR:")[1].split()[0])
                learning_rates.append(lr)

# Read death times
death_times = []
with open("death_times.log") as f:
    for line in f:
        try:
            death_times.append(float(line.strip()))
        except ValueError:
            continue

# Create subplots - now with 3 or 4 plots depending on available data
num_plots = 3 if not learning_rates else 4
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots))

if num_plots == 3:
    ax1, ax2, ax3 = axes
else:
    ax1, ax2, ax3, ax4 = axes

# Plot 1: Training Loss over Steps
if steps and losses:
    ax1.plot(steps, losses, 'b-', alpha=0.7)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Time")
    ax1.grid(True, alpha=0.3)
else:
    ax1.plot(losses, 'b-', alpha=0.7)
    ax1.set_xlabel("Training Iterations")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Time")
    ax1.grid(True, alpha=0.3)

# Plot 2: Epsilon Decay over Steps
if steps and epsilons:
    ax2.plot(steps, epsilons, 'r-', alpha=0.8)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Epsilon (Exploration Rate)")
    ax2.set_title("Epsilon Decay Over Time")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for key epsilon milestones
    if epsilons:
        current_eps = epsilons[-1]
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% exploration')
        ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='10% exploration')
        ax2.legend()
        ax2.text(0.02, 0.98, f'Current ε: {current_eps:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 3: Survival Time per Episode
episode_numbers = range(1, len(death_times) + 1)
ax3.plot(episode_numbers, death_times, 'g-o', markersize=4, alpha=0.7)
ax3.set_xlabel("Episode")
ax3.set_ylabel("Survival Time (seconds)")
ax3.set_title("Survival Time per Episode")
ax3.grid(True, alpha=0.3)

# Add trend line for survival times
if len(death_times) > 1:
    z = np.polyfit(episode_numbers, death_times, 1)
    p = np.poly1d(z)
    ax3.plot(episode_numbers, p(episode_numbers), "r--", alpha=0.8, label=f'Trend: {z[0]:.2f}s/episode')
    ax3.legend()

# Add stats text
if death_times:
    avg_survival = sum(death_times) / len(death_times)
    max_survival = max(death_times)
    last_10_avg = sum(death_times[-10:]) / min(10, len(death_times))
    
    stats_text = f'Avg: {avg_survival:.1f}s\nMax: {max_survival:.1f}s\nLast 10: {last_10_avg:.1f}s'
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 4: Learning Rate (if available)
if learning_rates and num_plots == 4:
    ax4.plot(steps[-len(learning_rates):], learning_rates, 'm-', alpha=0.8)
    ax4.set_xlabel("Training Steps")
    ax4.set_ylabel("Learning Rate")
    ax4.set_title("Learning Rate Schedule")
    ax4.set_yscale('log')  # Log scale for learning rate
    ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Print summary statistics
print("=== Training Summary ===")
print(f"Total episodes: {len(death_times)}")
print(f"Total training steps: {steps[-1] if steps else 'Unknown'}")
print(f"Current epsilon: {epsilons[-1]:.4f}" if epsilons else "Epsilon: Unknown")
print(f"Current learning rate: {learning_rates[-1]:.2e}" if learning_rates else "LR: Unknown")
print(f"Average survival time: {avg_survival:.2f}s" if death_times else "No episodes completed")
print(f"Best survival time: {max_survival:.2f}s" if death_times else "No episodes completed")

# Show improvement metrics
if len(death_times) >= 20:
    first_10 = sum(death_times[:10]) / 10
    last_10 = sum(death_times[-10:]) / 10
    improvement = ((last_10 - first_10) / first_10) * 100
    print(f"Improvement (last 10 vs first 10): {improvement:+.1f}%")

plt.show()