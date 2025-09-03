import matplotlib.pyplot as plt

# Read training losses
losses = []
with open("train_loss.log") as f:
    for line in f:
        if "Loss:" in line:
            loss = float(line.strip().split("Loss:")[1])
            losses.append(loss)

# Read death times
death_times = []
with open("death_times.log") as f:
    for line in f:
        try:
            death_times.append(float(line.strip()))
        except ValueError:
            continue

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot training loss
ax1.plot(losses)
ax1.set_xlabel("Epochs (cumulative)")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss Over Time")

# Plot survival time (death timer)
ax2.plot(death_times)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Survival Time (Death Timer)")
ax2.set_title("Survival Time per Episode")

plt.tight_layout()
plt.show()
