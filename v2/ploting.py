# Extracting data from the newly provided log file with more steps
import re

import numpy as np
from matplotlib import pyplot as plt

steps = []
losses = []
accuracies = []
perplexities = []


# Plotting the metrics with a moving average to smooth out fluctuations
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


with open("logs/TrainingLog.txt", "r") as file:
    for line in file:
        match = re.search(r"Step (\d+), Batch metrics: Loss: ([\d.]+), Accuracy: ([\d.]+), Perplexity: ([\d.]+)", line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            accuracy = float(match.group(3))
            perplexity = float(match.group(4))

            steps.append(step)
            losses.append(loss)
            accuracies.append(accuracy)
            perplexities.append(perplexity)

# Plotting the metrics with a moving average to smooth out fluctuations, using all steps
plt.figure(figsize=(15, 10))

# Loss Plot with Moving Average
plt.subplot(3, 1, 1)
smoothed_losses = moving_average(losses, window_size=50)
plt.plot(steps[49:], smoothed_losses, label='Loss (Moving Average)', color='blue')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss vs Steps (Smoothed with Moving Average)')
plt.legend()

# Accuracy Plot with Moving Average
plt.subplot(3, 1, 2)
smoothed_accuracies = moving_average(accuracies, window_size=50)
plt.plot(steps[49:], smoothed_accuracies, label='Accuracy (Moving Average)', color='green')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Steps (Smoothed with Moving Average)')
plt.legend()

# Perplexity Plot with Moving Average
plt.subplot(3, 1, 3)
smoothed_perplexities = moving_average(perplexities, window_size=50)
plt.plot(steps[49:], smoothed_perplexities, label='Perplexity (Moving Average)', color='red')
plt.xlabel('Steps')
plt.ylabel('Perplexity')
plt.title('Training Perplexity vs Steps (Smoothed with Moving Average)')
plt.legend()

plt.tight_layout()
plt.show()
