import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(timestamps, values, title=''):
    plt.figure(figsize=(8,3))
    plt.plot(timestamps, values)
    plt.title(title)
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

def save_plot(fig, path):
    fig.savefig(path)