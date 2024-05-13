import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt


def visualize_channel():
    x = cPickle.load(open("./DataFiles/s02" + ".dat", 'rb'), encoding="bytes")
    field_names = []
    for key in x.keys():
        field_names.append(key)

    # labels = x[field_names[0]]

    data = x[field_names[1]]
    print(data[0])

    lst = [0, 16, 2, 24]
    dat = []
    for i in range(40):
        tmp = []
        for j in lst:
            tmp.append(data[i][j])
        dat.append(tmp)
    dat = np.array(dat)
    feature = []
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.suptitle("Recorded Data Visualization", fontsize=15, color='purple')
    ax1, ax2, ax3, ax4 = axes.flatten()

    lst = [ax1, ax2, ax3, ax4]
    cls = ['r', 'g', 'b', 'k']
    chan = ['FP1', 'FP2', 'F3', 'C4']
    i = 0
    # channelNames = []

    for ch_idx in range(dat.shape[1]):
        channel_data = dat[:, ch_idx, :]
        lst[i].plot(channel_data[19][1500:1601], cls[i])
        lst[i].set_title(chan[i], color=cls[i])
        i = i + 1

    plt.show()


def visualize_label():
    x = cPickle.load(open("./DataFiles/s02" + ".dat", 'rb'), encoding="bytes")
    field_names = []
    for key in x.keys():
        field_names.append(key)

    labels = x[field_names[0]]
    # Define emotional dimensions and colors
    emotions = ['Valence', 'Arousal', 'Dominance', 'Liking']
    colors = ['r', 'g', 'b', 'k']

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 8))
    fig.suptitle("Label Data Visualization for Each Emotion", fontsize=15, color='purple')

    for idx, ax in enumerate(axes.flatten()):
        # Plot label data for each emotion
        ax.plot(labels[:, idx], color=colors[idx])
        ax.set_title(emotions[idx], color=colors[idx])

    plt.tight_layout()
    plt.show()


def visualize_label_stats():
    x = cPickle.load(open("./DataFiles/s02.dat", 'rb'), encoding="bytes")
    field_names = list(x.keys())
    labels = x[field_names[0]]

    # Calculate statistics for each emotion
    mean_values = np.mean(labels, axis=0)
    variance_values = np.var(labels, axis=0)
    std_dev_values = np.std(labels, axis=0)
    max_values = np.max(labels, axis=0)
    min_values = np.min(labels, axis=0)

    # Define emotional dimensions and colors
    emotions = ['Valence', 'Arousal', 'Dominance', 'Liking']

    # Plot bar graph for each statistic
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    fig.suptitle('Statistics for Each Emotion in Subject-1', fontsize=15, color='purple')

    for i, ax in enumerate(axes.flatten()):
        bars = ax.bar(['Mean', 'Variance', 'Standard Deviation', 'Max', 'Min'],
                      [mean_values[i], variance_values[i], std_dev_values[i], max_values[i], min_values[i]],
                      color=['blue', 'orange', 'green', 'red', 'purple'])
        ax.set_title(emotions[i])
        ax.set_ylabel('Values')

        # Annotate each bar with its value
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',  # Value to display (formatted to 2 decimal places)
                        xy=(bar.get_x() + bar.get_width() / 2, height),  # Position to annotate (center of bar)
                        xytext=(0, 3),  # Offset for text placement (above the bar)
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


