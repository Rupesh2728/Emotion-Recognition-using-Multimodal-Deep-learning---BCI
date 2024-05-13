import os
import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt


def apply_visualizations_to_files(data_files_dir, output_dir):
    for file_name in os.listdir(data_files_dir):
        if file_name.endswith(".dat"):
            subject_id = file_name.split(".")[0]
            print(f"Processing subject {subject_id}...")
            visualize_channel(data_files_dir, subject_id, output_dir)
            visualize_label(data_files_dir, subject_id, output_dir)
            visualize_label_stats(data_files_dir, subject_id, output_dir)
            print(f"Visualizations saved for subject {subject_id}")


def visualize_channel(data_files_dir, subject_id, output_dir):
    file_path = os.path.join(data_files_dir, subject_id + ".dat")
    x = cPickle.load(open(file_path, 'rb'), encoding="bytes")
    field_names = []
    for key in x.keys():
        field_names.append(key)

    data = x[field_names[1]]
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
    for ch_idx in range(dat.shape[1]):
        channel_data = dat[:, ch_idx, :]
        lst[i].plot(channel_data[19][1500:1601], cls[i])
        lst[i].set_title(chan[i], color=cls[i])
        i = i + 1
    plt.tight_layout()
    output_file_path = os.path.join(output_dir, f"{subject_id}_channel.png")
    plt.savefig(output_file_path)
    plt.close()


def visualize_label(data_files_dir, subject_id, output_dir):
    file_path = os.path.join(data_files_dir, subject_id + ".dat")
    x = cPickle.load(open(file_path, 'rb'), encoding="bytes")
    field_names = []
    for key in x.keys():
        field_names.append(key)

    labels = x[field_names[0]]
    emotions = ['Valence', 'Arousal', 'Dominance', 'Liking']
    colors = ['r', 'g', 'b', 'k']
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 8))
    fig.suptitle(f"Label Data Visualization for Each Emotion - Subject {subject_id}", fontsize=15, color='purple')
    for idx, ax in enumerate(axes.flatten()):
        ax.plot(labels[:, idx], color=colors[idx])
        ax.set_title(emotions[idx], color=colors[idx])
    plt.tight_layout()
    output_file_path = os.path.join(output_dir, f"{subject_id}_label.png")
    plt.savefig(output_file_path)
    plt.close()


def visualize_label_stats(data_files_dir, subject_id, output_dir):
    file_path = os.path.join(data_files_dir, subject_id + ".dat")
    x = cPickle.load(open(file_path, 'rb'), encoding="bytes")
    field_names = list(x.keys())
    labels = x[field_names[0]]
    mean_values = np.mean(labels, axis=0)
    variance_values = np.var(labels, axis=0)
    std_dev_values = np.std(labels, axis=0)
    max_values = np.max(labels, axis=0)
    min_values = np.min(labels, axis=0)
    emotions = ['Valence', 'Arousal', 'Dominance', 'Liking']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    fig.suptitle(f'Statistics for Each Emotion in Subject {subject_id}', fontsize=15, color='purple')
    for i, ax in enumerate(axes.flatten()):
        bars = ax.bar(['Mean', 'Variance', 'Standard Deviation', 'Max', 'Min'],
                      [mean_values[i], variance_values[i], std_dev_values[i], max_values[i], min_values[i]],
                      color=['blue', 'orange', 'green', 'red', 'purple'])
        ax.set_title(emotions[i])
        ax.set_ylabel('Values')
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.tight_layout()
    output_file_path = os.path.join(output_dir, f"{subject_id}_stats.png")
    plt.savefig(output_file_path)
    plt.close()


# Example usage:
data_files_dir = "./DataFiles"
output_dir = "./DataVisualisation"
apply_visualizations_to_files(data_files_dir, output_dir)
