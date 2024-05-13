import os
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt


def aggregate_statistics(data_files_dir):
    all_mean_values = []
    all_variance_values = []
    all_max_values = []
    all_min_values = []

    for file_name in os.listdir(data_files_dir):
        if file_name.endswith(".dat"):
            file_path = os.path.join(data_files_dir, file_name)
            x = cPickle.load(open(file_path, 'rb'), encoding="bytes")
            labels = x[list(x.keys())[0]]
            mean_values = np.mean(labels, axis=0)
            variance_values = np.var(labels, axis=0)
            max_values = np.max(labels, axis=0)
            min_values = np.min(labels, axis=0)

            all_mean_values.append(mean_values)
            all_variance_values.append(variance_values)
            all_max_values.append(max_values)
            all_min_values.append(min_values)

    all_mean_values = np.array(all_mean_values)
    all_variance_values = np.array(all_variance_values)
    all_max_values = np.array(all_max_values)
    all_min_values = np.array(all_min_values)

    return all_mean_values, all_variance_values, all_max_values, all_min_values


def plot_aggregated_statistics(all_mean_values, all_variance_values, all_max_values, all_min_values):
    emotions = ['Valence', 'Arousal', 'Dominance', 'Liking']
    stats = ['Mean', 'Variance', 'Maximum', 'Minimum']
    colors = ['blue', 'orange', 'green', 'red']

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15), sharex=True)
    fig.suptitle('Aggregated Statistics for All Subjects', fontsize=16, color='purple')

    for i, (emotion, mean_values, variance_values, max_values, min_values) in enumerate(
            zip(emotions, all_mean_values.T, all_variance_values.T, all_max_values.T, all_min_values.T)):
        ax = axes[i]
        bar_width = 0.2
        indices = np.arange(len(stats))

        ax.bar(indices - 1.5 * bar_width, mean_values, bar_width, color=colors[0], alpha=0.6, label='Mean')
        ax.bar(indices - 0.5 * bar_width, variance_values, bar_width, color=colors[1], alpha=0.6, label='Variance')
        ax.bar(indices + 0.5 * bar_width, max_values - min_values, bar_width, color=colors[2], alpha=0.6, label='Range')

        ax.set_ylabel('Values')
        ax.set_title(emotion)
        ax.set_xticks(indices)
        ax.set_xticklabels(stats)
        ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


# Example usage:
data_files_dir = "./DataFiles"
all_mean_values, all_variance_values, all_max_values, all_min_values = aggregate_statistics(data_files_dir)
plot_aggregated_statistics(all_mean_values, all_variance_values, all_max_values, all_min_values)
