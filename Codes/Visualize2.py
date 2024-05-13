import _pickle as cPickle
import matplotlib.pyplot as plt

def visualize_label():
    x = cPickle.load(open("./DataFiles/s02" + ".dat", 'rb'), encoding="bytes")
    field_names = []
    for key in x.keys():
        field_names.append(key)
    labels = x[field_names[0]]
    print(labels)
    # print(x[field_names[1]])
    print(len(x[field_names[1]][0][0]))
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


visualize_label()
