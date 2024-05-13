import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.svm_classifier import svm_classifier


def main():
    df = pd.read_csv("./features/features_normalized_2.csv", header=None)
    # df = pd.read_csv("./features_normalized_2.csv", header=None)
    data = np.array(df)
    att = data[:, 0:3]
    labels = data[:, 3:5]

    x_train, x_test, y_train, y_test = train_test_split(att, labels, test_size=0.1, random_state=42)

    # print(x_test.shape,y_test.shape,x_train.shape,y_train.shape)

    svm_classifier(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()

