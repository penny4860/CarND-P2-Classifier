# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "../dataset/train.p"
validation_file= "../dataset/valid.p"
testing_file = "../dataset/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(X, y, labels, title):
    def _get_n_samples_for_classes(y, n_classes):
        n_samples = []
        for i in range(n_classes):
            n_samples.append(len(y[y==i]))
        return n_samples

    plt.rcdefaults()
    fig, ax = plt.subplots()

    y_pos = np.arange(len(labels))
    n_samples_for_classes = _get_n_samples_for_classes(y, len(labels))

    ax.barh(y_pos, n_samples_for_classes, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('number of samples for each label')
    ax.set_title('{} Sample Distribution'.format(title))
     
    plt.show()


dataframe = pd.read_csv("../signnames.csv")
labels = list(dataframe['SignName'])
plot_histogram(X_train, y_train, labels, "Training")
plot_histogram(X_valid, y_valid, labels, "Valid")
plot_histogram(X_test, y_test, labels, "Test")



