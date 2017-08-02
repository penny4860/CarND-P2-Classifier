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

print(X_train.shape)

import pandas as pd
def int_to_str_label(i):
    dataframe = pd.read_csv("../signnames.csv")
    labels = dataframe['SignName']
    return labels[i]

n_classes = 43
for i in range(n_classes):
    print("{:50s}: {}".format(int_to_str_label(i), len(y_train[y_train==i])))


