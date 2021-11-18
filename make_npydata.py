import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

try:
    train_path = './data/train/'
    test_path = './data/val/'

    train_list = []
    for filename in os.listdir(train_path):
        if filename.split('.')[1] == ('jpg' or 'JPG'):
            train_list.append(train_path + filename)

    train_list.sort()
    np.save('./npydata/train.npy', train_list)

    test_list = []
    for filename in os.listdir(test_path):
        if filename.split('.')[1] == ('jpg' or 'JPG'):
            test_list.append(test_path + filename)
    test_list.sort()
    np.save('./npydata/test.npy', test_list)

    print("generate image list successfully", len(train_list), len(test_list))
except:
    print("The dataset path is wrong. Please check you path.")

