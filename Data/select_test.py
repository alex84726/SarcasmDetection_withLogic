import random

import numpy as np

TEST_PERCENTAGE = 0.1
UNBALANCED_PERC = 0.25
random.seed(0)
if __name__ == '__main__':
    pos = np.load('sarcasm_data_proc.npy')
    print("length of pos", len(pos))
    neg = np.load('nonsarc_data_proc.npy')
    print("length of pos", len(neg))
    length = int(len(pos) * TEST_PERCENTAGE) + 1
    '''
    select balanced data
    '''
    sample_pos_test = random.sample(range(len(pos)), length)
    train_balanced = []
    test_balanced = []
    train_unbalanced = []
    test_unbalanced = []
    for i in range(len(pos)):
        if i not in sample_pos_test:
            train_balanced.append([1, pos[i]])
            train_unbalanced.append([1, pos[i]])
        else:
            test_balanced.append([1, pos[i]])
            test_unbalanced.append([1, pos[i]])
    # print("length of train_balanced", len(train_balanced))
    # print("length of test_balanced", len(test_balanced))
    sample_neg = random.sample(range(len(neg)), len(pos))
    counter = 0
    for i in range(len(neg)):
        if i in sample_neg:
            if counter < len(sample_pos_test):
                test_balanced.append([0, neg[i]])
                counter += 1
            else:
                train_balanced.append([0, neg[i]])
    random.shuffle(train_balanced)
    random.shuffle(test_balanced)
    print("length of train_balanced", len(train_balanced))
    print("length of test_balanced", len(test_balanced))
    np.save('train_balanced.npy', train_balanced)
    np.save('test_balanced.npy', test_balanced)
    '''
    select unbalanced data
    '''
    length_unbalanced = len(pos) // UNBALANCED_PERC
    sample_neg = random.sample(range(len(neg)), int(length_unbalanced))
    counter = 0
    for i in range(len(neg)):
        if i in sample_neg:
            if counter < int(len(sample_pos_test) // UNBALANCED_PERC):
                test_unbalanced.append([0, neg[i]])
                counter += 1
            else:
                train_unbalanced.append([0, neg[i]])
    random.shuffle(train_unbalanced)
    random.shuffle(test_unbalanced)
    np.save('train_unbalanced.npy', train_unbalanced)
    np.save('test_unbalanced.npy', test_unbalanced)
    print("length of train_unbalanced", len(train_unbalanced))
    print("length of test_unbalanced", len(test_unbalanced))
