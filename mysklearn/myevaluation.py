##############################################
# Programmer: Carter Mooring
# Class: CPCS 322-02, Spring 2021
# Programming Assignment #6
# April 14th, 2021
# 
# Description: 
##############################################


import mysklearn.myutils as myutils
import numpy as np
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        # TODO: seed your random number generator
        # you can use the math module or use numpy for your generator
        # choose one and consistently use that generator throughout your code
        np.random.seed(0)
        pass
    
    if shuffle: 
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        myutils.shuffle(X,y)
        pass

    # Split the Tests off
    num_instances = len(X) # ex: 8
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size) # cell (8*0.33)

    split_index = num_instances - test_size # 8 - 2 = 6
    # x[:split_index] starts at [0 and goes to  6) aka 0->5
    # x[split_index:] starts at [6 and goes to  8) aka 6->7
    # same for y but for secont list
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    n = len(X)
    X_train_folds = []
    X_test_folds = []
    sample_size = []
    full_folds = n % n_splits

    # for index in the range of splits
    for i in range(n_splits):
        if i >= full_folds:
            sample_size.append(n // n_splits)
        else:
            sample_size.append((n // n_splits) + 1)
 
    # for index in the range of splits
    for i in range(n_splits):
        indices = [j for j in range(len(X))]
        range_size = sample_size[i]
        start_index = sum(sample_size[n] for n in range(i))
        test_fold = [k for k in range(start_index, start_index + range_size)]
        X_test_folds.append(test_fold)
        del indices[start_index:start_index + range_size]
        X_train_folds.append(indices)
 
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    total_folds = [[] for _ in range(n_splits)]
    X_train_folds = [[] for _ in range(n_splits)]
    X_test_folds = [[] for _ in range(n_splits)]
    curr = 0
    groups = myutils.group_by(X,y)

    # get the stratified index sets
    for group in groups:
        for i in group:
            total_folds[curr].append(i)
            curr = (curr + 1) % n_splits

    curr = 0
    i = 0
    # for index in the range of splits
    for j in range(n_splits):
        # for each fold
        for i, fold in enumerate(total_folds):
            # if i is not equal to j
            if(i != j):
                for val in fold:
                    X_train_folds[curr].append(val)
            else:
                X_test_folds[curr] = fold
        curr += 1
 
    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    # for each each range  value in labels
    for lab in range(len(labels)):
        matrix_row = []
        for labe in range(len(labels)):
            matrix_row.append(0)
        matrix.append(matrix_row)
 
    # for each each range value in labels
    for i in range(len(labels)):
        label = labels[i]
        for j in range(len(y_true)):
            if y_true[j] == label:
                index = labels.index(y_pred[j])
                matrix[i][index] = matrix[i][index] + 1
    return matrix

