##############################################
# Programmer: Carter Mooring
# Class: CPCS 322-02, Spring 2021
# Programming Assignment #6
# April 14th, 2021
# 
# Description: 
##############################################

import numpy as np
from numpy.lib.arraysetops import ediff1d
import mysklearn.mypytable as mypytable
import operator
import copy
import random
import itertools
from collections import Counter
from tabulate import tabulate
from operator import itemgetter
import math
import re

def get_att_domain(X_train):
    header = ['att' + str(i) for i in range(len(X_train[0]))]
    attribute_domains = {}
    for i, h in enumerate(header):
        attribute_domains[h] = []
        for x in X_train:
            if x[i] not in attribute_domains[h]:
                attribute_domains[h].append(x[i])

        for k, v in attribute_domains.items():
            attribute_domains[k] = sorted(v)
    return attribute_domains, header
    
def partition_instances(instances, split_attribute, attribute_domains, header):
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is "level"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0
    # lets build a dictionary
    partitions = {} # key (attribute value): value (list of instances with this attribute value)
    # task: try this!
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions
    
def all_same_class(instances):
    label = None
    first_iteration = True
    for row in instances:
        if first_iteration:
            first_iteration = False
            label = row[-1]
        if row[-1] != label:
            return False
    return True

def majority_vote(partition):
    class_count = {}
    print("2", partition)
    for row in partition:
        print("1", row)
        if row[-1] not in class_count:
            class_count[row[-1]] = 0
        class_count[row[-1]] += 1
    class_with_majority = max(class_count.items(), key=itemgetter(1))[0]
    return class_with_majority

def entropy(instances, idx):
    partitions = []
    distinct = []
    for instance in instances:
        if instance[idx] in distinct:
            partition_idx = distinct.index(instance[idx])
            partitions[partition_idx].append(instance)
        else:
            distinct.append(instance[idx])
            partitions.append([instance])
    return partitions

def select_attribute(instances, available_att):
    attribute_entropies = []
    for attribute in available_att:
        idx = int(attribute[-1])
        partitions = entropy(instances, idx)
        entropies = []
        denoms = []
        for partition in partitions:
            distinct_classifiers = []
            classifiers_counts = []
            for instance in partition:
                if instance[-1] in distinct_classifiers:
                    classifier_idx = distinct_classifiers.index(instance[-1])
                    classifiers_counts[classifier_idx] += 1
                else:
                    distinct_classifiers.append(instance[-1])
                    classifiers_counts.append(1)
            denom = len(partition)
            value_entropy = 0
            for count in classifiers_counts:
                if count == 0:
                    value_entropy = 0
                    break
                value_entropy -= count/denom * math.log(count/denom,2)
            entropies.append(value_entropy)
            denoms.append(denom/len(instances))
        
        total_entropy = 0
        for i in range(len(entropies)):
            total_entropy += entropies[i] * denoms[i]

        attribute_entropies.append(total_entropy)

    min_entropy = min(attribute_entropies)
    att_idx = attribute_entropies.index(min_entropy)
    return available_att[att_idx]


def tdidt(current_instances, available_attributes, attribute_domains, header):
    print(current_instances)
    print()
    print(available_attributes)
    print()
    print( attribute_domains)
    print()
    print( header)
    print()
    split_attribute = select_attribute(current_instances, available_attributes)
    available_attributes.remove(split_attribute) # cannot split on same attr twice in a branch
    tree = ['Attribute', split_attribute]

    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)

    prev = []
    for attribute_value, partition in partitions.items():
        value_subtree = ['Value', attribute_value]
        subtree = []
        # TODO: appending leaf nodes and subtrees appropriately to value_subtree
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition): # all same class checks if all the other values equal the first one
            print("0.1")
            subtree = ['Leaf', partition[0][-1], len(partition), len(current_instances)]
            value_subtree.append(subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            print("0.2")
            subtree = ['Leaf', majority_vote(partition), len(partition), len(current_instances)]
            value_subtree.append(subtree)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            print("0.3")
            return ['Leaf', majority_vote(prev), len(partition), len(current_instances)]
        else:
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains.copy(), header.copy())
            value_subtree.append(subtree)
            # need to append subtree to value_subtree and appropriately append value_subtree to tree
        tree.append(value_subtree)
        prev = partition

    return tree

def y_pred(instance, tree):
    y = None
    if tree[0] == 'Attribute':
        y = y_pred(instance, tree[2:])
    if tree[0][0] == 'Value':
        for i in range(len(tree)):
            if tree[i][1] in instance:
                y = y_pred(instance, tree[i][2])
                break
    if tree[0] == 'Leaf':
        return tree[1]
    return y


def Rules(tree, rules, chain, previous_value, class_name):
    if tree[0] == 'Attribute':
        if chain:
            chain += ' AND' + ' ' + str(tree[1]) + ' ' + '==' + ' '
        else:
            chain = 'IF' + ' ' + str(tree[1]) + ' ' + '==' + ' '
        rules = Rules(tree[2:], rules, chain, previous_value, class_name)
    if tree[0][0] == 'Value':
        for i in range(len(tree)):
            if previous_value and previous_value == chain[-len(previous_value):]:
                length = len(previous_value)
                chain = chain[:-length] + ' '
            chain += str(tree[i][1])
            previous_value = str(tree[i][1])
            rules = Rules(tree[i][2], rules, chain, previous_value, class_name)
    if tree[0] == 'Leaf':
        chain += ' THEN' + ' ' + class_name + ' ' + '=' + ' ' + str(tree[1])
        chain = re.sub(' +', ' ', chain)
        rules.append(chain)
    return rules





def convert_rating(mpg_list):
    """Gets the ratings of each list value
    """
    # fpr each index in the list
    for i in range(len(mpg_list)):
        mpg_list[i] = get_rating(mpg_list[i])
    return mpg_list
        
def get_rating(mpg):
    """determins the rating depending on the mpg
    """
    if mpg < 14:
        return 1
    elif mpg < 15:
        return 2
    elif mpg < 17:
        return 3
    elif mpg < 20:
        return 4
    elif mpg < 24:
        return 5
    elif mpg < 27:
        return 6
    elif mpg < 31:
        return 7
    elif mpg < 37:
        return 8
    elif mpg < 45:
        return 9
    return 10

def convert_weight(weight):
    """Converts the weight
    """
    res = []
    for val in weight:
        res.append(get_weight(val))
    return res

def get_weight(val):
    """Weight converting helper function
    """
    if val < 2000:
        category = 1
    elif val < 2500:
        category = 2
    elif val < 3000:
        category = 3
    elif val < 3500:
        category = 4
    else:
        category = 5
    return category

def get_rand_rows(table, num_rows):
    """get random rows from the table
    """
    rand_rows = []
    # for each index in the rumber of rows
    for i in range(num_rows):
        rand_rows.append(table.data[random.randint(0,len(table.data))-1])
    return rand_rows

def prediction_pretty_print(rows, actual, predicted):
    """print helper function
    """
    for i in range(len(rows)):
        print('instance:', rows[i])
        print('class:', predicted[i], 'actual:', actual[i])



def get_accuracy(actual, predicted):
    """gets the accuracy of our predicted value
    """
    predicted_correct = 0
    # for each index in the actual result
    for i in range(len(actual)):
        # if actual is the same as predicted
        if actual[i] == predicted[i]:
            predicted_correct+=1
    return predicted_correct/len(actual)

def get_from_folds(X_vals, y_vals, train_folds, test_folds):
    """values from the folds
    """
    X_train = []
    y_train = []
    for row in train_folds:
        for i in row:
            X_train.append(X_vals[i])
            y_train.append(y_vals[i])

    X_test = []
    y_test = []
    for row in test_folds:
        for i in row:
            X_test.append(X_vals[i])
            y_test.append(y_vals[i])

    return X_train, y_train, X_test, y_test

def print_tabulate(table, headers):
    print(tabulate(table, headers, tablefmt="rst"))

def add_conf_stats(matrix):
    del matrix[0]
    for i,row in enumerate(matrix):
        row[0] = i+1
        row.append(sum(row))
        row.append(round(row[i+1]/row[-1]*100,2))
        
def titanic_stats(matrix):
    for i,row in enumerate(matrix):
        row.append(sum(row))
        row.append(round(row[i]/row[-1]*100,2))
        row.insert(0, i+1)
    matrix.append(['Total', matrix[0][1]+matrix[1][1], matrix[0][2]+matrix[1][2], matrix[0][3]+matrix[1][3], \
                   round(((matrix[0][1]+matrix[1][2])/(matrix[0][3]+matrix[1][3])*100),2)])
    
def weightedRandom(y_train, X_test):
    randomWeightedChoice = [y_train[random.randint(0,len(y_train)-1)] for _ in X_test]
    #print(randomWeightedChoice)
    return randomWeightedChoice
    
def mean(x):
    """Computes the mean of a list of values
    """
    return sum(x)/len(x)

def compute_slope_intercept(x, y):
    """
    """
    mean_x = mean(x)
    mean_y = mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return float(m), float(b)

def compute_euclidean_distance(v1, v2):
    """
    """
    assert len(v1) == len(v2)
    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist

def scale(vals, test_vals):
    """
    """
    scaled_vals_list = []
    maxs_list = []
    mins_list = []

    # for each list in list vals, get each values max and min and store in a list accordingly
    for i in range(len(vals[0])):
        maxs_list.append(max([val[i] for val in vals]))
        mins_list.append(min([val[i] for val in vals]))

    # for each list in list vals, scale each value according to the max and min to be between [0, 1]
    for row in vals:
        curr = []
        for i in range(len(row)):
            curr.append((row[i]-mins_list[i])/(maxs_list[i]-mins_list[i]))
        scaled_vals_list.append(curr)

    # for each list in list test_vals, scale each value according to the max and min to be between [0, 1]
    for row in test_vals:
        curr = []
        for i in range(len(row)):
            curr.append((row[i]-mins_list[i])/(maxs_list[i]-mins_list[i]))
        scaled_vals_list.append(curr)
    
    # returns all scaled values from the vals list, then the scaled values from the test_vals list
    return scaled_vals_list[:len(vals)], scaled_vals_list[len(vals):]

def kneighbors_prep(scaled_X_train, scaled_X_test, n_neighbors):
    """
    """
    scaled_X_train = copy.deepcopy(scaled_X_train)
    scaled_X_test = copy.deepcopy(scaled_X_test)

    # for each scaled list in scaled_X_train
    for i, instance in enumerate(scaled_X_train):
        distance = compute_euclidean_distance(instance, scaled_X_test) 
        instance.append(i)  # append the original row index
        instance.append(distance)   # append the distance
    
    
    scaled_X_train_sorted = sorted(scaled_X_train, key=operator.itemgetter(-1)) # sort the list in assending order
    top_k = scaled_X_train_sorted[:n_neighbors] # get a list of the top_k neighbors

    distances_list = []
    indices_list = []

    # for each row in the top_k list, append the distances and indices to their own lists
    for row in top_k:
        distances_list.append(row[-1])
        indices_list.append(row[-2])
    
    # return the distances and indices lists
    return distances_list, indices_list

def get_label(labels):
    """
    """
    label_types = []
    # for each value in the labels list
    for val in labels:
        # if we have not see that label type
        if val not in label_types:
            label_types.append(val) # append to list of label types
    
    count_list = [0 for label_type in label_types]

    # for value in label types
    for i, val in enumerate(label_types):
        for label in labels:
            # if the value is == to the label then incremept the count for that position
            if val == label:
                count_list[i] += 1

    max_count = 0
    label_prediction = ''
    # for value in count_list
    for i, val in enumerate(count_list):
        if val > max_count:
            label_prediction = label_types[i]
 
    return label_prediction

def get_unique(vals):
    """
    """
    unique = []
    # for values in the vals list
    for val in vals:
        if val not in unique:
            unique.append(val)
    return unique

def group_by(x_train, y_train):
    """
    """
    unique = get_unique(y_train)
    grouped = [[] for _ in unique]
    # for each value in y_train
    for i, val in enumerate(y_train):
        for j, label in enumerate(unique):
            if val == label:
                grouped[j].append(i)
    return grouped

def shuffle(X, y):
    """
    """
    for i in range(len(X)):
        rand_index = random.randrange(0, len(X)) # [0, len(X))
        X[i], X[rand_index] = X[rand_index], X[i] # this is the temporary value swap but in one line
        if y is not None:
            y[i], y[rand_index] = y[rand_index], y[i]

def get_from_folds(X_vals, y_vals, train_folds, test_folds):
    """
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # for each fold
    for row in train_folds:
        for i in row:
            X_train.append(X_vals[i])
            y_train.append(y_vals[i])

    # for each test fold
    for row in test_folds:
        for i in row:
            X_test.append(X_vals[i])
            y_test.append(y_vals[i])

    return X_train, y_train, X_test, y_test

# TODO: your reusable general-purpose functions here
def get_column(table, i):
    res = []
    for row in table:
        res.append(row[i])
    return res

def get_unique(vals):
    unique = []
    for val in vals:
        if val not in unique:
            unique.append(val)
    return unique

def priors(y_train):
    unique = get_unique(y_train)
    res = {}
    for val in unique:
        res[val] = 0
    for val in y_train:
        for u in unique:
            if val == u:
                res[u] += 1
    for u in unique:
        res[u] /= len(y_train)
    return res

def posteriors(X_train, y_train, priors):
    posteriors = {} # create initial outermost dictionary

    # for each key in the priors (y-train) dictionary
    for key, v in priors.items():
        #print("key:", key, "    val", v, "    X_train[0]:", X_train[0], "    len(X_train[0]):", len(X_train[0]))
        posteriors[key] = {}
        # for the amount of values in the current X_train list (e.g. [1, 5] len = 2)
        for i in range(len(X_train[0])):
            posteriors[key][i] = {}
    
    #print()
    # for the length of X_train
    for j in range(len(X_train)):
        # for the amount of values in the current X_train list
        for k in range(len(X_train[j])):
            prior_label = y_train[j]    # store the y_train value for the current X_train position
            posterior_label = X_train[j][k] # store the current value in the current X_train list
            #print("     y_train[j]:", y_train[j])
            #print("      X_train[j][k]:", X_train[j][k])
            denominator = priors[prior_label] * len(y_train)    # stores the denominator based on mulitplying y_train label odds by the length of y table
            #print("      priors[prior_label]:", priors[prior_label], "    len(y_train):", len(y_train), "     denominator:", denominator)
            #print()
            # if the current value in the current X_train list   exists in   current table then give it its posterior value 
            #print("if else posterior_label:", posterior_label, "    posteriors[prior_label][k]:", posteriors[prior_label][k])
            if posterior_label in posteriors[prior_label][k]:
                #print("     posteriors[prior_label][k][posterior_label]:", posteriors[prior_label][k][posterior_label])
                posteriors[prior_label][k][posterior_label] = ((posteriors[prior_label][k][posterior_label] * denominator) + 1) / denominator
                #print("     posteriors[prior_label][k][posterior_label]2:", posteriors[prior_label][k][posterior_label])
            else:
                posteriors[prior_label][k][posterior_label] = 1 / denominator
                #print("     posteriors[prior_label][k][posterior_label]4:", posteriors[prior_label][k][posterior_label])
    return posteriors

def get_prediction_index(vals):
    max_index = 0
    for i in range(len(vals)):
        if vals[i] > vals[max_index]:
            max_index = i
    return max_index