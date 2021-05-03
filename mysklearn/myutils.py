import copy
import operator
import math
import re
from functools import reduce
from tabulate import tabulate
from operator import itemgetter
import random

def compute_slop_intercept(x, y):
    """Computes the slop intercet of two data sets

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
    Returns:
        m: the m in the equation y = mx + b
        b: the b in the equation y = mx + b
    """
    x_mean = sum(x)/len(x)
    y_mean = sum(y)/len(y)
    n = len(x)

    m_numer = sum([xi*yi for xi,yi in zip(x, y)]) - n * x_mean * y_mean
    m_denom = sum([xi**2 for xi in x]) - n * x_mean**2

    m = m_numer / m_denom
    b = y_mean - m * x_mean

    return m, b

def kneighbors_helper(scaled_train, scaled_test, n_neighbors):
    """Helper function for KNN to compute distances and indicies.

    Args:
        scaled_train(list of list of obj): list of training set values
        scaled_test(list of obj): List of testing set values
        n_neighbors(int): amount of folds
    Returns:
        distace: List of distance values
        indicies: List of indice values
    """
    # deep copy so you don't modify the original
    scaled_train_copy = copy.deepcopy(scaled_train)
    scaled_test_copy = copy.deepcopy(scaled_test)
    for i, instance in enumerate(scaled_train_copy):
        # append the original row index
        instance.append(i)
        # append the distance
        dist = compute_euclidean_distance(instance[:-1], scaled_test_copy)
        instance.append(dist)
    
    train_sorted = sorted(scaled_train_copy, key=operator.itemgetter(-1))
 
    top_k = train_sorted[:n_neighbors]
    distances = []
    indices = []
    for row in top_k:
        distances.append(row[-1])
        indices.append(row[-2])
    
    return distances, indices

def compute_euclidean_distance(v1, v2):
    """Computes the euclidian distance of two points

    Args:
        v1(List of ints or floats): List of points 
        v2(Lists of ints of floats): List of points
    Returns:
        the average Euclidian distance of all points
    """
    assert len(v1) == len(v2)

    dist = (sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))])) ** (1/2)
    return dist 

def get_rand_rows(table, num_rows):
    """gets specified number of random rows

    Args:
         table(List of obj): data set
         num_rows(int): number of random rows to get
    Returns:
            rand_rows: the specified number of rows
    """
    rand_rows = []
    for _ in range(num_rows):
        rand_rows.append(table.data[random.randint(0,len(table.data))-1])
    return rand_rows

def scale(vals, test_vals):
    """Scales all values passed in

    Args:
         vals(list of list of numeric vals): The list of test instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
         test_vals(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
    Returns:
        res[:len(vals)]: List of all train values scaled
        res[len(vals):]: List of all test values scaled
    """
    res = []
    max_vals = []
    min_vals = []
    for i in range(len(vals[0])):
        max_vals.append(max([val[i] for val in vals]))
        min_vals.append(min([val[i] for val in vals]))
    for row in vals:
        curr = []
        for i in range(len(row)):
            curr.append((row[i]-min_vals[i])/(max_vals[i]-min_vals[i]))
        res.append(curr)
    for row in test_vals:
        curr = []
        for i in range(len(row)):
            curr.append((row[i]-min_vals[i])/(max_vals[i]-min_vals[i]))
        res.append(curr)
    return res[:len(vals)], res[len(vals):]

def get_label(labels):
    """Gets all the unique labels from the list passsed in 

    Args:
         labels(List of values): List of all labels
    Returns:
        res: List of all unique labels
    """
    unique_labels = []
    for val in labels:
        if val not in unique_labels:
            unique_labels.append(val)
    counts = [0 for _ in unique_labels]
    for i, val in enumerate(unique_labels):
        for lab in labels:
            if val == lab:
                counts[i] += 1
    max_count = 0
    res = ''
    for i, val in enumerate(counts):
        if val > max_count:
            res = unique_labels[i]
 
    return res

def randomize_in_place(alist, parallel_list=None):
    """Randomixes two list in parallel

    Args:
         alist: List of values
         parallel_list: List of values but could be none
    """
    for i in range(len(alist)):
        # generate a random index to swap the element at i with
        rand_index = random.randrange(0, len(alist)) # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def get_from_folds(X_vals, y_vals, train_folds, test_folds):
    """This method gives me my training sets and my testing sets

    Args:
        X_vals: list of x values
        y_vals: List of y values
        train_folds: List of training folds
        test_folds: List of testing folds
         
    Returns:
        X_train: List of X trining sets
        y_train: List of y training sets
        X_test: List of X test sets
        y_test List of y test sets
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

def group_by(x, y):
    """Groups x based on the values in y

    Args:
         x: List of values
         y: List of values
    Returns:
        grouped: list of all values in there grouped order 
    """
    group1 = []
    group2 = []
    grouped = []
    instance = y[0]

    for i in range(len(y)):
        if y[i] == instance:
            group1.append(i)
        else:
            group2.append(i)

    
    grouped.append(group1)
    grouped.append(group2)
            
    return grouped

def add_conf_stats(matrix):
    """adds stats to our matrix correctly

    Args:
        matrix: 2D list of values
    """
    del matrix[0]
    for i,row in enumerate(matrix):
        row[0] = i+1
        row.append(sum(row))
        row.append(round(row[i+1]/row[-1]*100,2))

def print_tabulate(table, headers):
    """Prints our matrix nicely

    Args:
        table: List of all table values
        headers: List of our column headers
    """
    print(tabulate(table, headers, tablefmt="rst"))

def get_priors(y_train):
    """Gets priors based on y_train results passed in

    Args:
        y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
    Returns:
        priors_dict: This is a dictionary filled with all the priors
    """
    unique = []
    counts = []
    for label in y_train:
        if label in unique:
            index = unique.index(label)
            counts[index] = counts[index] + 1
        else:
            unique.append(label)
            counts.append(1)
    
    denom = len(y_train)
    priors_dict = {}
    for i in range(len(unique)):
        label = unique[i]
        priors_dict[label] = counts[i]/denom
    
    return priors_dict

def get_posteriors(X_train, y_train, priors):
    """Gets posteriors based on X_train, y_train and prior results passed in

    Args:
        X_train(list of list of obj): The list of training instances (samples). 
            The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train)
            The shape of y_train is n_train_samples
        priors(dictionary): The prior probabilities computed for each
            label in the training set.
    Returns:
        priors_dict: This is a dictionary filled with all the priors
    """
    posteriors = {}

    for k, _ in priors.items():
        posteriors[k] = {}
        for i in range(len(X_train[0])):
            posteriors[k][i] = {}
    
    for j in range(len(X_train)):
        for i in range(len(X_train[j])):
            prior_label = y_train[j]
            posterior_label = X_train[j][i]
            denom = priors[prior_label] * len(y_train)
            if posterior_label in posteriors[prior_label][i]:
                posteriors[prior_label][i][posterior_label] = ((posteriors[prior_label][i][posterior_label] * denom) + 1) / denom
            else:
                posteriors[prior_label][i][posterior_label] = 1 / denom
    return posteriors

def multiply(a, b):
    """Returns product of both number passed in

    Args:
        a(int): Any number passed in my user
        b(int): Any number passed in by user
    Returns:
        a*b
    """
    return a*b

def compute_probs(test, priors, posteriors):
    """Computes probability of outcome based on test, priors, and posteriors

    Args:
        test (list of list of obj): The list of testing instances (samples). 
        priors(dictionary): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Returns:
        probs_dictionary: Dictionary of all probability values
    """
    probs_dictionary = {}
    for k, v in priors.items():
        prior = v
        dictionary = posteriors[k]
        probs = []
        probs.append(prior)
        for i in range(len(test)):
            if test[i] in dictionary[i]:  
                probs.append(dictionary[i][test[i]])
            else:
                probs.append(0)
        probability = reduce(multiply, probs)
        probs_dictionary[k] = probability
    
    return probs_dictionary

def predict_from(probs_dictionary):
    """Computes prediction values based on probability values passed in

    Args:
        probs_dictionary(dictionary): Dictionary of all probability values
    Returns:
        prediction(string): prediction based on probability values
    """
    max = 0
    prediction = ""
    for k, v, in probs_dictionary.items():
        if v >= max:
            prediction = k
            max = v
    return prediction

def convert_to_rating(mpg_list):
    """Accepts a list of mpg values and rankes them based on a rating metric

    Args:
        mpg_list(List): various mpg values
    Returns:
        mpg_list(list): List of all mpg values replaced by there coresponding rating
    """
    for i in range(len(mpg_list)):
        mpg_list[i] = get_rating(mpg_list[i])
    return mpg_list
        
def get_rating(mpg):
    """ Accepts a mpg value and returns the rating for that value 

    Args:
        mpg(int/float): unique mpg value 
    Returns:
        a rating from 1 - 10
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

def convert_to_rank(list):
    for i in range(len(list)):
        list[i] = get_rank(list[i])
    return list

def get_rank(pop):
    if pop < 11:
        return 1
    elif pop < 21:
        return 2
    elif pop < 31:
        return 3
    elif pop < 41:
        return 4
    elif pop < 51:
        return 5
    elif pop < 61:
        return 6
    elif pop < 71:
        return 7
    elif pop < 81:
        return 8
    elif pop < 91:
        return 9
    else:
        return 10

def convert_weight(weight):
    """Accepts a list of weight values and rankes them based on a rating metric 

    Args:
        weight(List): various weight values
    Returns:
        res(list): all weight values converted to rating
    """
    res = []
    for val in weight:
        res.append(get_weight(val))
    return res

def get_weight(val):
    """ Accepts a weight value and returns the rating for that value 

    Args:
        val(int/float): unique weight value 
    Returns:
        cur(int): a rating from 1 - 5
    """
    if val < 2000:
        curr = 1
    elif val < 2500:
        curr = 2
    elif val < 3000:
        curr = 3
    elif val < 3500:
        curr = 4
    else:
        curr = 5
    return curr

def get_accuracy(y_split, predicted):
    """takes in y_split and predicted values and computes the accuracy of our predictions

    Args:
        y_split(list): list of y split values
        predicted(list): list of all predicted values
    Returns:
        accuracy of predictions
    """
    correct = 0
    for index in range(len(y_split)):
        if y_split[index] == predicted[index]:
            correct += 1
    return correct / len(predicted)



def titanic_stats(matrix):
    """creates a row that displays the total of all columns in a matrix at the bottom of the matric passed in

    Args:
        matrix(list): a list of values that are displayed as a confusion matrix
    """
    for i,row in enumerate(matrix):
        row.append(sum(row))
        row.append(round(row[i]/row[-1]*100,2))
        row.insert(0, i+1)
    matrix.append(['Total', matrix[0][1]+matrix[1][1], matrix[0][2]+matrix[1][2], matrix[0][3]+matrix[1][3], \
                   round(((matrix[0][1]+matrix[1][2])/(matrix[0][3]+matrix[1][3])*100),2)])

def get_unique(vals):
    """get all unique values from list passed in

    Args:
        vals(list): list of values all of same type
    Returns:
        unique(list): all unique values in vals
    """
    unique = []
    for val in vals:
        if val not in unique:
            unique.append(val)
    return unique

def get_att_domain(X_train):
    """get attribute domains based on X_train

    Args:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
    Returns:
        Attribute_domains(Dict): domain of evvery attribute
        header(list): att + str(i) for the lengh if X_train[0]
    """
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


def entropy(instances, idx):
    """calculates entropy based on distance and index passed in

    Args:
        instances(list): list of values
    Returns:
        partitions(list): list of partitions
    """
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
    """Selects which attribute to split on using entropy
    Args:
        instances(list): list of values
        available_att(list): list of available attributes
    Returns:
        available_att[att_idx](str): specific attribute to split on
    """
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

def partition_instances(instances, split_attribute, attribute_domains, header):
    """Selects a partition instance
    Args:
        instances(list): list of instances
        split_attribute(str): attribute to split on
        attribute_domains(list): list of available attributes
        header(list): list of header names
    Returns:
        partitions(dic): dictionary of all partitions in tree
    """
    attribute_domain = attribute_domains[split_attribute]
    attribute_index = header.index(split_attribute)
    partitions = {}
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions

def check_all_same_class(instances):
    """checks all instances to see if they are in same class
    Args:
        instances(list): list of instances
    Returns:
        true: if in same class false otherwise
    """
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True

def majority_vote(partition):
    """computes majority vote in the case of a Case 2
    Args:
        partitions(dic): dictionary of all partitions in tree
    Returns:
        class_with_majority: majority class name
    """
    class_count = {}
    for row in partition:
        if row[-1] not in class_count:
            class_count[row[-1]] = 0
        class_count[row[-1]] += 1
    class_with_majority = max(class_count.items(), key=itemgetter(1))[0]
    return class_with_majority

def tdidt(current_instances, available_attributes, attribute_domains, header):
    """computes TDIDT on a set of values
    Args:
        instances(list): list of instances
        available_attributes(list): list of available attributes 
        attribute_domains(list): list of available attributes
        header(list): list of header names
    Returns:
        tree(nested list): tree representation of data
    """
    split_attribute = select_attribute(current_instances, available_attributes)
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute]

    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)
    classes_count = {}
    total_instances = 0
    for _, values in partitions.items():
        total_instances += len(values)
        for val in values:
            if val:
                if val[-1] not in classes_count:
                    classes_count[val[-1]] = 0
                classes_count[val[-1]] += 1
    majority_class = max(classes_count.items(), key=itemgetter(1))
    m_vote, majority_count = majority_class[0], majority_class[1]

    for attribute_value, partition in partitions.items():
        value_subtree = ["Value", attribute_value]
        if len(partition) > 0 and check_all_same_class(partition):
            value_subtree.append(["Leaf", partition[0][-1], len(partition), total_instances])
        elif len(partition) > 0 and len(available_attributes) == 0:
            majority_class = majority_vote(partition)
            value_subtree.append(["Leaf", majority_class, len(partition), total_instances])
        elif len(partition) == 0:
            tree = ["Leaf", m_vote, majority_count, total_instances]
            return tree 
        else:
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains, header)
            value_subtree.append(subtree)
        tree.append(value_subtree)
    return tree

def classifySample(instance, tree):
    """ Classifies samples based on instances and a tree
    Args:
        instances(list): list of instances
        tree(nested list): tree representation of data
    Returns:
        y: classify Sample
    """
    y = None
    if tree[0] == 'Attribute':
        y = classifySample(instance, tree[2:])
    if tree[0][0] == 'Value':
        for i in range(len(tree)):
            if tree[i][1] in instance:
                y = classifySample(instance, tree[i][2])
                break
    if tree[0] == 'Leaf':
        return tree[1]
    return y

def Get_Rules(tree, rules, chain, previous_value, class_name):
    """ prints out rules of the tree passed in
    Args:
        tree(nested list): tree representation of data
        rules(list): empty list for rules
        chain(str): chain sting
        previous_vlaue(str): predifined empty string value
        class_name(str): name of specific class
    Returns:
        rules(list): list of all rules that will help read the tree
    """
    if tree[0] == 'Attribute':
        if chain:
            chain += ' AND' + ' ' + str(tree[1]) + ' ' + '==' + ' '
        else:
            chain = 'IF' + ' ' + str(tree[1]) + ' ' + '==' + ' '
        rules = Get_Rules(tree[2:], rules, chain, previous_value, class_name)
    if tree[0][0] == 'Value':
        for i in range(len(tree)):
            if previous_value and previous_value == chain[-len(previous_value):]:
                length = len(previous_value)
                chain = chain[:-length] + ' '
            chain += str(tree[i][1])
            previous_value = str(tree[i][1])
            rules = Get_Rules(tree[i][2], rules, chain, previous_value, class_name)
    if tree[0] == 'Leaf':
        chain += ' THEN' + ' ' + class_name + ' ' + '=' + ' ' + str(tree[1])
        chain = re.sub(' +', ' ', chain)
        rules.append(chain)
    return rules

def format_num(list):
    for i in range(len(list)):
        list[i] = round(list[i] * 100, 2)
    return list

def bin_vals(list):
    bins = [[],[],[],[],[],[],[],[],[],[]]
    for val in list:
        if val < 11:
            bins[0].append(val)
        elif val < 21:
            bins[1].append(val)
        elif val < 31:
            bins[2].append(val)     
        elif val < 41:
            bins[3].append(val)
        elif val < 51:
            bins[4].append(val)
        elif val < 61:
            bins[5].append(val)
        elif val < 71:
            bins[6].append(val)
        elif val < 81:
            bins[7].append(val)
        elif val < 91:
            bins[8].append(val)
        else:
            bins[9].append(val)
    return bins

def bin_loudness(list):
    bins = [[],[],[],[],[]]
    for val in list:
        if val > -5.6322:
            bins[0].append(val)
        elif val > -10.5644:
            bins[1].append(val)
        elif val > -15.4966:
            bins[2].append(val)
        elif val > -20.4288:
            bins[3].append(val)
        else:
            bins[4].append(val)
    return bins

def bin_tempo(list):
    bins = [[],[],[],[],[]]
    for val in list:
        if val < 70.809:
            bins[0].append(val)
        elif val < 106.141:
            bins[1].append(val)
        elif val < 141.473:
            bins[2].append(val)
        elif val < 176.805:
            bins[3].append(val)
        else:
            bins[4].append(val)
    return bins


def get_bin_count(bins):
    for bin in bins:
        bin[0] = len(bin[0])
    return bins

def unique_index(vals, i): 
    unique = []
    for row in vals:
        if row[i] not in unique:
            unique.append(row[i])
    return unique

def tdidt_forest(current_instances, available_attributes, attribute_domains, header, F): 
    # basic approach (uses recursion!!):
    subset_attributes = compute_random_subset(available_attributes, F)
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, subset_attributes, attribute_domains, header)
    #print('splitting on', split_attribute)
    available_attributes.remove(split_attribute) # cannot split on same attr twice in a branch
    # python is pass by object reference!!
    tree = ['Attribute', split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)
    #print('partitions:', partitions)

    prev = []
    prev_instances = 0
    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        #print('working with partition for', attribute_value)
        value_subtree = ['Value', attribute_value]
        subtree = []
        # TODO: appending leaf nodes and subtrees appropriately to value_subtree
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and check_all_same_class(partition): # all same class checks if all the other values equal the first one
            subtree = ['Leaf', partition[0][-1], len(partition), len(current_instances)]
            value_subtree.append(subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            subtree = ['Leaf', majority_vote(partition), len(partition), len(current_instances)]
            value_subtree.append(subtree)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            return ['Leaf', majority_vote(prev), len(prev), prev_instances]
        else:
            subtree = tdidt_forest(partition, available_attributes.copy(), attribute_domains.copy(), header.copy(), F)
            value_subtree.append(subtree)
            # need to append subtree to value_subtree and appropriately append value_subtree to tree
        tree.append(value_subtree)
        prev = partition
        prev_instances = len(current_instances)

    return tree

def compute_random_subset(values, num_values): 
    shuffled = values[:] # shallow copy 
    random.shuffle(shuffled)
    return sorted(shuffled[:num_values])

def compute_bootstrapped_sample(table):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])
    return sample

def convert_tempo(tempo):
    res = []
    for val in tempo:
        res.append(get_tempo(val))
    return res

def get_tempo(val):
    if val < 70.809:
        curr = 1
    elif val < 106.141:
        curr = 2
    elif val < 141.473:
        curr = 3
    elif val < 176.805:
        curr = 4
    else:
        curr = 5
    return curr

def convert_loudness(loudness):
    res = []
    for val in loudness:
        res.append(get_loudness(val))
    return res

def get_loudness(val):
    if val > -5.6322:
        curr = 1
    elif val > -10.5644:
        curr = 2
    elif val > -15.4966:
        curr = 3
    elif val > -20.4288:
        curr = 4
    else:
        curr = 5
    return curr

def compute_average(list):
    return sum(list) / len(list)
