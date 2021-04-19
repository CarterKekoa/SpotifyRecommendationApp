##############################################
# Programmer: Carter Mooring
# Class: CPCS 322-02, Spring 2021
# Programming Assignment #6
# April 14th, 2021
# 
# Description: 
##############################################

import mysklearn.myutils as myutils

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        # New lists used for when the there is a nested list
        new_list_x = []
        new_list_y = []
        # These will store true if there is a nested list passed through
        use_new_x = any(isinstance(i, list) for i in X_train)
        use_new_y = any(isinstance(i, list) for i in y_train)
        
        # if there exists a nested list
        if (use_new_x and use_new_y):
            for j in X_train:
                new_list_x.append(j[0])
            for k in y_train:
                new_list_y.append(k[0])
            m, b = myutils.compute_slope_intercept(new_list_x, new_list_y)
        elif (use_new_x):
            for j in X_train:
                new_list_x.append(j[0])
            m, b = myutils.compute_slope_intercept(new_list_x, y_train)
        elif (use_new_y):
            for j in y_train:
                new_list_y.append(j[0])
            m, b = myutils.compute_slope_intercept(X_train, new_list_y)
        else:
            # else there is no nested list
            m, b = myutils.compute_slope_intercept(X_train, y_train)

        # store the slope and intercept
        self.slope = m
        self.intercept = b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        # for each value in the X_test list
        for val in X_test:
            # y = mx + b equation
            prediction = (self.slope * val[0]) + self.intercept
            y_predicted.append(prediction)
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        # scale the list and test list
        scaled_X_train, scaled_X_test = myutils.scale(self.X_train, X_test)
        
        all_distances = []
        all_indices = []
        # for each value in the scaled test list
        for val in scaled_X_test:
            # the prep gives us the distance and indice list 
            distance_list, indice_list = myutils.kneighbors_prep(scaled_X_train, val, self.n_neighbors)
            all_distances.append(distance_list)
            all_indices.append(indice_list)
        return all_distances, all_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        distance, indices = self.kneighbors(X_test)
        labels = []

        # for row in indices list
        for row in indices:
            temp = []
            for i in row:
                val = self.y_train[i]
                temp.append(val)
            labels.append(temp)
 
        # for each row in labels
        for row in labels:
            predictions.append(myutils.get_label(row))
 
        return predictions

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.priors = myutils.priors(y_train)
        self.posteriors = myutils.posteriors(X_train, y_train, self.priors)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        prediction = []
        prob = 1
        unique_y_vals = myutils.get_unique(self.y_train)

        # for each row in X_test
        for row in X_test:
            probability = []
            # for each unique value in the y_train
            for unique in unique_y_vals:
                prob *= self.priors[unique]
                # go through each value in the row and calculate its probabilty
                for i, val in enumerate(row):
                    # if the value is noted in the table
                    if val in self.posteriors[unique][i]:
                        prob *= self.posteriors[unique][i][val]
                    else:
                        prob = 0
                probability.append(prob)
                prob = 1
            # compare to see which label to pick
            prediction.append(unique_y_vals[myutils.get_prediction_index(probability)])
        #print(prediction)
        return prediction

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        att_vals = {}
        headers = ['att' + str(i) for i in range(len(self.X_train[0]))]
        
        # for each header, grab the index also
        for i, header in enumerate(headers):
            att_vals[header] = [] # for each header, make an empty list
            # for each value in the x_train
            for val in self.X_train:
                # if the value is not yet in the attribute dict, then add it
                if val[i] not in att_vals[header]:
                    att_vals[header].append(val[i])
        # for each key and value in the dictionary, sort the values
        for key, value in att_vals.items():
            att_vals[key] = sorted(value)

        training_set = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]
        attributes = headers.copy()
        self.tree = myutils.tdidt(training_set, attributes, att_vals, headers) # call the tdidt function
        
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = [] # initialize empty list

        # for each instance in x_test
        for instance in X_test:
            # get the predicted target y valeus with helper function
            y_prediction = myutils.y_pred(instance, self.tree)
            y_predicted.append(y_prediction) # append to list
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rules = myutils.Rules(tree=self.tree, rules=[], chain='' , previous_value='', class_name=class_name)
        for rule in rules:
            print(rule)

