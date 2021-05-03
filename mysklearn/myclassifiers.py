import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation

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

        self.priors = myutils.get_priors(y_train)
        self.posteriors = myutils.get_posteriors(X_train, y_train, self.priors)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for test in X_test:
            probs = myutils.compute_probs(test, self.priors, self.posteriors)
            print("probs:", probs)
            prediction = myutils.predict_from(probs)
            y_predicted.append(prediction)
        return y_predicted

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

        attribute_domains, header = myutils.get_att_domain(X_train)

        training_set = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]
        available_attributes = header.copy()
        self.tree = myutils.tdidt(training_set, available_attributes, attribute_domains, header)


        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            y_pred = myutils.classifySample(instance, self.tree)
            y_predicted.append(y_pred)
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rules = myutils.Get_Rules(tree=self.tree, rules=[], chain='' , previous_value='', class_name=class_name)
        for rule in rules:
            print(rule)

class MyRandomForestClassifier: 
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
    def __init__(self, F=2, N=4, M=3):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.trees = []
        self.N = N
        self.M = M
        self.F = F

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
        header = ['att' + str(i) for i in range(len(X_train[0]))]
        attribute_domains = {}
        for i, val in enumerate(header):
            attribute_domains[val] = myutils.unique_index(X_train, i)

        self.X_train = X_train
        self.y_train = y_train
        sample_X_train, sample_x_test, sample_y_train, sample_y_test = myevaluation.train_test_split(X_train, y_train, test_size=0.33, shuffle=True)
        train = [sample_X_train[i] + [sample_y_train[i]] for i in range(len(sample_X_train))]
        
        for _ in range(self.N):
            available_attributes = header.copy()
            self.trees.append(myutils.tdidt_forest(myutils.compute_bootstrapped_sample(train), available_attributes, attribute_domains, header, self.F))
        
        accuracies = []
        for tree in self.trees:
            header = ['att' + str(i) for i in range(len(sample_x_test[0]))]
            prediction = []
            for row in sample_x_test:
                prediction.append(myutils.tdidt_predict(header, tree, row))
            accuracy = 0
            for i in range(len(prediction)):
                if prediction[i] == sample_y_test[i]:
                    accuracy += 1
            accuracy /= len(sample_y_test)
            accuracies.append([accuracy])
        # find m most accurate
        m_trees = []
        for i in range(len(accuracies)):
            accuracies[i].append(i)
        accuracies = sorted(accuracies)
        for i in range(self.M):
            m_trees.append(self.trees[accuracies[-(i+1)][1]])
        self.trees = m_trees
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = ['att' + str(i) for i in range(len(X_test[0]))]
        res = []
        for row in X_test:
            curr = []
            for tree in self.trees:
                curr.append(myutils.tdidt_predict(header, tree, row))
            res.append(curr)
        return myutils.get_majority_votes(res)

