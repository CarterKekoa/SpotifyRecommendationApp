from mysklearn.myclassifiers import MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

def test_random_forest_classifier_fit():
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

    X_train = [x[:-1] for x in interview_table]
    y_train = [x[-1] for x in interview_table]

    my_tree = MyRandomForestClassifier()
    my_tree.fit(X_train, y_train)
    
def test_random_forest_classifier_predict():
    pass