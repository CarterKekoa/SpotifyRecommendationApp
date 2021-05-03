# we are going to use Flask, a micro web framework
import os
import pickle
from flask import Flask, jsonify, request 

# make a Flask app
app = Flask(__name__)

# we need to add two routes (functions that handle requests)
# one for the homepage
@app.route("/", methods=["GET"])
def index():
    # return content and a status code
    return "<h1>Welcome to my App</h1>", 200

# one for the /predict 
@app.route("/predict", methods=["GET"])
def predict():
    # goal is to extract the 4 attribute values from query string
    # use the request.args dictionary
    level = request.args.get("level", "") # check for the key, and the default
    lang = request.args.get("lang", "")
    tweets = request.args.get("tweets", "")
    phd = request.args.get("phd", "")
    print("level:", level, lang, tweets, phd)
    # task: extract the remaining 3 args

    # get a prediction for this unseen instance via the tree
    # return the prediction as a JSON response

    # TODO: fix the hardcoding
    prediction = predict_interviews_well([level, lang, tweets, phd]) # if anything goes wrong, this function will return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        # failure!!
        return "Error making prediction", 400


def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        print()
        attribute_index = header.index(tree[1])
        print("tree[1]: ", tree[1])
        print("attribute_index: ", attribute_index)
        instance_value = instance[attribute_index]
        print("instance_value: ", instance_value)
        # now I need to find which "edge/branch" to follow recursively
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match! recurse!
                return tdidt_predict(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label


# this is the predict for pa6 my decision tree classifier
def predict_interviews_well(instance):
    # 1. we need a tree (and its header) to make a prediction
    #   we need to save a trained model (fit()) to a file
    #   so we can load that file into memory in another python
    #   process as a python object (predict())
    # import pickle and load the header and interview tree
    #   as Python objects we can use for step 2
    infile = open("APIServiceFun/tree.p", "rb") # r for read, b for binary
    header, tree = pickle.load(infile)
    infile.close()
    print("header: ", header)
    print("tree: ", tree)

    # 2. use the tree to make a prediction
    try:
        # test url = "https://interview-flask-app.herokuapp.com/predict?level=Junior&language=Java&tweets=yes&phd=no"
        return tdidt_predict(header, tree, instance) # recursive function
    except:
        return None

if __name__ == "__main__":
    # deployment notes
    # two main categories of how to deploy
    # host your own server OR use a cloud provider
    # there are lots of options for cloud providers... AWS, Heroku, Azure, DigtalOcean, Vercel, ...
    # we are going to use Heroku (Backend as a Service BaaS)
    # there are lots of ways to deploy a flask app to Heroku
    # 1. deploy the app directly as a web app running on the ubuntu "stack" 
    # (e.g. Procfile and requirements.txt)
    # 2. deploy the app as a Docker container running on the container "stack"
    # (e.g. Dockerfile)
    # 2.A. build the docker image locally and push it to a container registry (e.g. Heroku's)
    # **2.B.** define a heroku.yml and push our source code to Heroku's git repo
    #  and Heroku will build the docker image for us
    # 2.C. define a main.yml and push our source code to Github, where a Github Action
    # builds the image and pushes it to the Heroku registry

    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port) # TODO: set debug to False for production
    # by default, Flask runs on port 5000