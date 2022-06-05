import bentoml

from sklearn import svm
from sklearn import datasets

# Load training data set
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X, y)


# Save model to the BentoML local model store

bentoml.sklearn.save_model("iris_clf", clf)


# INFO  [cli] Using default model signature `{"predict": {"batchable": False}}` for sklearn model
# INFO  [cli] Successfully saved Model(tag="iris_clf:2uo5fkgxj27exuqj", path="~/bentoml/models/iris_clf/2uo5fkgxj27exuqj/")

