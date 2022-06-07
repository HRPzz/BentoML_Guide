import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC(gamma='scale')
clf.fit(X, y)

bentoml.sklearn.save_model("iris_clf", clf)


bentoml.sklearn.save_model(
    "demo_mnist",  # model name in the local model store
    clf,  # model instance being saved
    signatures={   # model signatures for runner inference
        "predict": {"batchable": False,}
    }
    )

runner = bentoml.sklearn.get("demo_mnist:latest").to_runner()
runner.init_local()
runner.predict.run([[5.9, 3., 5.1, 1.8]])

