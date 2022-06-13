import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])


@svc.api(
    input=NumpyNdarray(),
    output=NumpyNdarray(),
    route="/v2/models/my_model/versions/v0/infer",
)
def predict(input_array: np.ndarray) -> np.ndarray:
    return input_array
