# Service and API

## 서비스 생성

- `BentoML`서비스는 `Runner`와 `API`로 구성됩니다.

```python
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result

```

- 만약 저번 과정을 하지 않았다면 위 코드를 작동시켜 모델을 로컬에 저장해보도록 하십니다.

```python
# Create the iris_classifier_service with the ScikitLearn runner
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])
```
- 서비스는 `bentoml.Service` 호출을 서비스 이름과 `Runner` 통해 초기화됩니다.

```python
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result
```

- `svc`는 API를 지정하는 데코레이터 입니다.


## Runner

- 러너(`Runner`)는 처리량(`throughput`) 과 리소스 사용률(`resource utilization`)을 극대화할 수 있는 서비스 로직의 단위를 나타냅니다.

  - `runner = bentoml.sklearn.get("iris_clf:latest").to_runner()` 

- 모델에서 작성된 Runner는 대상 ML 프레임워크에 고유한 최적의 Runner 구성을 자동으로 선택합니다.

## Service API

- `Inference APIs` 서비스 기능에 원격으로 액세스할 수 있는 방법을 정의합니다.

- API는 입력/출력 사양과 콜백 함수로 구성됩니다.

```python
# "svc"를 통해 API 생성
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())  # input,output 정의
def predict(input_array: np.ndarray) -> np.ndarray:
    # Define business logic
    # Define pre-processing logic
    result = runner.run(input_array)  #  model inference 호출
    # Define post-processing logic
    return result
```

- API 함수 안에서는 DB에 대한 접근, 전처리와 같은 로직을 정의하기 좋습니다.

- 
