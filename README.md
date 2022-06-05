# BentoML 이란?

- 머신러닝과 딥러닝 모델을 만들고 `model`을 어떻게 `Serving`할 것인가는 큰 고민거리입니다.

- 단순히 `Model`의 결과를 `DB`에 저장하여 사용할 것인지?

- 아니면 `API`를 사용할 것인지?

- ML 개발자들은 `Model`을 만들면 해당 모델을 쉽게 배포, 테스트 등을 할수 있어야 합니다.

- `BentoML`은 이러한 고민들을 해결 하기 위해 만들어진 라이브러리 입니다.


## BentoML 튜토리얼 설정

- `BentoML` 튜토리얼 하기 위해서는 두가지 방법이 있습니다.

  - 하나는 `Colab`을 이용하는 방법

  - 나머지는 `Local`에서 이요하는 방법


- 저는 `Local`에서 이용되는 방법을 사용할것이고 해당 소스 코드는 [링크](https://github.com/bentoml/gallery/) 에서 확인이 가능합니다.

- 또한 로컬에서 작업을 하기 위해서는 `python 3.7`이상이 필요합니다.

- 그럼 `BentoML`에서 필요한 종속 라이브러리를 설치 해보도록 해보겠습니다.

- 필자는 `pyenv`를 통해 가상환경을 `3.9.10`  설정하고 진행해보았습니다.

```bash

pip install --pre bentoml

pip install scikit-learn pandas

```

### BentoML 모델 저장

- `BentoML`을 시작하려면 `BentoML`에서 제공하는 모델 저장소에 저장을 해야합니다.

- 간단한 예제 코드를 통해 한번 보도록 해보죠

```python
import bentoml # bentoml API load

from sklearn import svm # scikit learn에서 svm 가져오기 
from sklearn import datasets # iris 데이터를 가져오기 위해서 로드 

# 데이터셋 로드 
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 모델에 학습 시키기 
clf = svm.SVC(gamma='scale')
clf.fit(X, y)


# Bentoml 저장소에 저장시키기

bentoml.sklearn.save_model("iris_clf", clf)


# 실행 시키면 아래 처럼 결과물이 나오게 될겁니다.
# INFO  [cli] Using default model signature `{"predict": {"batchable": False}}` for sklearn model
# INFO  [cli] Successfully saved Model(tag="iris_clf:2uo5fkgxj27exuqj", path="~/bentoml/models/iris_clf/2uo5fkgxj27exuqj/")

```
- 해당 코드를 실행 시키면 결과물이 아래처럼 나오게 될것입니다.

![1](./imgs/1.png) 

- 이 뜻은 `BentoML`에서 `Local`에 모델을 저장 했다는것입니다.

- 해당 디렉토리로 가면 폴더가 있는것을 볼수있습니다. 
![2](./imgs/2.png) 

- 우리는 저장한 모델을 불러올수 있게 되는데 아래 코드를 통해 불러올수 있습니다.

```python
import bentoml # bentoml API load

# 해당 결과물에서 나온 코드값으로 모델을 불러올수  있으며 
model = bentoml.sklearn.load_model("iris_clf:7dtkvtxe2oru2aav")

# 또는 가장 최근에 저장한 모델을 불러올수 있습니다.
model = bentoml.sklearn.load_model("iris_clf:latest")

```

- 해당 예시는 `scikit-learn`에서 저장하는 방법입니다.

- `FrameWork`별 차이가 있긴 하지만 `pytorch`의 경우에는 `bentoml.pytorch.save_model`, `tensorflow`의 경우는 `bentoml.tensorflow.save_model`과 같이 나타나게 됩니다. 자세한 사항은 [링크]("https://docs.bentoml.org/en/latest/frameworks/index.html") 



























