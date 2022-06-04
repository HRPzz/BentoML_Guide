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




