import bentoml
bento_model: bentoml.Model = bentoml.models.get("iris_clf:latest")

print(bento_model.path)
print(bento_model.info.metadata)
print(bento_model.info.labels)
