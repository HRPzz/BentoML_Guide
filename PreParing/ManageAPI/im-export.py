import bentoml
bentoml.models.export_model('iris_clf:latest', '/path/to/folder/my_model.bentomodel')


bentoml.models.import_model('/path/to/folder/my_model.bentomodel')
