import bentoml

model = bentoml.sklearn.load_model("iris_clf:7dtkvtxe2oru2aav")

# Alternatively, use `latest` to find the newest version
model = bentoml.sklearn.load_model("iris_clf:latest")
