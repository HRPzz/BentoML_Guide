import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

bentoml.pytorch.save_model(
     "demo_mnist",  # model name in the local model store
     trained_model,  # model instance being saved

     signatures={   # model signatures for runner inference

         "classify": {

             "batchable": False,

         }

     }

 )

 runner = bentoml.pytorch.get("demo_mnist:latest").to_runner()
 runner.init_local()

 runner.classify.run( MODEL_INPUT )
