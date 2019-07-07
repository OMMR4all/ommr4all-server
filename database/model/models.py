import os
from .model import Model
from typing import List, Optional


class Models:
    def __init__(self, models_path):
        self.models_path = models_path
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def list_models(self) -> List[Model]:
        return [Model(os.path.join(self.models_path, d)) for d in reversed(sorted(os.listdir(self.models_path)))]

    def newest_model(self) -> Optional[Model]:
        models = self.list_models()
        if len(models) == 0:
            return None

        return models[0]
