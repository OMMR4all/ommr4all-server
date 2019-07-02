from mashumaro import DataClassJSONMixin
from dataclasses import dataclass


@dataclass()
class TaskTrainerParams(DataClassJSONMixin):
    nTrain: float = 0.8
    includeAllTrainingData: bool = False
