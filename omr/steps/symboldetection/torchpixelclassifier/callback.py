from omr.steps.algorithm import TrainerCallback
from segmentation.network import TrainProgressCallback

class PCTorchTrainerCallback(TrainProgressCallback):
    def __init__(self, callback: TrainerCallback):
        super().__init__()
        self.callback = callback

    def init(self, total_iters, early_stopping_iters):
        self.callback.init(total_iters, early_stopping_iters)

    def update_loss(self, batch: int, loss: float, acc: float):
        self.callback.next_iteration(batch, loss, acc)

    def next_best(self, epoch, acc, n_best):
        self.callback.next_best_model(epoch, acc, n_best)
