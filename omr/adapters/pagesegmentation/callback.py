from pagesegmentation.lib.trainer import TrainProgressCallback
from omr.steps.algorithm import TrainerCallback


class PCTrainerCallback(TrainProgressCallback):
    def __init__(self, callback: TrainerCallback):
        super().__init__()
        self.callback = callback

    def init(self, total_iters, early_stopping_iters):
        self.callback.init(total_iters, early_stopping_iters)

    def next_iteration(self, iter: int, loss: float, acc: float, fgpa: float):
        self.callback.next_iteration(iter, loss, acc)

    def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
        self.callback.next_best_model(best_iter, best_acc, best_iters)

    def early_stopping(self):
        self.callback.early_stopping()
