from segmentation.network import TrainCallback
from segmentation.stats import EpochStats

from omr.steps.algorithm import TrainerCallback
from loguru import logger


class SegmentationProcessTrainCallback(TrainCallback):
    """
    A callback for training segmentation models, inheriting from the base TrainCallback.
    It can be used to implement custom behavior during the training process.
    """
    def __init__(self, callback: TrainerCallback, epochs=0, metric_watcher_index=0):
        super().__init__()

        self.max_epochs = epochs
        self.current_epoch = 0
        self.callback = callback
        self.callback.init(self.max_epochs, self.max_epochs)
        self.stats: EpochStats = None
        self.metric_watcher_index = metric_watcher_index
        self.best_loss = -1
        self.best_loss_iter = -1
        self.best_acc = -1
        # Initialize any additional attributes if needed

    def on_train_epoch_end(self, epoch, acc, loss):
        # Custom logic for when a training epoch ends
        pass

    def on_train_epoch_start(self):
        # Custom logic for when a training epoch starts
        return 0  # If -1 returned, then epoch is skipped

    def on_val_epoch_end(self, epoch, acc, loss):
        acc: EpochStats = acc
        if self.stats is None or acc.stats[self.metric_watcher_index].value() > self.stats.stats[
            self.metric_watcher_index].value():
            accuracy_before = 0 if self.stats is None else self.stats.stats[self.metric_watcher_index].value()
            self.stats = acc
            self.best_loss = loss
            self.best_loss_iter = epoch
            self.best_acc= acc.stats[self.metric_watcher_index].value().item()
        # Custom logic for when a validation epoch ends
        rounded = round(acc.stats[self.metric_watcher_index].value().item(), 2)
        acc = acc.stats[self.metric_watcher_index].value().item()
        self.current_epoch = epoch
        self.callback.next_iteration(epoch, loss, acc / 100)
        self.callback.next_best_model(epoch,  acc / 100, self.best_loss_iter)
        logger.info(f"Epoch {epoch}: loss={loss}, acc_rounded = {rounded}, acc={acc}, max_epochs={self.max_epochs}, ")

    def on_batch_end(self, batch, loss, acc, logs=None):
        pass
