"""
Training callbacks for monitoring and controlling the training process.
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import logging
import time
import json

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """
    Abstract base class for training callbacks.

    Callbacks can hook into various stages of training:
    - on_train_start: Called when training begins
    - on_train_end: Called when training ends
    - on_epoch_start: Called at the start of each epoch
    - on_epoch_end: Called at the end of each epoch
    - on_batch_start: Called before each batch
    - on_batch_end: Called after each batch
    """

    def on_train_start(self, trainer: "Trainer", **kwargs) -> None:
        """Called when training starts."""
        pass

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        """Called when training ends."""
        pass

    def on_epoch_start(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: "Trainer", batch_idx: int, **kwargs) -> None:
        """Called before each batch."""
        pass

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, loss: float, **kwargs
    ) -> None:
        """Called after each batch."""
        pass


class EarlyStopping(TrainingCallback):
    """
    Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric to monitor (default: "val_loss")
        patience: Number of epochs with no improvement to wait
        min_delta: Minimum change to qualify as improvement
        mode: "min" or "max" (whether lower or higher is better)

    Examples:
        >>> early_stop = EarlyStopping(monitor="val_loss", patience=5)
        >>> trainer.fit(callbacks=[early_stop])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best_value: Optional[float] = None
        self._counter = 0
        self._stopped_epoch = 0

    def on_train_start(self, trainer: "Trainer", **kwargs) -> None:
        self._best_value = None
        self._counter = 0

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs
    ) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            logger.warning(f"EarlyStopping: metric '{self.monitor}' not found")
            return

        if self._best_value is None:
            self._best_value = current
            return

        if self._is_improvement(current):
            self._best_value = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                trainer._stop_training = True
                self._stopped_epoch = epoch
                logger.info(f"EarlyStopping: stopped at epoch {epoch}")

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self._best_value - self.min_delta
        return current > self._best_value + self.min_delta

    def __repr__(self) -> str:
        return f"EarlyStopping(monitor='{self.monitor}', patience={self.patience})"


class ModelCheckpoint(TrainingCallback):
    """
    Save model checkpoints during training.

    Args:
        filepath: Path template for saving (can include {epoch}, {val_loss}, etc.)
        monitor: Metric to monitor for best model
        save_best_only: Only save when monitored metric improves
        mode: "min" or "max"

    Examples:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath="checkpoints/model_{epoch:02d}.pt",
        ...     monitor="val_loss",
        ...     save_best_only=True,
        ... )
    """

    def __init__(
        self,
        filepath: str = "checkpoint.pt",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode

        self._best_value: Optional[float] = None
        self._best_path: Optional[str] = None

    def on_train_start(self, trainer: "Trainer", **kwargs) -> None:
        self._best_value = None
        # Create directory if needed
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs
    ) -> None:
        current = metrics.get(self.monitor)

        # Format filepath with metrics
        filepath = self.filepath.format(epoch=epoch, **metrics)

        if self.save_best_only:
            if current is None:
                logger.warning(f"ModelCheckpoint: metric '{self.monitor}' not found")
                return

            if self._best_value is None or self._is_improvement(current):
                self._best_value = current
                self._best_path = filepath
                self._save_checkpoint(trainer, filepath, epoch, metrics)
                logger.info(f"ModelCheckpoint: saved best model to {filepath}")
        else:
            self._save_checkpoint(trainer, filepath, epoch, metrics)
            logger.debug(f"ModelCheckpoint: saved to {filepath}")

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self._best_value
        return current > self._best_value

    def _save_checkpoint(
        self,
        trainer: "Trainer",
        filepath: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save checkpoint to file."""
        try:
            import torch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "metrics": metrics,
                "config": trainer.config,
            }
            torch.save(checkpoint, filepath)
        except ImportError:
            # Fallback: save metrics only
            with open(filepath + ".json", "w") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    f,
                    indent=2,
                )

    @property
    def best_path(self) -> Optional[str]:
        """Get path to best saved model."""
        return self._best_path

    def __repr__(self) -> str:
        return f"ModelCheckpoint(filepath='{self.filepath}', monitor='{self.monitor}')"


class ProgressLogger(TrainingCallback):
    """
    Log training progress to console.

    Args:
        log_frequency: How often to log (every N batches)

    Examples:
        >>> logger = ProgressLogger(log_frequency=10)
    """

    def __init__(self, log_frequency: int = 10):
        self.log_frequency = log_frequency
        self._epoch_start_time: float = 0
        self._batch_losses: list = []

    def on_epoch_start(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        self._epoch_start_time = time.time()
        self._batch_losses = []
        logger.info(f"Epoch {epoch + 1}/{trainer.epochs}")

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, loss: float, **kwargs
    ) -> None:
        self._batch_losses.append(loss)

        if (batch_idx + 1) % self.log_frequency == 0:
            avg_loss = np.mean(self._batch_losses[-self.log_frequency :])
            logger.info(f"  Batch {batch_idx + 1}: loss={avg_loss:.4f}")

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs
    ) -> None:
        elapsed = time.time() - self._epoch_start_time
        avg_loss = np.mean(self._batch_losses) if self._batch_losses else 0

        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(
            f"Epoch {epoch + 1} completed in {elapsed:.1f}s - loss={avg_loss:.4f}, {metrics_str}"
        )

    def __repr__(self) -> str:
        return f"ProgressLogger(log_frequency={self.log_frequency})"


class LRScheduler(TrainingCallback):
    """
    Learning rate scheduler callback.

    Args:
        scheduler_type: Type of scheduler ("step", "cosine", "plateau")
        **kwargs: Arguments for the scheduler

    Examples:
        >>> # Step decay every 10 epochs
        >>> scheduler = LRScheduler("step", step_size=10, gamma=0.1)
        >>>
        >>> # Cosine annealing
        >>> scheduler = LRScheduler("cosine", T_max=100)
        >>>
        >>> # Reduce on plateau
        >>> scheduler = LRScheduler("plateau", patience=5, factor=0.5)
    """

    def __init__(self, scheduler_type: str = "step", **kwargs):
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = kwargs
        self._scheduler = None

    def on_train_start(self, trainer: "Trainer", **kwargs) -> None:
        try:
            import torch.optim.lr_scheduler as lr_scheduler

            if self.scheduler_type == "step":
                self._scheduler = lr_scheduler.StepLR(
                    trainer.optimizer, **self.scheduler_kwargs
                )
            elif self.scheduler_type == "cosine":
                self._scheduler = lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer, **self.scheduler_kwargs
                )
            elif self.scheduler_type == "plateau":
                self._scheduler = lr_scheduler.ReduceLROnPlateau(
                    trainer.optimizer, **self.scheduler_kwargs
                )
            else:
                logger.warning(f"Unknown scheduler type: {self.scheduler_type}")
        except ImportError:
            logger.warning("PyTorch not available for LR scheduling")

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs
    ) -> None:
        if self._scheduler is None:
            return

        if self.scheduler_type == "plateau":
            val_loss = metrics.get("val_loss", metrics.get("loss", 0))
            self._scheduler.step(val_loss)
        else:
            self._scheduler.step()

        # Log current learning rate
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        logger.debug(f"Learning rate: {current_lr:.6f}")

    def __repr__(self) -> str:
        return f"LRScheduler(type='{self.scheduler_type}')"


class TensorBoardLogger(TrainingCallback):
    """
    Log training metrics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard logs

    Examples:
        >>> tb_logger = TensorBoardLogger(log_dir="runs/experiment1")
    """

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self._writer = None

    def on_train_start(self, trainer: "Trainer", **kwargs) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir)
            logger.info(f"TensorBoard logging to {self.log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available")

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, loss: float, **kwargs
    ) -> None:
        if self._writer is None:
            return

        global_step = trainer.current_epoch * trainer.batches_per_epoch + batch_idx
        self._writer.add_scalar("train/batch_loss", loss, global_step)

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs
    ) -> None:
        if self._writer is None:
            return

        for name, value in metrics.items():
            self._writer.add_scalar(f"metrics/{name}", value, epoch)

        # Log learning rate
        lr = trainer.optimizer.param_groups[0]["lr"]
        self._writer.add_scalar("train/learning_rate", lr, epoch)

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        if self._writer is not None:
            self._writer.close()

    def __repr__(self) -> str:
        return f"TensorBoardLogger(log_dir='{self.log_dir}')"
