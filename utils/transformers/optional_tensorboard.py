"""Optional TensorBoard logging for evaluation entry points."""

from typing import Any, Protocol


class SummaryWriterLike(Protocol):
    """Minimal writer surface used by the model evaluators."""

    def add_histogram(self, *args: Any, **kwargs: Any) -> None:
        """Record a histogram when TensorBoard is installed."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        """Record a scalar when TensorBoard is installed."""

    def close(self) -> None:
        """Flush and close the writer."""


class _NoOpSummaryWriter:
    """Preserve evaluator execution when optional TensorBoard is absent."""

    def add_histogram(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def close(self) -> None:
        pass


def create_summary_writer(*, log_dir: str) -> SummaryWriterLike:
    """Create TensorBoard's writer, or a no-op equivalent if unavailable."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return _NoOpSummaryWriter()
    return SummaryWriter(log_dir=log_dir)
