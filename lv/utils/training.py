"""Utilities for training models."""


class EarlyStopping:
    """Observes a numerical value and determines when it has not improved."""

    def __init__(self, patience: int = 4, decreasing: bool = True):
        """Initialize the early stopping tracker.

        Args:
            patience (int, optional): Allow tracked value to not improve over
                its best value this many times. Defaults to 4.
            decreasing (bool, optional): If True, the tracked value "improves"
                if it decreases. If False, it "improves" if it increases.
                Defaults to True.

        """
        self.patience = patience
        self.decreasing = decreasing
        self.best = float('inf') if decreasing else float('-inf')
        self.num_bad = 0

    def __call__(self, value: float) -> bool:
        """Considers the new tracked value and decides whether to stop.

        Args:
            value (float): The new tracked value.

        Returns:
            bool: True if patience has been exceeded.

        """
        improved = self.decreasing and value < self.best
        improved |= not self.decreasing and value > self.best
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1

        return self.num_bad > self.patience
