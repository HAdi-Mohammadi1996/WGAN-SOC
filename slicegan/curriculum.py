"""
Staged conditioning curriculum for Conditional SliceGAN.

The lambda_reg weight on the regression loss is annealed over three stages:

  Stage 1  (epochs 1–stage1_end):       lambda_reg = 0.0   (unconditional baseline)
  Stage 2  (stage1_end+1–stage2_end):   lambda_reg linearly from lambda_start to lambda_end
  Stage 3  (stage2_end+1–∞):            lambda_reg = lambda_end
"""

from dataclasses import dataclass


@dataclass
class LambdaSchedule:
    """Bundle of curriculum hyper-parameters for easy passing and testing."""
    stage1_end: int = 50
    stage2_end: int = 150
    lambda_start: float = 0.1
    lambda_end: float = 1.0

    def get(self, epoch: int) -> float:
        """Return lambda_reg for the given epoch (1-indexed)."""
        return get_lambda_reg(
            epoch,
            stage1_end=self.stage1_end,
            stage2_end=self.stage2_end,
            lambda_start=self.lambda_start,
            lambda_end=self.lambda_end,
        )


def get_lambda_reg(
    epoch: int,
    stage1_end: int = 50,
    stage2_end: int = 150,
    lambda_start: float = 0.1,
    lambda_end: float = 1.0,
) -> float:
    """Return the regression loss weight for a given training epoch.

    Parameters
    ----------
    epoch : int
        Current epoch number, 1-indexed.
    stage1_end : int
        Last epoch of Stage 1 (lambda_reg == 0).
    stage2_end : int
        Last epoch of Stage 2 (end of linear ramp).
    lambda_start : float
        lambda_reg at the beginning of Stage 2.
    lambda_end : float
        lambda_reg at the end of Stage 2 and throughout Stage 3.

    Returns
    -------
    float
        lambda_reg value for this epoch.
    """
    if epoch <= stage1_end:
        return 0.0
    if epoch > stage2_end:
        return lambda_end
    t = (epoch - stage1_end) / (stage2_end - stage1_end)
    return lambda_start + t * (lambda_end - lambda_start)
