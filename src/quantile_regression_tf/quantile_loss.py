import numpy as np
from typing import List, Union
import tensorflow as tf


def quantile_loss(
        quantiles: tf.Tensor, ygt: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Returns the total quantile loss.

    The length of ygt and the length of the first dimension of y must match
    i.e., the number of examples must be compatible. The last dimension of y
    must match the number of quantiles, i.e., there must be a prediction for
    each quantile value.

    Arguments:
        quantiles: The values of the quantiles to be fit.
        ygt: Ground truth values.
        y: Quantile predictions for each ground truth value.

    Returns:
        Loss
    """

    assert y.shape[0] == ygt.shape[0], (
            "QuantileLoss: Batch sizes for predictions and ground truth values "
            f"must be equal. ({y.shape[0]} vs. {ygt.shape[0]})")

    assert y.shape[-1] == quantiles.shape[0], (
            "QuantileLoss: Predictions must match the number of quantiles. "
            f"({y.shape[-1]} vs. {quantiles.shape[0]})")

    error = ygt - y
    return tf.reduce_mean(tf.maximum(
        quantiles * error, error * (quantiles - 1)), axis=-1)


class QuantileLoss:
    """Loss function for a set of quantiles.

    Arguments:
        quantiles: Either an explicit enumeration of the of quantiles to be fit
            as a List, np.ndarray, or tf.Tensor or the number of quantiles to be
            fit. In the latter case, the quantiles are spaced evenly between 0.0
            and 1.0, excluding the boundaries.
    """

    def __init__(self, quantiles: Union[int, List, np.ndarray, tf.Tensor]):
        if isinstance(quantiles, int):
            self.quantiles = tf.linspace(0.0, 1.0, quantiles + 2)[1:-1]
        elif isinstance(quantiles, (List, np.ndarray)):
            self.quantiles = tf.convert_to_tensor(quantiles, dtype=tf.float32)
        elif isinstance(quantiles, tf.Tensor):
            self.quantiles = quantiles
        else:
            assert False, (
                    "QuantileLoss requires an int, List, np.ndarray, or "
                    "tf.Tensor.")

        for quantile in self.quantiles:
            assert quantile >= 0.0 and quantile <= 1.0, (
                    "QuantileLoss: All quantiles must be between 0.0 and 1.0.")

    def __call__(self, ygt: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Returns the total quantile loss.

        The length of ygt and the length of the first dimension of y must match,
        i.e., the number of examples must be compatible. The last dimension of
        y must match the number of quantiles in self.quantiles, i.e., there must
        be a prediction for each quantile value.

        Arguments:
            ygt: Ground truth values.
            y: Quantile predictions for each ground truth value.

        Returns:
            Loss
        """

        return quantile_loss(self.quantiles, ygt, y)

    def quantile_labels(self) -> List[str]:
        """Returns labels for each of the quantiles.

        Returns:
            A list with strings describing each quantile.
        """

        return [f'Q{int(100 * q):02d}%' for q in self.quantiles]
