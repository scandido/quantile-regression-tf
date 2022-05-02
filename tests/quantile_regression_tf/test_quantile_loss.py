import numpy as np
from src.quantile_regression_tf import QuantileLoss
import tensorflow as tf
import unittest


class TestQuantileLoss(unittest.TestCase):

    def test_create_quantile_loss_with_int(self) -> None:
        ql = QuantileLoss(quantiles=9)
        np.testing.assert_allclose(
                ql.quantiles.numpy(),
                np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

    def test_create_quantile_loss_with_list(self) -> None:
        ql = QuantileLoss(quantiles=[0.23, 0.88, 0.45])
        np.testing.assert_allclose(
                ql.quantiles.numpy(),
                np.array([0.23, 0.88, 0.45]))

    def test_create_quantile_loss_with_numpy_array(self) -> None:
        ql = QuantileLoss(quantiles=np.array([0.23, 0.88, 0.45]))
        np.testing.assert_allclose(
                ql.quantiles.numpy(),
                np.array([0.23, 0.88, 0.45]))

    def test_create_quantile_loss_with_numpy_tensor(self) -> None:
        ql = QuantileLoss(quantiles=tf.constant([0.23, 0.88, 0.45]))
        np.testing.assert_allclose(
                ql.quantiles.numpy(),
                np.array([0.23, 0.88, 0.45]))

    def test_create_quantile_loss_doesnt_accept_other_types(self) -> None:
        with self.assertRaises(Exception) as context:
            ql = QuantileLoss(quantiles=None)
        self.assertTrue('QuantileLoss requires an ' in str(context.exception))

    def test_create_quantile_loss_requires_quantiles_in_unit_interval(
            self) -> None:
        with self.assertRaises(Exception) as context:
            ql = QuantileLoss(quantiles=[0.3, 0.7, 1.26])
        self.assertTrue(
                'All quantiles must be between' in str(context.exception))

    def test_quantile_loss_works(self) -> None:
        ql = QuantileLoss(quantiles=[0.3, 0.7, 0.99])
        loss = ql(
                tf.ones((22, 1), dtype=tf.float32),
                tf.ones((22, 3), dtype=tf.float32))
        self.assertEqual(loss.shape, (22,))

    def test_quantile_loss_requires_same_batch_size(self) -> None:
        ql = QuantileLoss(quantiles=[0.3, 0.7, 0.99])
        with self.assertRaises(Exception) as context:
            loss = ql(
                    tf.ones((23, 1), dtype=tf.float32),
                    tf.ones((22, 3), dtype=tf.float32))
        self.assertTrue('Batch sizes for predictions' in str(context.exception))

    def test_quantile_loss_requires_predictions_for_all_quantiles(self) -> None:
        ql = QuantileLoss(quantiles=[0.3, 0.7, 0.99])
        with self.assertRaises(Exception) as context:
            loss = ql(
                    tf.ones((22, 1), dtype=tf.float32),
                    tf.ones((22, 2), dtype=tf.float32))
        self.assertTrue('Predictions must match the' in str(context.exception))

    def test_quantile_loss_quantile_labels(self) -> None:
        ql = QuantileLoss(quantiles=[0.33, 0.1, 0.982])
        self.assertTrue(ql.quantile_labels(), ['Q33%', 'Q01%', 'Q98%'])


if __name__ == "__main__":
  unittest.main()
