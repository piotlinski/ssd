"""Loss functions."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from numpy import inf


class MultiBoxLoss(nn.Module):
    """SSD multiple bounding box loss.

    Combines classification loss and smooth L1 regression loss.
    """

    def __init__(self, negative_positive_ratio: float):
        super().__init__()
        self.negative_positive_ratio = negative_positive_ratio

    @staticmethod
    def hard_negative_mining(
        loss: torch.Tensor, labels: torch.Tensor, negative_positive_ratio: float
    ):
        """Suppress the presence of a large number of negative predictions
        Works on image level not batch level. For any example/image, it keeps all the
         positive predictions and cuts the number of negative predictions to make sure
         the ratio between the negative examples and positive examples is no more than
         the given ratio for an image.

        :param loss: (N, num_priors) the loss for each example
        :param labels: (N, num_priors) the labels
        :param negative_positive_ratio: the ratio between the negative examples and
            positive examples
        :return: mining mask
        """
        positive_mask = labels > 0
        n_positive = positive_mask.long().sum(dim=1, keepdim=True)
        n_negative = n_positive * negative_positive_ratio

        loss[positive_mask] = -inf
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        negative_mask = orders < n_negative

        return positive_mask | negative_mask

    def classification_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate classification loss

        :param predictions: (batch_size, num_priors, num_classes) class predictions
        :param labels: (batch_size, num_priors) real labels of all the priors
        :return: classification loss
        """
        n_classes = predictions.size(2)
        with torch.no_grad():
            loss = -functional.log_softmax(predictions, dim=2)[:, :, 0]
            mask = self.hard_negative_mining(
                loss=loss,
                labels=labels,
                negative_positive_ratio=self.negative_positive_ratio,
            )

        predictions = predictions[mask, :]
        return functional.cross_entropy(
            input=predictions.view(-1, n_classes), target=labels[mask], reduction="sum"
        )

    @staticmethod
    def box_regression_loss(
        predicted: torch.Tensor, ground_truth: torch.Tensor, positive_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate bounding box regression loss (smooth L1 loss)

        :param predicted: (batch_size, num_priors, 4) predicted locations
        :param ground_truth: (batch_size, num_priors, 4) real boxes corresponding all
            the priors
        :param positive_mask: mask for filtering only positive entries
        :return: regression loss
        """
        predicted = predicted[positive_mask, :].view(-1, 4)
        ground_truth = ground_truth[positive_mask, :].view(-1, 4)
        return functional.smooth_l1_loss(
            input=predicted, target=ground_truth, reduction="sum"
        )

    def forward(
        self,
        confidence: torch.Tensor,
        predicted_locations: torch.Tensor,
        labels: torch.tensor,
        gt_locations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate loss

        :param confidence: (batch_size, num_priors, num_classes) class predictions
        :param predicted_locations: (batch_size, num_priors, 4) predicted locations
        :param labels: (batch_size, num_priors) real labels of all the priors
        :param gt_locations: (batch_size, num_priors, 4) real boxes corresponding all
            the priors
        :return: regression loss and classification loss
        """
        classification_loss = self.classification_loss(
            predictions=confidence, labels=labels
        )

        positive_mask = labels > 0
        regression_loss = self.box_regression_loss(
            predicted=predicted_locations,
            ground_truth=gt_locations,
            positive_mask=positive_mask,
        )

        n_positive = gt_locations[positive_mask, :].view(-1, 4).size(0)
        return regression_loss / n_positive, classification_loss / n_positive
