"""Metrics for segmentation.
"""

import math

import numpy as np
import torch


class Metrics:
    """Tracking mean metrics"""

    def __init__(self, labels):
        """Creates an new `Metrics` instance.

        Args:
          labels: the labels for all classes.
        """

        self.labels = labels

        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0
        self.actual = 0
        self.predicted = 0
        self.M = 0

    def add(self, actual, predicted):
        """Adds an observation to the tracker.

        Args:
          actual: the ground truth labels.
          predicted: the predicted labels.
        """

        assert predicted.size(0) == 2
        masks = torch.argmax(predicted, 0)
        self.add_binary(actual, masks)

    def add_binary(self, actual, masks):
        # confusion = masks.view(-1).float() / actual.view(-1).float()

        self.tn += torch.sum(
            torch.logical_and(torch.logical_not(actual), torch.logical_not(masks))
        ).item()
        self.fn += torch.sum(torch.logical_and(torch.logical_not(masks), actual)).item()
        self.fp += torch.sum(torch.logical_and(masks, torch.logical_not(actual))).item()
        self.tp += torch.sum(torch.logical_and(actual, masks)).item()
        self.actual += torch.sum(actual).item()
        self.predicted += torch.sum(masks).item()
        self.M += actual.nelement()

    def get_miou(self):
        """Retrieves the mean Intersection over Union score.

        Returns:
          The mean Intersection over Union score for all observations seen so far.
        """
        try:
            miou = np.nanmean(
                [
                    self.tn / (self.tn + self.fn + self.fp),
                    self.tp / (self.tp + self.fn + self.fp),
                ]
            )
        except ZeroDivisionError:
            miou = float("NaN")

        return miou

    def get_fg_iou(self):
        """Retrieves the foreground Intersection over Union score.

        Returns:
          The foreground Intersection over Union score for all observations seen so far.
        """

        try:
            iou = self.tp / (self.tp + self.fn + self.fp)
        except ZeroDivisionError:
            iou = float("NaN")

        return iou

    def get_mcc(self):
        """Retrieves the Matthew's Coefficient Correlation score.

        Returns:
          The Matthew's Coefficient Correlation score for all observations seen so far.
        """

        try:
            mcc = (self.tp * self.tn - self.fp * self.fn) / math.sqrt(
                (self.tp + self.fp)
                * (self.tp + self.fn)
                * (self.tn + self.fp)
                * (self.tn + self.fn)
            )
        except ZeroDivisionError:
            mcc = float("NaN")

        return mcc


# Todo:
# - Rewrite mIoU to handle N classes (and not only binary SemSeg)
