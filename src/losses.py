"""PyTorch-compatible losses and loss functions.
"""

import torch
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    """Cross-entropy.

    See: http://cs231n.github.io/neural-networks-2/#losses
    """

    def __init__(self, weight=None):
        """Creates an `CrossEntropyLoss2d` instance.

        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)


class FocalLoss2d(nn.Module):
    """Focal Loss.

    Reduces loss for well-classified samples putting focus on hard
    mis-classified samples.

    See: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weight=None):
        """Creates a `FocalLoss2d` instance.

        Args:
          gamma: the focusing parameter; if zero this loss is equivalent with
          `CrossEntropyLoss2d`.
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.type(torch.int64)
        penalty = (1 - nn.functional.softmax(inputs, dim=1)) ** self.gamma
        return self.nll_loss(
            penalty * nn.functional.log_softmax(inputs, dim=1), targets
        )


class mIoULoss2d(nn.Module):
    """mIoU Loss.

    See:
      - http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
      - http://www.cs.toronto.edu/~wenjie/papers/iccv17/mattyus_etal_iccv17.pdf
    """

    def __init__(self, weight=None):
        """Creates a `mIoULoss2d` instance.

        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):

        targets = targets.type(torch.int64)
        N, C, H, W = inputs.size()

        softs = nn.functional.softmax(inputs, dim=1).permute(1, 0, 2, 3)
        masks = (
            torch.zeros(N, C, H, W)
            .to(targets.device)
            .scatter_(1, targets.view(N, 1, H, W), 1)
            .permute(1, 0, 2, 3)
        )

        inters = softs * masks
        unions = (softs + masks) - (softs * masks)

        miou = (
            1.0 - (inters.view(C, N, -1).sum(2) / unions.view(C, N, -1).sum(2)).mean()
        )

        return max(
            miou, self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)
        )


class LovaszLoss2d(nn.Module):
    """Lovasz Loss.

    See: https://arxiv.org/abs/1705.08790
    """

    def __init__(self):
        """Creates a `LovaszLoss2d` instance."""
        super().__init__()

    def forward(self, inputs, targets):

        targets = targets.type(torch.int64)
        N, C, H, W = inputs.size()
        masks = (
            torch.zeros(N, C, H, W)
            .to(targets.device)
            .scatter_(1, targets.view(N, 1, H, W), 1)
        )

        loss = 0.0

        for mask, input in zip(masks.view(N, -1), inputs.view(N, -1)):

            max_margin_errors = 1.0 - ((mask * 2 - 1) * input)
            errors_sorted, indices = torch.sort(max_margin_errors, descending=True)
            labels_sorted = mask[indices.data]

            inter = labels_sorted.sum() - labels_sorted.cumsum(0)
            union = labels_sorted.sum() + (1.0 - labels_sorted).cumsum(0)
            iou = 1.0 - inter / union

            p = len(labels_sorted)
            if p > 1:
                iou[1:p] = iou[1:p] - iou[0:-1]

            loss += torch.dot(nn.functional.relu(errors_sorted), iou)

        return loss / N


class FLoss2d(nn.Module):
    """F-score-like differentiable loss

    based on
    https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook
    and
    https://link.springer.com/chapter/10.1007/978-3-642-38679-4_37

    The best loss function would be, of course the metric itself. Then the
    misalignment disappears. The macro F1-score has one big trouble. It's
    non-differentiable. Which means we cannot use it as a loss function.

    But we can modify it to be differentiable. Instead of accepting 0/1 integer
    predictions, let's accept probabilities instead. Thus if the ground truth is
    1 and the model prediction is 0.4, we calculate it as 0.4 true positive and
    0.6 false negative. If the ground truth is 0 and the model prediction is
    0.4, we calculate it as 0.6 true negative and 0.4 false positive.

    Also, we minimize 1-F1 (because minimizing 1−f(x) is same as maximizing f(x))
    """

    def __init__(self):
        """Creates a `FLoss2d` instance."""
        super().__init__()
        self.eps = 1e-10

    def forward(self, inputs, targets):
        inputs = inputs.type(torch.float32)
        targets = targets.type(torch.int64)
        probs = nn.functional.softmax(inputs, dim=1)[:, 1]  # Class 1 probability
        tp = torch.sum((targets * probs))
        fp = torch.sum(((1 - targets) * probs))
        fn = torch.sum(((targets) * (1 - probs)))

        p = tp / (tp + fp + self.eps)
        r = tp / (tp + fn + self.eps)

        f1 = 2 * p * r / (p + r + self.eps)
        return 1 - f1
