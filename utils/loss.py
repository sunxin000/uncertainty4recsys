import torch
from torch.nn.modules.loss import _Loss

class ConfidencePenalty(_Loss):

    epsilon = 1e-12

    def __init__(self, weight: float = 1.0, threshold: float = -1., reduction='mean'):
        super().__init__(reduction=reduction)
        self.weight = weight
        self.threshold = threshold

    def forward(self, input: torch.Tensor):
        probs = torch.clamp(input, self.epsilon, 1.-self.epsilon)
        loss = -self.weight * torch.mul(probs, torch.log(probs))
        if self.threshold > 0:
            loss = torch.maximum(0.0, self.threshold - loss)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'batchmean':
            return torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise AttributeError("Unknown reduction type \'%s\'." % self.reduction)