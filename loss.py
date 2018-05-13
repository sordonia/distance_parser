import torch
import torch.nn.functional as F


def _assert_no_grad(variable):
  assert not variable.requires_grad, \
    "nn criterions don't compute the gradient w.r.t. targets - please " \
    "mark these variables as not requiring gradients"


def dist_loss(input, target):
  mask = (input > 0).float()
  diff = input[:, :, None] - input[:, None, :]
  target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
  mask = mask[:, :, None] * mask[:, None, :] * target_diff
  loss = F.relu(target_diff - diff)
  loss = (loss * mask).sum() / (mask.sum() + 1e-9)
  return loss


label_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
unary_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
bce = torch.nn.BCELoss(size_average=False)
