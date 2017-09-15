import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MultiBoxLoss(nn.Module):

    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = 21

    def forward(self, loc_preds, loc_targets, conf_preds, label_targets):
        """Compute loss.
        Note: all arguments are Variables.

        Arguments:
            loc_preds: a float tensor of shape [n, 8732, 4],
                predicted locations.
            loc_targets:  a float tensor of shape [n, 8732, 4],
                encoded target locations.
            conf_preds: a float tensor of shape [n, 8732, 21],
                predicted class confidences.
            label_targets: a long tensor of shape [n, 8732],
                encoded target classes.

        Returns:
            a float number
        """

        # background class = 0,
        # pos means that a default bounding box is matched to
        # some ground truth box
        pos = label_targets > 0  # [n, 8732], n is batch size
        num_matched_boxes = pos.data.long().sum()
        if num_matched_boxes == 0:
            return Variable(torch.FloatTensor([0.0]))

        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)  # [n, 8732, 4]
        pos_loc_preds = loc_preds[pos_mask].view(-1, 4)  # [#pos, 4]
        pos_loc_targets = loc_targets[pos_mask].view(-1, 4)  # [#pos, 4]
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=False)

        conf_loss = cross_entropy_loss(
            conf_preds.view(-1, self.num_classes), label_targets.view(-1)
        )  # [n*8732]
        neg = hard_negative_mining(conf_loss, pos)  # [n, 8732]

        pos_and_neg = (pos + neg).gt(0)
        # it could happen that pos and neg overlap, but it is a very low
        # probability event. It shouldn't happen even if there are ~1000
        # pos boxes in an image. gt(0) operation is just in case

        mask = pos_and_neg.unsqueeze(2).expand_as(conf_preds)  # [n, 8732, 21]
        preds = conf_preds[mask].view(-1, self.num_classes)  # [#pos + #neg, 21]
        targets = conf_targets[pos_and_neg]  # [#pos + #neg]
        conf_loss = F.cross_entropy(preds, targets, size_average=False)
        # we are computing cross entropy the second time,
        # can we skip it?

        loc_loss /= num_matched_boxes
        conf_loss /= num_matched_boxes
        return loc_loss + conf_loss


def cross_entropy_loss(logits, targets):
    """Cross entropy loss w/o averaging across all samples.

    Arguments:
      logits: a float tensor of shape [n, d].
      targets: a long tensor of shape [n].

    Returns:
        a float tensor of shape [n].
    """
    x = logits
    x_max = x.data.max()
    log_sum_exp = torch.log(torch.sum(torch.exp(x - x_max), 1)) + x_max
    return log_sum_exp - x.gather(1, targets.view(-1, 1)).view(-1)


def hard_negative_mining(conf_loss, pos):
    """Return negative indices that is 3x the number as postive indices.

    Arguments:
        conf_loss: a float tensor of shape [n*8732],
            cross entropy loss between conf_preds and label_targets.
        pos: a byte tensor of shape [n, 8732], ones and zeros,
            positive(matched) box's indices.

    Returns:
        a byte tensor of shape [n, 8732], ones and zeros.
    """
    batch_size, num_boxes = pos.size()

    # i don't completely understand why
    # reshaping here works correctly
    conf_loss = conf_loss.view(batch_size, -1)  # [n, 8732]

    # we are only interested in boxes where
    # true background is confused with something else
    conf_loss[pos] = -1.0
    # so we don't consider boxes with true objects in them

    # sort default boxes by loss
    _, idx = conf_loss.sort(1, descending=True)  # [n, 8732]

    # rank of each default box,
    # lower rank -> higher loss
    _, rank = idx.sort(1, descending=False)  # [n, 8732]

    # number of matched boxes per image in batch
    num_pos = pos.long().sum(1)  # [n]
    num_neg = torch.clamp(3*num_pos, max=num_boxes - 1)  # [n]

    neg = rank < num_neg.unsqueeze(1).expand_as(rank)  # [n, 8732]
    # binary tensor, with ones where boxes are with high loss
    return neg
