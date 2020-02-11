''' Computes a Huber loss for neural net training and evaluation.

Computes the loss across buckets, but only on buckets that are
possible on a given board.
'''

from Source.Settings.arguments import arguments
import torch.nn as nn
import torch

def smoothL1LossForward(outputs, targets):
    ''' Calculate SmoothL1Loss of 2 vectors '''
    n = torch.abs(outputs - targets)
    beta = 1
    cond = n < beta
    z = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return z.mean()

def smoothL1LossGrad(outputs, targets):
    ''' Calculate gradiant of SmoothL1Loss with respect to outputs'''
    d = outputs - targets
    n = torch.abs(d)
    dloss_dn = n.clone()
    dloss_dn[n <= -1] = -1
    dloss_dn[n > 1] = 1
    dn_doutput = d / (n + 1e-10)
    dloss_doutput = dloss_dn / n.nelement() * dn_doutput
    return dloss_doutput

class MaskedHuberLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, outputs, targets, mask):
        ''' Computes the loss over a batch of neural net outputs and targets.

        Params:
            outputs: an NxM tensor containing N vectors of values over buckets,
                output by the neural net
            targets: an NxM tensor containing N vectors of actual values over
                buckets, produced by @{data_generation_call}
            mask: an NxM tensor containing N mask vectors generated with
                @{bucket_conversion.get_possible_bucket_mask}
        Return the sum of Huber loss applied elementwise on `outputs` and `targets`,
        masked so that only valid buckets are included'''
        batch_size = outputs.size(0)
        feature_size = outputs.size(1)
        
        # 1.0 zero out the outputs/target so that the error does not depend on these
        outputs.mul_(mask)
        targets.mul_(mask)
        
        loss = smoothL1LossForward(outputs, targets)
        
        # 2.0 if the batch size has changed, create new storage for the sum, otherwise reuse
        mask_placeholder = arguments.Tensor(mask.size()).fill_(0)
        mask_sum = arguments.Tensor(batch_size).fill_(0)
        mask_multiplier = mask_sum.clone().fill_(0).view(-1, 1)
        
        # 3.0 compute mask sum for each batch
        mask_placeholder.copy_(mask)
        mask_sum = mask_placeholder.sum(dim=1, keepdim=True)
        
        # 3.1 mask multiplier - note that mask is 1 for impossible features
        mask_multiplier.fill_(feature_size)
        mask_multiplier.sub_(mask_sum)
        mask_multiplier.div_(feature_size)
        
        # 4.0 multiply to get a new losss
        # loss is not really computed batch-wise correctly,
        # but that does not really matter now since gradients are correct
        loss_multiplier = (batch_size * feature_size) / (batch_size * feature_size - mask_sum.sum() )
        new_loss = loss_multiplier * loss

        ctx.save_for_backward(outputs, targets, mask_multiplier)
        
        return new_loss

    @staticmethod
    def backward(ctx, grad_out):
        ''' Computes the gradient of the loss function @{forward} with
        arguments `outputs`, `targets`, and `mask`.

        Must be called after a @{forward} call with the same arguments.

        Params:
            outputs: an NxM tensor containing N vectors of values over buckets,
                output by the neural net
            targets: an NxM tensor containing N vectors of actual values over
                buckets, produced by @{data_generation_call}
            mask: an NxM tensor containing N mask vectors generated with
                @{bucket_conversion.get_possible_bucket_mask}
        Return the gradient of @{forward} applied to the arguments'''
        outputs, targets, mask_multiplier = ctx.saved_tensors
        dloss_doutput = smoothL1LossGrad(outputs, targets)
        
        # we use the multiplier computed with the mask during forward call
        dloss_doutput.div_(mask_multiplier.expand_as(dloss_doutput))
        
        return dloss_doutput, None, None