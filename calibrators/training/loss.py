from multiprocessing import reduction
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dtw import dtw
from .exceptions import ParamException

from calibrators.training.flsd import FocalLossAdaptive

class FocalLoss(nn.Module):
    def __init__(self, ignore_index=0, alpha=1, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = alpha
        self.ignore_index = ignore_index

    def forward(self, input, target, *arg):
        input, target  = input.view(-1, input.shape[-1]), target.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(target != self.ignore_index).squeeze()
            input = input[index, :]
            target = target[index]

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return loss.mean()

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

class ClassficationAndMDCA(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=3.0, ignore_index=0, **kwargs):
        super(ClassficationAndMDCA, self).__init__()
        self.beta = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.classification_loss = FocalLoss(gamma=self.gamma)
        self.MDCA = MDCA()

    def forward(self, inputs, targets, *args):
        inputs, targets  = inputs.view(-1, inputs.shape[-1]), targets.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]
        loss_cls = self.classification_loss(inputs, targets)
        loss_cal = self.MDCA(inputs, targets)
        return loss_cls + self.beta * loss_cal

class BrierScore(nn.Module):
    def __init__(self, ignore_index=0, **kwargs):
        super(BrierScore, self).__init__()
        self.ignore_index = ignore_index
    def forward(self, logits, target, *args):
        logits, target  = logits.view(-1, logits.shape[-1]), target.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(target != self.ignore_index).squeeze()
            logits = logits[index, :]
            target = target[index]     

        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(logits.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = torch.softmax(logits, dim=1)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(logits.shape[0])
        return loss

class FLSD(nn.Module):
    def __init__(self, ignore_index=0, gamma=3.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.criterion = FocalLossAdaptive(gamma=self.gamma)

    def forward(self, logits, target, *args):
        logits, target  = logits.view(-1, logits.shape[-1]), target.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(target != self.ignore_index).squeeze()
            logits = logits[index, :]
            target = target[index]   
        return self.criterion.forward(logits, target)

class LogitMarginL1(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)

    Args:
        margin (float, optional): The margin value. Defaults to 10.
        alpha (float, optional): The balancing weight. Defaults to 0.1.
        ignore_index (int, optional):
            Specifies a target value that is ignored
            during training. Defaults to -100.

        The following args are related to balancing weight (alpha) scheduling.
        Note all the results presented in our paper are obtained without the scheduling strategy.
        So it's fine to ignore if you don't want to try it.

        schedule (str, optional):
            Different stragety to schedule the balancing weight alpha or not:
            "" | add | multiply | step. Defaults to "" (no scheduling).
            To activate schedule, you should call function
            `schedula_alpha` every epoch in your training code.
        mu (float, optional): scheduling weight. Defaults to 0.
        max_alpha (float, optional): Defaults to 100.0.
        step_size (int, optional): The step size for updating alpha. Defaults to 100.
    """
    def __init__(self,
                 margin: float = 10,
                 alpha: float = 0.1,
                 ignore_index: int = 0,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 **kwargs):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()


    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets, *args):
        inputs, targets  = inputs.view(-1, inputs.shape[-1]), targets.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff-self.margin).mean()
        loss = loss_ce + self.alpha * loss_margin

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 ignore_index=0,
                 **kwargs):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    def forward(self, preds, target, *args):
        loss = self.loss_func(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        return loss

class boost_soft(nn.Module):
    def __init__(self,
                 ignore_index: int,
                 alpha: float,
                 normalize_length: bool = True,
                 **kwargs):
        """Construct an LabelSmoothingLoss object."""
        super(boost_soft, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
        self.beta = 0.95

    def forward(self, preds: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
        batch_size = preds.size(0)
        preds_softmax = F.softmax(preds, dim=2)

        preds = preds.view(-1, preds.size(-1))
        preds_softmax = preds_softmax.view(-1, preds_softmax.size(-1))
        target = target.view(-1)

        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(preds)
        true_dist.fill_(self.smoothing / (preds.size(-1) - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        soft_dist = self.beta * true_dist + (1-self.beta) * preds_softmax
        kl = self.criterion(torch.log_softmax(preds, dim=1), soft_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class boost_hard(nn.Module):
    def __init__(self,
                 ignore_index: int,
                 alpha: float,
                 normalize_length: bool = True,
                 **kwargs):
        """Construct an LabelSmoothingLoss object."""
        super(boost_hard, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
        self.beta = 0.95
        self.max_length=26
        self.eos_index = 1

    def forward(self, preds: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
        batch_size = preds.size(0)
        hard_dist = torch.zeros_like(preds)
        _, preds_index = preds.max(2)  # B T 
        
        b = torch.nonzero(preds_index==1)
        # eos_pos_list = []
        new_preds_index = torch.zeros_like(preds_index)
        i = 0
        for n in range(len(b)):
            if i == batch_size:
                break
            else:
                if b[n][0] == i:
                    # eos_pos_list.append(b[n][1])
                    new_preds_index[i][:(b[n][1]+1)] = preds_index[i][:(b[n][1]+1)]
                    i = i+1
        
        new_preds_index = new_preds_index.view(-1)
        ignore = new_preds_index == self.padding_idx
        new_preds_index = new_preds_index.masked_fill(ignore, 0)

        preds = preds.view(-1, preds.size(-1))
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(preds)
        new_pred_dist = torch.zeros_like(preds)
        true_dist.fill_(self.smoothing / (preds.size(-1) - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        new_pred_dist.scatter_(1, new_preds_index.unsqueeze(1), self.confidence)
        
        hard_dist = self.beta * true_dist + (1-self.beta) * new_pred_dist

        kl = self.criterion(torch.log_softmax(preds, dim=1), hard_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class LabelSmoothingLoss(nn.Module):
    def __init__(self,
                 ignore_index: int,
                 alpha: float,
                 normalize_length: bool = True,
                 **kwargs):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length

    def forward(self, preds: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
        batch_size = preds.size(0)
        preds = preds.view(-1, preds.size(-1))
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(preds)
        true_dist.fill_(self.smoothing / (preds.size(-1) - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(preds, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class robust(nn.Module):
    def __init__(self, ignore_index: int,
                 alpha: float,
                 normalize_length: bool = True,
                 num_iter = 500000,
                 B = 8.0,
                 t = 0.2,
                 **kwargs):
        super(robust, self).__init__()
        self.kldiv = nn.KLDivLoss(reduction="none")
        self.ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length

        self.total_iterations = num_iter
        self.exp_base = B
        self.counter = "iteration"
        self.epsilon = None
        self.transit_time_ratio = t

        if not (self.exp_base > 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero"

            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))
    
    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[2]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 2
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform
            #avg T
            # example_trust = example_trust.mean(1)
            # the trade-off
            self.epsilon = global_trust * example_trust
            # from shape [N] to shape [N, 1]
            # self.epsilon = self.epsilon[:, None]

    def forward(self, preds: torch.Tensor, target: torch.Tensor, cur_time, *args) -> torch.Tensor:
        
        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        
        B, T, C = preds.size()
        preds_prob = F.softmax(preds, dim=1)
        self.update_epsilon_progressive_adaptive(preds_prob, cur_time)
        
        loss = self.ce_criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        # kl_inp = preds.transpose(0, 1)
        
        # uni_distri = torch.full_like(preds, 1 / C)
        ent_loss = 0
        for i in range(B):
            ent_i = (preds_prob[i,:,:] * torch.log(preds_prob[i,:,:]) * torch.unsqueeze(self.epsilon[i],1).expand(-1, C)).mean()
            ent_loss += ent_i

        ent_loss = ent_loss / B  
        loss = loss - ent_loss
        
        return loss

class robust_wog(nn.Module):
    def __init__(self, ignore_index: int,
                 alpha: float,
                 normalize_length: bool = True,
                 num_iter = 300000,
                 B = 10.0,
                 t = 0.2,
                 **kwargs):
        super(robust_wog, self).__init__()
        self.kldiv = nn.KLDivLoss(reduction="none")
        self.ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length

        self.total_iterations = num_iter
        self.exp_base = B
        self.counter = "iteration"
        self.epsilon = None
        self.transit_time_ratio = t

        if not (self.exp_base > 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero"

            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))
    
    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[2]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 2
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform
            #avg T
            # example_trust = example_trust.mean(1)
            # the trade-off
            global_trust = 1
            self.epsilon = global_trust * example_trust
            # from shape [N] to shape [N, 1]
            # self.epsilon = self.epsilon[:, None]

    def forward(self, preds: torch.Tensor, target: torch.Tensor, cur_time, *args) -> torch.Tensor:
        
        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        
        B, T, C = preds.size()
        preds_prob = F.softmax(preds, dim=1)
        self.update_epsilon_progressive_adaptive(preds_prob, cur_time)
        
        loss = self.ce_criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        # kl_inp = preds.transpose(0, 1)
        
        # uni_distri = torch.full_like(preds, 1 / C)
        ent_loss = 0
        for i in range(B):
            ent_i = (preds_prob[i,:,:] * torch.log(preds_prob[i,:,:]) * torch.unsqueeze(self.epsilon[i],1).expand(-1, C)).mean()
            ent_loss += ent_i

        ent_loss = ent_loss / B  
        loss = loss - ent_loss
        
        return loss


class robust_wol(nn.Module):
    def __init__(self, ignore_index: int,
                 alpha: float,
                 normalize_length: bool = True,
                 num_iter = 300000,
                 **kwargs):
        super(robust_wol, self).__init__()
        self.kldiv = nn.KLDivLoss(reduction="none")
        self.ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length

        self.total_iterations = num_iter
        self.exp_base = 10
        self.counter = "iteration"
        self.epsilon = None
        self.transit_time_ratio = 0.2

        if not (self.exp_base > 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero"

            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))
    
    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[2]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 2
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform
            #avg T
            # example_trust = example_trust.mean(1)
            # the trade-off
            example_trust_l = torch.ones_like(example_trust)
            self.epsilon = global_trust * example_trust_l
            # from shape [N] to shape [N, 1]
            # self.epsilon = self.epsilon[:, None]

    def forward(self, preds: torch.Tensor, target: torch.Tensor, cur_time, *args) -> torch.Tensor:
        
        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        
        B, T, C = preds.size()
        preds_prob = F.softmax(preds, dim=1)
        self.update_epsilon_progressive_adaptive(preds_prob, cur_time)
        
        loss = self.ce_criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        # kl_inp = preds.transpose(0, 1)
        
        # uni_distri = torch.full_like(preds, 1 / C)
        ent_loss = 0
        for i in range(B):
            ent_i = (preds_prob[i,:,:] * torch.log(preds_prob[i,:,:]) * torch.unsqueeze(self.epsilon[i],1).expand(-1, C)).mean()
            ent_loss += ent_i

        ent_loss = ent_loss / B  
        loss = loss - ent_loss
        
        return loss
        


class CaslsChineseAttnLoss(nn.Module):
    def __init__(self, ignore_index, matric, alpha=0.0, normalize_length=True, device=None, **kwargs):
        super(CaslsChineseAttnLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smoothing = alpha
        self.matric = matric
        self.device = device
        self.kldiv = nn.KLDivLoss(reduction="none")
        self.normalize_length = normalize_length
    def forward(self, inputs, targets, _, labels):
        inputs, targets  = inputs.view(-1, inputs.shape[-1]), targets.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]
        nclass = inputs.shape[-1]

        i = 0
        need_smoothed = []
        for j in torch.nonzero(targets==1).squeeze():
            forth = torch.ones_like(targets[i:j+1]) * (targets.shape[-1] - 1)
            forth[1:] = targets[i:j]
            need_smoothed.append(torch.tensor(self.matric[forth.tolist(), targets[i:j+1].tolist()]))
            i = j + 1
        need_smoothed = torch.cat(need_smoothed, 0)

        length = torch.IntTensor(list(map(len, labels))) + 1
        smoothing = 1-torch.pow(1-self.smoothing, 1/length).view(-1,1)
        smooth_list = []
        for sm, l in zip(smoothing, length):
            sm = sm.repeat(l)
            smooth_list.append(sm)
        smoothing = torch.cat(smooth_list) * need_smoothed

        weight = torch.ones_like(inputs) * (smoothing / (nclass-1)).view(-1,1).to(self.device)
        src = (1. - weight.sum(dim=1))
        weight = weight.scatter(-1, targets.view(-1,1), src.view(-1,1))
        kl = self.kldiv(torch.log_softmax(inputs, dim=1), weight)

        return kl.sum()/length.sum()

class PairWiseWeightSmoothLoss(nn.Module):
    def __init__(self, 
                ignore_index, 
                matric, 
                alpha=0.0, 
                normalize_length=True, 
                device=None, 
                **kwargs):
        super(PairWiseWeightSmoothLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.matric = torch.tensor(matric, dtype=torch.float32)[:-1, :-1, :-1]
        self.normalize_length = normalize_length
        self.device = device
    def forward(self, input, target, _, labels):
        length = torch.FloatTensor(list(map(len, labels))) + 1.0
        batch_size, max_time_len, _ = input.shape

        forth_target = torch.zeros_like(target)
        forth_target[:, 1:] = target[:, :-1]
        forth_target = forth_target.view(-1)
        target = target.contiguous().view(-1)
        ignore = target == self.padding_idx
        total = (ignore == True).sum().item()
        input = input.view(-1, input.shape[-1])
        log_prob = F.log_softmax(input, dim=-1)

        smoothing = (1 - torch.pow((1-self.smoothing), 1.0 / length)).unsqueeze(1).repeat(1, max_time_len).view(-1)
        weight = (smoothing.unsqueeze(1) * self.matric[forth_target.tolist(), target.tolist(), :]).to(self.device)
        src = (1. -  weight.sum(dim=1))
        src = src.unsqueeze(-1).repeat(1, weight.size(1)) 
        weight.scatter_(-1, target.unsqueeze(1), src)
        weight = weight.type_as(log_prob)

        denom = total if self.normalize_length else batch_size
        loss = (-weight * log_prob).masked_fill(ignore.unsqueeze(1), 0).sum() / denom

        return loss

class CASLSRobustSmoothLoss(nn.Module):
    def __init__(self, 
                ignore_index, 
                matric, 
                alpha=0.0, 
                normalize_length=True, 
                device=None, 
                **kwargs):
        super(CASLSRobustSmoothLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kldiv = nn.KLDivLoss()
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.matric = torch.tensor(matric, dtype=torch.float32)[:-1, :-1, :-1]
        self.normalize_length = normalize_length
        self.device = device
    def forward(self, input, target, _, labels):
        length = torch.FloatTensor(list(map(len, labels))) + 1.0
        batch_size, max_time_len, _ = input.shape

        ce_loss = self.ce_loss(input.view(-1, input.shape[-1]), target.contiguous().view(-1))

        
        forth_target = torch.zeros_like(target)
        forth_target[:, 1:] = target[:, :-1]
        forth_target = forth_target.view(-1)
        target = target.contiguous().view(-1)
        ignore = target == self.padding_idx
        total = (ignore == True).sum().item()
        input = input.view(-1, input.shape[-1])
        log_prob = F.log_softmax(input, dim=-1)
        preds_prob = F.softmax(input, dim=1)

        smoothing = (1 - torch.pow((1-self.smoothing), 1.0 / length)).unsqueeze(1).repeat(1, max_time_len).view(-1)
        weight = (smoothing.unsqueeze(1) * self.matric[forth_target.tolist(), target.tolist(), :]).to(self.device)
        
        
        src = (1. -  weight.sum(dim=1))
        src = src.unsqueeze(-1).repeat(1, weight.size(1)) 
        weight.scatter_(-1, target.unsqueeze(1), src)
        weight = weight.type_as(preds_prob)

        # denom = total if self.normalize_length else batch_size
        # loss = (-weight * log_prob).masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        kl_loss =  self.kldiv(preds_prob, weight)
        loss = ce_loss + kl_loss

        return loss

class CASLSRobustSmoothLoss_v2(nn.Module):
    def __init__(self, opt,
                ignore_index, 
                matric, 
                alpha=0.0, 
                normalize_length=True, 
                device=None, 
                **kwargs):
        super(CASLSRobustSmoothLoss_v2, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kldiv = nn.KLDivLoss()
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.matric = torch.tensor(matric, dtype=torch.float32)[:-1, :-1, :-1]
        self.normalize_length = normalize_length
        self.device = device

        self.total_iteration = opt.num_iter
        self.exp_base = 4
        self.counter = "iteration"
        self.epsilon = None
        self.transit_time_ratio = 0.2

        if not (self.exp_base > 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero"

            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))
    
    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[1]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 1
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform
            # the trade-off
            self.epsilon = global_trust * example_trust
            # from shape [N] to shape [N, 1]
            self.epsilon = self.epsilon[:, None]


    def forward(self, input, target, _, labels, cur_time):

        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))

        length = torch.FloatTensor(list(map(len, labels))) + 1.0
        batch_size, max_time_len, nclass = input.shape

        ce_loss = self.ce_loss(input.view(-1, input.shape[-1]), target.contiguous().view(-1))

        
        forth_target = torch.zeros_like(target)
        forth_target[:, 1:] = target[:, :-1]
        forth_target = forth_target.view(-1)
        target = target.contiguous().view(-1)
        ignore = target == self.padding_idx
        total = (ignore == True).sum().item()
        input = input.view(-1, input.shape[-1])
        log_prob = F.log_softmax(input, dim=-1)
        preds_prob = F.softmax(input, dim=1)

        self.update_epsilon_progressive_adaptive(preds_prob, cur_time)

        smoothing = (1 - torch.pow((1-self.smoothing), 1.0 / length)).unsqueeze(1).repeat(1, max_time_len).view(-1)
        weight = (smoothing.unsqueeze(1) * self.matric[forth_target.tolist(), target.tolist(), :]).to(self.device)
        
        
        src = (1. -  weight.sum(dim=1))
        src = src.unsqueeze(-1).repeat(1, weight.size(1)) 
        weight.scatter_(-1, target.unsqueeze(1), src)
        weight = weight.type_as(preds_prob)

        # denom = total if self.normalize_length else batch_size
        # loss = (-weight * log_prob).masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        kl_loss =  self.kldiv(preds_prob, weight)


        loss = ce_loss + self.epsilon * kl_loss

        return loss

class EntropyRegularAttn(nn.Module):
    def __init__(self, ignore_index, alpha=0.05, **kwargs):
        super(EntropyRegularAttn, self).__init__()
        self.beta = alpha
        self.ignore_index = ignore_index
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, *args): 
        inputs, targets  = inputs.view(-1, inputs.shape[-1]), targets.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]
        ce_loss = self.ce_loss(inputs, targets)
        entrop_all = -(inputs.softmax(-1)*inputs.log_softmax(-1)).sum() / inputs.shape[0]
        loss = ce_loss - self.beta * entrop_all
        return loss

class GraduatedLabelSmoothingAttn(nn.Module):
    def __init__(self, 
                ignore_index=0,
                alpha=0.0,
                 normalize_length=True,
                **kwargs):
        super(GraduatedLabelSmoothingAttn, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
    def forward(self, input, target, *args):
        self.size = input.size(-1)
        batch_size = input.size(0)
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        
        pred_probability, _ = torch.softmax(input, dim=1).max(1)
        smoothing = self.smoothing * torch.ones_like(input)
        smoothing[pred_probability >= 0.7, :] = 3 * self.smoothing
        smoothing[pred_probability <= 0.3, :] = 0.0
        true_dist = smoothing / (self.size - 1)
        confidence = 1 - true_dist.sum(-1)
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        #target = target.masked_fill(ignore, 0)  0
        true_dist.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        kl = self.criterion(torch.log_softmax(input, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


class SequenceSmoothLoss_v10(nn.Module):
    '''
        只把vis和semantic的str拿出来, 然后 loss = (1 - self.alpha) * loss_master + self.alpha * loss_smooth,
        没有用到conf, 而且每个str的损失是一致对待
    '''
    def __init__(self, 
                 converter,
                 semantic,
                 ignore_index=0,
                 alpha=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ],[0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    def forward(self, preds, target, visual, labels):
        loss_master = self.loss_func(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]

        text, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])
        text, length = torch.cat(text,dim=0), torch.cat(length,dim=0)
        text = text[:, 1:] 

        preds = preds.unsqueeze(1).repeat(1, text.shape[0] // preds.shape[0] , 1, 1)
        loss_smooth = self.loss_func(preds.view(-1, preds.shape[-1]), text.contiguous().view(-1))
        loss = (1 - self.alpha) * loss_master + self.alpha * loss_smooth
        return loss



class SequenceSmoothLoss_v11(nn.Module):
    '''
        在v10基础上, 加入随置信度减小而增大的校准强度
    '''
    def __init__(self, 
                 converter,
                 semantic,
                 device,
                 smooth_tail=0.,
                 ignore_index=0,
                 eos=1,
                 gamma=2,
                 alpha=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ], [0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eos = eos
        self.smooth_tail = smooth_tail
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.loss_master = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.device = device
    def forward(self, preds, target, visual, labels): 
        bs =  preds.shape[0]
        loss_master = self.loss_master(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        loss_master = loss_master.view(*preds.shape[:-1]).sum(dim=-1) / (torch.tensor(list(map(len, labels))) + 1).to(self.device)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        _, preds_index = preds.max(2)

        eos_loc = torch.tensor([i.tolist().index(self.eos) if self.eos in i else (preds_index.shape[-1] + 1) for i in preds_index])
        arange = torch.arange(preds_index.shape[-1])
        mask = arange.expand(*preds_index.shape)
        mask = mask >= eos_loc.unsqueeze(-1)
        preds_max_prob[mask] = 1.0
        preds_str_prob = preds_max_prob.cumprod(dim=-1)[:, -1].detach().clone()




        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]
        texts, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])

        loss_smooth = []
        for text, pred in zip(texts, preds):
            text = text[:, 1:]
            pred = pred.unsqueeze(0).repeat(text.shape[0], 1, 1)
            loss_smooth.append(self.loss_func(pred.view(-1, pred.shape[-1]), text.contiguous().view(-1)).unsqueeze(0))
        loss_smooth = torch.cat(loss_smooth)


        ranking = self.smooth_tail + (1.0 - self.smooth_tail) * torch.pow((1 - preds_str_prob), 2)

        # ranking[ranking < 0.1] = 0.1
        # _, idx = preds_str_prob.topk(int(0.05 * preds_str_prob.shape[0]))
        # ranking = torch.ones_like(preds_str_prob)
        # ranking[idx] = 1

        loss = loss_master + self.alpha * ranking * loss_smooth


        return loss.mean()

class SequenceSmoothLoss_v12(nn.Module):
    '''
        在v10基础上, 加入随置信度减小而增大的校准强度
    '''
    def __init__(self, 
                 converter,
                 semantic,
                 device,
                 ignore_index=0,
                 eos=1,
                 gamma=2,
                 alpha=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ], [0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eos = eos
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.loss_master = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.device = device
    def forward(self, preds, target, visual, labels): 
        bs =  preds.shape[0]
        loss_master = self.loss_master(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        loss_master = loss_master.view(*preds.shape[:-1]).sum(dim=-1) / (torch.tensor(list(map(len, labels))) + 1).to(self.device)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        _, preds_index = preds.max(2)

        eos_loc = torch.tensor([i.tolist().index(self.eos) if self.eos in i else (preds_index.shape[-1] + 1) for i in preds_index])
        arange = torch.arange(preds_index.shape[-1])
        mask = arange.expand(*preds_index.shape)
        mask = mask >= eos_loc.unsqueeze(-1)
        preds_max_prob[mask] = 1.0
        preds_str_prob = preds_max_prob.cumprod(dim=-1)[:, -1].detach().clone()




        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]
        texts, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])

        loss_smooth = []
        for text, pred in zip(texts, preds):
            text = text[:, 1:]
            pred = pred.unsqueeze(0).repeat(text.shape[0], 1, 1)
            loss_smooth.append(self.loss_func(pred.view(-1, pred.shape[-1]), text.contiguous().view(-1)).unsqueeze(0))


        loss_smooth = torch.cat(loss_smooth)

        ranking = torch.pow((1 - preds_str_prob), self.gamma)
        ranking[ranking < 0.03] = 0.03
        _, idx = preds_str_prob.topk(int(0.05 * preds_str_prob.shape[0]))
        # ranking = torch.ones_like(preds_str_prob) * 0.1
        ranking[idx] = 0.3

        loss = loss_master + self.alpha * ranking * loss_smooth
        #loss = loss_master + self.alpha * loss_smooth
        # smoothing = self.alpha * (1 - preds_str_prob)
        # for smooth, master, conf in zip(loss_smooth, loss_master, preds_str_prob):
        #     if conf > 0.80:
        #         alpha = 0.00
        #     else:
        #         alpha = 0.1
        #     loss += ((1 - alpha) * master + alpha * smooth)

        return loss.mean()

class SequenceSmoothLoss_v13(nn.Module):
    def __init__(self, 
                 converter,
                 semantic,
                 device,
                 smooth_tail=0.,
                 ignore_index=0,
                 eos=1,
                 gamma=2,
                 alpha=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ], [0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eos = eos
        self.smooth_tail = smooth_tail
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.loss_master = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.device = device
    def forward(self, preds, target, visual, labels): 
        bs =  preds.shape[0]
        loss_master = self.loss_master(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        loss_master = loss_master.view(*preds.shape[:-1]).sum(dim=-1) / (torch.tensor(list(map(len, labels))) + 1).to(self.device)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        _, preds_index = preds.max(2)

        eos_loc = torch.tensor([i.tolist().index(self.eos) if self.eos in i else (preds_index.shape[-1] + 1) for i in preds_index])
        arange = torch.arange(preds_index.shape[-1])
        mask = arange.expand(*preds_index.shape)
        mask = mask >= eos_loc.unsqueeze(-1)
        preds_max_prob[mask] = 1.0
        preds_str_prob = preds_max_prob.cumprod(dim=-1)[:, -1].detach().clone()




        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]
        texts, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])

        loss_smooth = []
        for text, pred in zip(texts, preds):
            text = text[:, 1:]
            pred = pred.unsqueeze(0).repeat(text.shape[0], 1, 1)
            loss_smooth.append(self.loss_func(pred.view(-1, pred.shape[-1]), text.contiguous().view(-1)).unsqueeze(0))
        loss_smooth = torch.cat(loss_smooth)


        #ranking = self.smooth_tail + (1.0 - self.smooth_tail) * torch.pow((1 - preds_str_prob), 1/2)

        # ranking[ranking < 0.1] = 0.1
        # _, idx = preds_str_prob.topk(int(0.05 * preds_str_prob.shape[0]))
        ranking = torch.ones_like(preds_str_prob)
        # ranking[idx] = 1

        loss = loss_master + self.alpha * ranking * loss_smooth


        return loss.mean()


class SequenceSmoothLoss_num(nn.Module):
    def __init__(self, smooth_matrix, statistic, smooth_head=0.05, smooth_tail=0.015, class_num='0123456789abcdefghijklmnopqrstuvwxyz#'):
        super(SequenceSmoothLoss_num, self).__init__()

        self.smoothing_map = smooth_matrix
        self.statistic = statistic
        self.smooth_head = smooth_head
        self.smooth_tail =smooth_tail
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to('cuda')
        self.n_1 = math.sqrt(max(statistic.values()))
        self.n_K = math.sqrt(min(statistic.values()))
        self.smooth_caculate = lambda n_y: self.smooth_head + (self.smooth_head - self.smooth_tail) * np.sin(1.5 * np.pi + (n_y - self.n_K) * np.pi / (2 * (self.n_1 - self.n_K)))
    def forward(self, input, gt_strs, encode):
        norm_len = input.shape[0] * input.shape[1]
        target_ls = []
        input_ls = []
        smoothing_ls = []
        for idx, gt in enumerate(gt_strs):
            text, _ = encode([gt], batch_max_length=25)
            target_ls.append(text[:, 1:])
            input_ls.append(input[idx])
            if gt in self.smoothing_map.keys() and len(self.smoothing_map[gt]) > 0:
                smoothing = self.smooth_caculate(math.sqrt(self.statistic[gt]))
                smoothing_ls.append((1 - smoothing)*torch.full_like(text[:, 1:], 1.0).float())
                candidate_dict = self.smoothing_map[gt]
                sum_num = sum(candidate_dict.values())
                for k,v in candidate_dict.items():
                    input_ls.append(input[idx])
                    text, _ = encode([k], batch_max_length=25)
                    target_ls.append(text[:, 1:])
                    smoothing_ls.append((v/sum_num) * smoothing*torch.full_like(text[:, 1:], 1.0).float())
            else:
                smoothing_ls.append(torch.full_like(text[:, 1:], 1.0).float())
        target = torch.cat(target_ls).cuda()
        input = torch.cat(input_ls).cuda()
        smoothing = torch.cat(smoothing_ls).cuda()
        cost = (smoothing.view(-1) * self.criterion(input.view(-1, input.shape[-1]), target.contiguous().view(-1))).sum() / norm_len
        return cost


class SequenceSmoothLoss_v1(nn.Module):
    def __init__(self, smooth_matrix, alpha=0.05):
        super(SequenceSmoothLoss_v1, self).__init__()
        self.smoothing = alpha
        self.smoothing_map = smooth_matrix
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to('cuda')

    def forward(self, input, gt_strs, encode):
        norm_len = input.shape[0] * input.shape[1]
        target_ls = []
        input_ls = []
        smoothing_ls = []
        for idx, gt in enumerate(gt_strs):
            text, length = encode([gt], batch_max_length=25)
            target_ls.append(text[:, 1:])
            input_ls.append(input[idx])
            if gt in self.smoothing_map.keys() and len(self.smoothing_map[gt]) > 0:
                smoothing_ls.append((1 - self.smoothing)*torch.full_like(text[:, 1:], 1.0).float())
                candidate_dict = self.smoothing_map[gt]
                sum_num = sum(candidate_dict.values())
                for k,v in candidate_dict.items():
                    input_ls.append(input[idx])
                    text, _ = encode([k], batch_max_length=25)
                    target_ls.append(text[:, 1:])
                    smoothing_ls.append((v/sum_num) * self.smoothing*torch.full_like(text[:, 1:], 1.0).float())
            else:
                smoothing_ls.append(torch.full_like(text[:, 1:], 1.0).float())
        target = torch.cat(target_ls).cuda()
        input = torch.cat(input_ls).cuda()
        smoothing = torch.cat(smoothing_ls).cuda()
        cost = (smoothing.view(-1) * self.criterion(input.view(-1, input.shape[-1]), target.contiguous().view(-1))).sum() / norm_len
        return cost

class SequenceSmoothLoss_v2(nn.Module): #小分布平均
    def __init__(self, smooth_matrix, alpha=0.05):
        super(SequenceSmoothLoss_v2, self).__init__()
        self.smoothing = alpha
        self.smoothing_map = smooth_matrix
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to('cuda')

    def forward(self, input, gt_strs, encode):
        norm_len = input.shape[0] * input.shape[1]
        target_ls = []
        input_ls = []
        smoothing_ls = []
        for idx, gt in enumerate(gt_strs):
            text, length = encode([gt], batch_max_length=25)
            target_ls.append(text[:, 1:])
            input_ls.append(input[idx])
            if gt in self.smoothing_map.keys() and len(self.smoothing_map[gt]) > 0:
                smoothing_ls.append((1 - self.smoothing)*torch.full_like(text[:, 1:], 1.0).float())
                candidate_dict = self.smoothing_map[gt]
                sum_num = len(candidate_dict.values())
                for k,v in candidate_dict.items():
                    input_ls.append(input[idx])
                    text, _ = encode([k], batch_max_length=25)
                    target_ls.append(text[:, 1:])
                    smoothing_ls.append((1.0/sum_num) * self.smoothing*torch.full_like(text[:, 1:], 1.0).float())
            else:
                smoothing_ls.append(torch.full_like(text[:, 1:], 1.0).float())
        target = torch.cat(target_ls).cuda()
        input = torch.cat(input_ls).cuda()
        smoothing = torch.cat(smoothing_ls).cuda()
        cost = (smoothing.view(-1) * self.criterion(input.view(-1, input.shape[-1]), target.contiguous().view(-1))).sum() / norm_len
        return cost

class SequenceSmoothLoss_hard(nn.Module):
    # 通过遗忘事件
    def __init__(self, smooth_matrix, hard_map, alpha=0.05, class_num = '0123456789abcdefghijklmnopqrstuvwxyz#'):
        super(SequenceSmoothLoss_hard, self).__init__()
        '''
        num_step: Attention step
        smooth_matrix: np.array，(num_step X num_classes X num_classes) or (num_classes X num_classes)
        '''
        self.smoothing = alpha
        self.smoothing_map = smooth_matrix
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to('cuda')
        self.hard_map = hard_map
    def forward(self, input, gt_strs, encode, img_names):
        norm_len = input.shape[0] * input.shape[1]
        target_ls = []
        input_ls = []
        smoothing_ls = []
        for idx,(gt, img_name)  in enumerate(zip(gt_strs, img_names)):
            text, length = encode([gt], batch_max_length=25)
            target_ls.append(text[:, 1:])
            input_ls.append(input[idx])
            if gt in self.smoothing_map and len(self.smoothing_map[gt]) > 0 :
                smoothing_ls.append((1 - self.smoothing)*torch.full_like(text[:, 1:], 1.0).float())
                candidate_dict = self.smoothing_map[gt]
                sum_num = sum(candidate_dict.values())
                for k,v in candidate_dict.items():
                    input_ls.append(input[idx])
                    text, _ = encode([k], batch_max_length=25)
                    target_ls.append(text[:, 1:])
                    smoothing_ls.append((v/sum_num) * self.smoothing * torch.full_like(text[:, 1:], 1.0).float())

            else:
                smoothing_ls.append(torch.full_like(text[:, 1:], 1.0).float())
        target = torch.cat(target_ls).cuda()
        input = torch.cat(input_ls).cuda()
        smoothing = torch.cat(smoothing_ls).cuda()
        cost = (smoothing.view(-1) * self.criterion(input.view(-1, input.shape[-1]), target.contiguous().view(-1))).sum() / norm_len
        return cost

class SequenceSmoothLoss_hard_depend_loss(nn.Module):
    def __init__(self, smooth_matrix, alpha=0.05):
        super(SequenceSmoothLoss_hard_depend_loss, self).__init__()
        self.smoothing = alpha
        self.smoothing_map = smooth_matrix
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to('cuda')
    def forward(self, input, target, gt_strs, encode, length):
        batch_size = input.size(0)
        norm_len = length.sum().item()
        cost = self.criterion(input.view(-1, input.shape[-1]), target.contiguous().view(-1))
        cost_for_refer = cost.clone().detach()
        cost_for_refer = cost_for_refer.view(batch_size, -1).sum(-1) / length
        cost = cost.sum() / norm_len
        

        smoothings = torch.pow(cost_for_refer, torch.tensor(0.25).cuda()) * self.smoothing
        smoothings[cost_for_refer > 1] = smoothings[cost_for_refer > 1] * 2
        norm_len = 0
        target_ls = []
        input_ls = []
        smoothing_ls = []
        for idx, (gt, sm) in enumerate(zip(gt_strs, smoothings)):
            if gt in self.smoothing_map and len(self.smoothing_map[gt]) > 0 :
                #smoothing_ls.append((1 - self.smoothing)*torch.full_like(text[:, 1:], 1.0).float())
                candidate_dict = self.smoothing_map[gt]
                sum_num = sum(candidate_dict.values())
                for k,v in candidate_dict.items():
                    text, length = encode([k], batch_max_length=25)
                    input_ls.append(input[idx])
                    target_ls.append(text[:, 1:])
                    norm_len += length
                    smoothing_ls.append((v/sum_num) * sm * torch.full_like(text[:, 1:], 1.0).float())
            # else:
            #     smoothing_ls.append(torch.full_like(text[:, 1:], 1.0).float())
        target = torch.cat(target_ls).cuda()
        input = torch.cat(input_ls).cuda()
        smoothing = torch.cat(smoothing_ls).cuda()

        cost_ls = (smoothing.view(-1) * self.criterion(input.view(-1, input.shape[-1]), target.contiguous().view(-1))).sum() / norm_len


        return cost + cost_ls

