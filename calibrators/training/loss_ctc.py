from multiprocessing import reduction
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dtw import dtw
from .exceptions import ParamException

from tools.ctc_utils import ctc_prefix_beam_search, get_op_seq

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self, output, target):
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

class CTCClassficationAndMDCA(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0, ignore_index=0, **kwargs):
        super(CTCClassficationAndMDCA, self).__init__()
        self.beta = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.classification_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.MDCA = MDCA()

    def forward(self, preds, text, preds_size, length, *args):
        preds_cls = preds.log_softmax(2).permute(1, 0, 2)
        loss_cls = self.classification_loss(preds_cls, text, preds_size, length)

        inputs, targets  = preds.view(-1, preds.shape[-1]), text.contiguous().view(-1)
        loss_cal = self.MDCA(inputs, targets)
        return loss_cls + self.beta * loss_cal


class CTC_Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, *args):
        preds = preds.log_softmax(2).permute(1, 0, 2)
        loss = self.loss_func(preds, text, preds_size, length)
        return loss

class CTCEntropyRegular(nn.Module):
    def __init__(self, alpha=0.1, **kwargs):
        super(CTCEntropyRegular, self).__init__()
        self.beta = alpha
        self.ctc = nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, *args):
        preds_cls = preds.log_softmax(2).permute(1, 0, 2)
        ctc_loss = self.ctc(preds_cls, text, preds_size, length)

        total_input = preds.view(-1, preds.shape[-1])
        entrop_all = -(total_input.softmax(1)*total_input.log_softmax(1)).sum() / total_input.shape[0]
        loss = ctc_loss - self.beta*entrop_all
        return loss

class CTCLabelSmoothLoss(nn.Module):
    def __init__(self, alpha=0.0, **kwargs):
        super(CTCLabelSmoothLoss, self).__init__()
        self.smoothing = alpha
        self.ctc = nn.CTCLoss(zero_infinity=True)
        self.kldiv = nn.KLDivLoss()

    def forward(self, preds, text, preds_size, length, *args):
        preds = preds.log_softmax(2).permute(1, 0, 2)
        ctc_loss = self.ctc(preds, text, preds_size, length)
        kl_inp = preds.transpose(0, 1)
        uni_distri = torch.full_like(kl_inp, 1 / preds.shape[-1])
        kldiv_loss = self.kldiv(kl_inp, uni_distri)
        loss = (1 - self.smoothing) * ctc_loss + self.smoothing * kldiv_loss
        return loss

class CTCrobust(nn.Module):
    def __init__(self, alpha=0.0, num_iter= 300000, **kwargs):
        super(CTCrobust, self).__init__()
        self.smoothing = alpha
        self.ctc = nn.CTCLoss(zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction="none")

        self.total_iterations = num_iter
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
            class_num = pred_probs.shape[2]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 2
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform
            #avg T
            example_trust = example_trust.mean(0)
            # the trade-off
            self.epsilon = global_trust * example_trust
            # from shape [N] to shape [N, 1]
            # self.epsilon = self.epsilon[:, None]

    def forward(self, preds, text, preds_size, length, cur_time, *args):

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
        preds = preds.log_softmax(2).permute(1, 0, 2)
        preds_prob = F.softmax(preds, dim=1)
        self.update_epsilon_progressive_adaptive(preds_prob, cur_time) #26 192

        ctc_loss = self.ctc(preds, text, preds_size, length)
        kl_inp = preds.transpose(0, 1)  #192 26 37
        uni_distri = torch.full_like(kl_inp, 1 / preds.shape[-1])
        kldiv_loss = 0
        for i in range(B):
            kldiv_loss_i = self.kldiv(kl_inp[i,:,:], uni_distri[i,:,:]).mean() * self.epsilon[i]        
            kldiv_loss += kldiv_loss_i

        kldiv_loss = kldiv_loss / B  
        # kldiv_loss = self.kldiv(kl_inp[i,:,:], uni_distri[i,:,:])
        # loss = (1 - self.epsilon) * ctc_loss + self.epsilon * kldiv_loss
        loss = ctc_loss + kldiv_loss
        return loss

class CTCLogitMarginL1(nn.Module):
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
        self.ctc = nn.CTCLoss(zero_infinity=True)

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

    def forward(self, input, target, input_length, target_length, *args):
        input_ctc = input.log_softmax(2).permute(1, 0, 2)
        loss_ctc = self.ctc(input_ctc, target, input_length, target_length)
        
        input = input.view(-1, input.shape[-1])
        diff = self.get_diff(input)

        loss_margin = F.relu(diff-self.margin).mean()
        loss = loss_ctc + self.alpha * loss_margin
        return loss

class CTCGraduatedLabelSmoothing(nn.Module):
  def __init__(self, 
                alpha=0.0,
                sequence_normalize=True,
                **kwargs):
    super(CTCGraduatedLabelSmoothing, self).__init__()
    self.criterion = nn.KLDivLoss(reduction="none")
    self.confidence = 1.0 - alpha
    self.smoothing = alpha
    self.ctc = nn.CTCLoss(zero_infinity=True)
    self.normalize_length = sequence_normalize
  def forward(self, preds, text, preds_size, length, *args):
    preds_cls = preds.log_softmax(2).permute(1, 0, 2)
    ctc_loss = self.ctc(preds_cls, text, preds_size, length)

    size = preds.size(-1)
    preds = preds.view(-1, size)
    pred_probability, _ = torch.softmax(preds, dim=1).max(1)
    smoothing = self.smoothing * torch.ones_like(preds[:, 0])
    smoothing[pred_probability >= 0.7] = 3*self.smoothing
    smoothing[pred_probability <= 0.3] = 0.0
    smoothing.unsqueeze_(1)
    uni_distri = torch.full_like(preds, 1 / size)
    kl = self.criterion(torch.log_softmax(preds, dim=1), uni_distri)
    return (1 - smoothing).mean() * ctc_loss + (smoothing * kl).mean()


class SequenceSmoothLossCtc_v10(nn.Module):
    '''
        只把vis和semantic的str拿出来, 然后 loss = (1 - self.alpha) * loss_master + self.alpha * loss_smooth,
        没有用到conf, 而且每个str的损失是一致对待
    '''
    def __init__(self, 
                 converter,
                 semantic,
                 alpha=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ], [0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, visual, labels, *args):
        time_length, size = preds.shape[1:]

        preds_master = preds.log_softmax(2).permute(1, 0, 2)
        loss_master = self.loss_func(preds_master, text, preds_size, length)

        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]
        text, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])

        text, length = torch.cat(text, dim=0), torch.cat(length, dim=0)
        preds = preds.unsqueeze(1).repeat(1, text.shape[0] // preds.shape[0] , 1, 1).view(-1, time_length, size)
        preds_size = torch.IntTensor([time_length] * preds.size(0))

        preds_smooth = preds.log_softmax(2).permute(1, 0, 2)
        loss_smooth = self.loss_func(preds_smooth, text, preds_size, length)
        loss = (1 - self.alpha) * loss_master + self.alpha * loss_smooth
        return loss


class SequenceSmoothLossCtc_v11(nn.Module):
    '''
        只把vis和semantic的str拿出来, 然后 loss = (1 - self.alpha) * loss_master + self.alpha * loss_smooth,
        没有用到conf, 而且每个str的损失是一致对待
    '''
    def __init__(self, 
                 converter,
                 semantic,
                 alpha=0.0,
                 smooth_tail=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ], [0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.smooth_tail = smooth_tail
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, visual, labels, *args):
        with torch.no_grad():
            confidence_score_list = []
            for logit in preds:
                _, confidence_score = ctc_prefix_beam_search(logit.unsqueeze(0), beam_size=1)
                confidence_score_list.append(confidence_score[0])


        time_length, size = preds.shape[1:]

        preds_master = preds.log_softmax(2).permute(1, 0, 2)
        loss_master = self.loss_func(preds_master, text, preds_size, length)

        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]
        text, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])
        preds = preds.unsqueeze(1).repeat(1, text[0].shape[0], 1, 1)

        loss_cal = []
        for pred, tx, lth in zip(preds, text, length):
            preds_size = torch.IntTensor([time_length] * pred.size(0))
            pred = pred.log_softmax(2).permute(1, 0, 2)
            loss_smooth = self.loss_func(pred, tx, preds_size, lth)
            loss_cal.append(loss_smooth.unsqueeze(0))

        loss_cal = torch.cat(loss_cal)

        confi_list = torch.tensor(confidence_score_list, device='cuda')
        #ranking = torch.pow((1 - confi_list), 2)

        #ranking = torch.ones_like(confi_list)
        ranking = self.smooth_tail + (1.0 - self.smooth_tail) * torch.pow((1 - confi_list), 2)

        loss = loss_master + (self.alpha * ranking * loss_cal).mean()
        return loss


class SequenceSmoothLossCtc_ablation(nn.Module):
    '''
        只把vis和semantic的str拿出来, 然后 loss = (1 - self.alpha) * loss_master + self.alpha * loss_smooth,
        没有用到conf, 而且每个str的损失是一致对待
    '''
    def __init__(self, 
                 converter,
                 semantic,
                 alpha=0.0,
                 smooth_tail=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ], [0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.smooth_tail = smooth_tail
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, visual, labels, *args):
        with torch.no_grad():
            confidence_score_list = []
            for logit in preds:
                _, confidence_score = ctc_prefix_beam_search(logit.unsqueeze(0), beam_size=1)
                confidence_score_list.append(confidence_score[0])


        time_length, size = preds.shape[1:]

        preds_master = preds.log_softmax(2).permute(1, 0, 2)
        loss_master = self.loss_func(preds_master, text, preds_size, length)

        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]
        text, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])
        preds = preds.unsqueeze(1).repeat(1, text[0].shape[0], 1, 1)

        loss_cal = []
        for pred, tx, lth in zip(preds, text, length):
            preds_size = torch.IntTensor([time_length] * pred.size(0))
            pred = pred.log_softmax(2).permute(1, 0, 2)
            loss_smooth = self.loss_func(pred, tx, preds_size, lth)
            loss_cal.append(loss_smooth.unsqueeze(0))

        loss_cal = torch.cat(loss_cal)

        confi_list = torch.tensor(confidence_score_list, device='cuda')
        #ranking = torch.pow((1 - confi_list), 2)

        #ranking = torch.ones_like(confi_list)
        #ranking = self.smooth_tail + (1.0 - self.smooth_tail) * torch.pow((1 - confi_list), 1/2)
        ranking = torch.ones_like(confi_list)

        loss = loss_master + (self.alpha * ranking * loss_cal).mean()
        return loss

class CTCCASLSRobustSmoothLoss(nn.Module):
    def __init__(self, matric, converter, blank=0, alpha=0.05, **kwargs):
        super(CTCCASLSRobustSmoothLoss, self).__init__()
        self.smoothing = alpha
        self.matric = np.zeros((38,38,38))
        self.matric[0] = np.eye(38)
        self.matric[:,0,0] = 1
        self.matric[1:,1:,1:] = matric
        self.smooth_matrix = matric
        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(size_average=True, reduce=False)
        self.converter = converter

    def forward(self, input, target, input_length, target_length, _, labels,*args):
        '''
        input: B T C
        target: T S
        input_length: T
        target_length: T
        '''
        _, preds_index = input.max(2)
        nclass = input.shape[-1]
        ctc_input = input.log_softmax(2).permute(1, 0, 2)
        ctc_loss = self.ctc(ctc_input, target, input_length, target_length)

        kl_inp = input
        B, T, C = kl_inp.size()  # 192 26 37
        # step_matric = np.zeros((B, T, C))

        kldiv_loss = 0
        
        preds_str,input_pos = self.converter.decode_add_pos(preds_index.data, input_length.data)

        for i in range(B):
            op_str = get_op_seq(preds_str[i], labels[i])
            label, pred_str, op_str = list(map(list, [labels[i], preds_str[i], op_str]))

            sample_input = kl_inp[i, :, :]  # 26 37(ignore the separation blank class)
            sample_input = sample_input.log_softmax(1)  # softmax  -> linear transformation?
            # i_sum = sample_input.sum(1).unsqueeze(1).expand(26,37)
            # sample_input = sample_input / i_sum

            # sample_length = target_length[i]
            sample_pos = input_pos[i]  # target_length (for selecting the probability of the corresponding pos
            selected_pred = sample_input[sample_pos]
            GT_target = target[i, :target_length[i]].data.cpu().numpy().astype(int).tolist()

            # pred_align = selected_pred
            pred_align = torch.zeros([len(op_str), (C-1)])


            align_index = []
            m = 0

            for j, op in enumerate(op_str):
                if op == '#':
                    align_index.append(selected_pred[None, m, :])
                    m = m + 1
                elif op == 's':
                    align_index.append(selected_pred[None, m, :])
                    m = m + 1
                elif op == 'i':
                    align_index.append(torch.full([1, C], 1 / nclass).log_softmax(1).cuda())
                elif op == 'd':
                    align_index.append(selected_pred[None, m, :])
                    GT_target.insert(j,0)
                    m = m + 1
            try:
                pred_align = torch.cat(align_index, 0)
            except:
                continue
            
            forth_target = [C-1] * len(GT_target)
            forth_target[1:] = GT_target[:-1]

            if len(pred_align):
                # for j in range(T):
                smoothing = 1 - math.pow((1 - self.smoothing), 1 / len(pred_align))
                step_matric = self.matric[forth_target, GT_target]

                SLS_distri = torch.from_numpy(step_matric).float().cuda() #(3,38)
                eps = SLS_distri.new_ones(SLS_distri.size()).float().cuda() * (1e-10)
                kldiv_loss += smoothing * (SLS_distri.sum(1) * self.kldiv((pred_align), (SLS_distri + eps)).mean(1)).mean()

        loss = ctc_loss + kldiv_loss
        # loss = (1 - self.smoothing) * ctc_loss + self.smoothing * kldiv_loss

        return loss