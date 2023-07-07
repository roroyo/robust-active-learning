import torch
from torch import nn
from .loss import FLSD, BrierScore, CaslsChineseAttnLoss, EntropyRegularAttn, GraduatedLabelSmoothingAttn, PairWiseWeightSmoothLoss, SequenceSmoothLoss_v10, CrossEntropyLoss, SequenceSmoothLoss_v11, LabelSmoothingLoss, LogitMarginL1, ClassficationAndMDCA, FocalLoss, SequenceSmoothLoss_v12, SequenceSmoothLoss_v13, CASLSRobustSmoothLoss,CASLSRobustSmoothLoss_v2, robust,robust_wog, robust_wol,boost_soft, boost_hard
from .loss_ctc import CTC_Loss, CTCClassficationAndMDCA, CTCEntropyRegular, CTCGraduatedLabelSmoothing, CTCLogitMarginL1, SequenceSmoothLossCtc_ablation, SequenceSmoothLossCtc_v10, CTCLabelSmoothLoss, SequenceSmoothLossCtc_v11,CTCCASLSRobustSmoothLoss, CTCrobust
LOSS_FUNC={
    'CE':CrossEntropyLoss,
    'LS':LabelSmoothingLoss,
    'SeqLSv1_0':SequenceSmoothLoss_v10,
    'SeqLSv1_1':SequenceSmoothLoss_v11,
    'SeqLSv1_2':SequenceSmoothLoss_v12,
    'SeqLS_ab':SequenceSmoothLoss_v13,
    'MBLS':LogitMarginL1,
    'MDCA':ClassficationAndMDCA,
    'FL':FocalLoss,
    'FLSD':FLSD,
    'ER':EntropyRegularAttn,
    'BS':BrierScore,
    'GLS':GraduatedLabelSmoothingAttn,
    'CASLS-EN':PairWiseWeightSmoothLoss,
    'CASLS-ZH':CaslsChineseAttnLoss,
    'robust':robust,
    'robust_wog':robust_wog,
    'robust_wol':robust_wol,
    'boost_soft':boost_soft,
    'boost_hard':boost_hard,
    'CASLS-robust':CASLSRobustSmoothLoss_v2,
}

LOSS_FUNC_CTC={
    'CTC':CTC_Loss,
    'CTCLS':CTCLabelSmoothLoss,
    'SeqLSv1_0_ctc':SequenceSmoothLossCtc_v10,
    'MDCA':CTCClassficationAndMDCA,
    'MBLS':CTCLogitMarginL1,
    'ER':CTCEntropyRegular,
    'robust':CTCrobust,
    'GLS':CTCGraduatedLabelSmoothing,
    'SeqLSv1_1_ctc':SequenceSmoothLossCtc_v11,
    'SeqLS_ctc_ab':SequenceSmoothLossCtc_ablation,
    'CTCCASLS-robust':CTCCASLSRobustSmoothLoss,
}