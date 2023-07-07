import json
import os
import time
import string
import argparse
import re

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance
from tools.calibration import ACE
from tools.ctc_utils import ctc_prefix_beam_search
from create_lmdb_dataset import createDataset 

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset_al import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                      'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    # # To easily compute the total accuracy of our paper.
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 
    #                   'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']

    if calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data, preds_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_data_log)
        print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
        log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
        print(dashed_line)
        log.write(dashed_line + '\n')

    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

    return None


def validation(model, criterion, al_datapool, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    preds_data = []

    unlabel_dataloader = al_datapool.get_al_batch(opt)

    for i, (image_tensors, labels, _, _, raw_images) in enumerate(unlabel_dataloader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length) #[192 27]

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTC loss format
            if opt.baiduCTC:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            if opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        
        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for raw_image, gt, pred, pred_max_prob, logit  in zip(raw_images, labels, preds_str, preds_max_prob, preds):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                except:
                    confidence_score = 0
                token_confs = pred_max_prob.tolist()

            if 'CTC' in opt.Prediction:
                hyps, confidence_score = ctc_prefix_beam_search(logit.unsqueeze(0), beam_size=1)
                confidence_score = confidence_score[0]
                pred = "".join([converter.idict[i] for i in hyps[0][0]])
                token_confs = []

            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1

            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            confidence_score_list.append(confidence_score)
            bald = compute_bald(logit)
            avg_margin = compute_margin(logit)
            # preds_data.append([confidence_score, token_confs, pred, gt, raw_image])
            preds_data.append([confidence_score, token_confs, bald, avg_margin, pred, gt, raw_image])

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return preds_data

def compute_margin(logit):
    probs = F.softmax(logit, dim=1).cpu().numpy()
    T = probs.shape[0]
    margin = []
    for t in range(T):
        p_t = probs[t]
        p_sort = p_t[np.argsort(p_t)]
        m = p_sort[-1] - p_sort[-2]
        margin.append(m)

    avg_margin = np.mean(margin)

    return avg_margin

def compute_bald(logit):
    probs = F.softmax(logit, dim=1).cpu().numpy()

    entropy = -np.mean(np.sum(probs  * np.log(probs + 1e-16), axis=1), axis=0)
    avg_preds = np.mean(probs, axis=1)
    avg_entropy = -np.sum(avg_preds * np.log(avg_preds + 1e-16), axis=0)
    bald = avg_entropy - entropy

    return bald

def query_data(opt, preds_data, al_iter_sample_start, al_iter_sample_end):

    if opt.query == "uncali":
        sorted_data = sorted(preds_data, key=lambda x: x[0])           
        sampled_data = sorted_data[al_iter_sample_start : al_iter_sample_end]
    elif opt.query == "random":
        import random
        num_sample = al_iter_sample_start - al_iter_sample_end
        sampled_data = random.sample(preds_data, num_sample)
    elif opt.query == "bald":
        sorted_data = sorted(preds_data, key=lambda x: x[2])           
        sampled_data = sorted_data[al_iter_sample_start : al_iter_sample_end]
    elif opt.query == "margin":
        sorted_data = sorted(preds_data, key=lambda x: x[3])           
        sampled_data = sorted_data[al_iter_sample_start : al_iter_sample_end]
    elif opt.query == "cali":
        sorted_data = sorted(preds_data, key=lambda x: x[0])   
        num_sample = al_iter_sample_start - al_iter_sample_end
        for i in range(len(sorted_data)):
            if sorted_data[i][0] > 0.5:
                print(i, sorted_data[i][0])
                start = i
                break            
        sampled_data = sorted_data[start : start + num_sample]
    else:
        print("There is no such query strategy !!!")
    
    return sampled_data

def test(opt):

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    al_datapool = Batch_Balanced_Dataset(opt)

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    sample_ratio = opt.ratio
    with torch.no_grad():

        preds_data = validation(model, criterion, al_datapool, converter, opt)
        num_data = len(preds_data)
        num_sample = int(num_data * sample_ratio) 
        iter_num = 1
        al_iter_sample_start = int((iter_num-1) * num_sample)
        al_iter_sample_end =  al_iter_sample_start + num_sample

        sampled_data = query_data(opt, preds_data, al_iter_sample_start, al_iter_sample_end )

        return sampled_data, al_datapool 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--with_file_name', action='store_true')
    parser.add_argument('--with_vis', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.01, help='ratio for sampling unlabel data')
    parser.add_argument('--query',  type=str, required=True, help='whether to train the initial model')
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='0.3',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--calibrator', type=str, default='FL_MDCA')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--mode',  type=str, required=True, help='whether to train the initial model')
    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.language}/{opt.exp_name}-{opt.mode}/{opt.calibrator}/{opt.alpha}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    if opt.language == 'en':
        with open('charset_36.txt','r') as f:
            opt.character = sorted(f.read())
    elif opt.language == 'zh':
        with open('benchmark_alphabet.txt','r') as f:
            opt.character = sorted(f.read())
    else:
        raise RuntimeError("language should be zh/en, not {}".format(opt.language))
        
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    RawImagePath = "raw_data/TRBA/uncali_conf_p15"
    SaveTextPath = "al_data_list_txt/TRBA_uncali_conf_iter12.txt"

    # test(opt)
    sampled_data, al_datapool = test(opt)

    for i in range(len(sampled_data)):
        img = sampled_data[i][-1]

        img_name = RawImagePath + f"/{sampled_data[i][-2]}-{i}" + ".jpg"

        print(i, img_name)
        # cv2.imwrite(img_name, img)
        img.save(img_name)

    # RawImagePath = "raw_data/TRBA/uncali_conf_p15"
    # SaveTextPath = "al_data_list_txt/TRBC_uncali_conf_iter6.txt"
    # RawImagePath = "/home/mdisk1/luoyu/Calibration/deep-text-recognition-benchmark_from_four_card/raw_data/TRBC_cali_p10"
    # RawImagePath = "/home/mdisk1/luoyu/Calibration/deep-text-recognition-benchmark_from_four_card/raw_data/TRBA_cali_p15"
    f = open(SaveTextPath, 'w')
    filenames = os.listdir(RawImagePath)
    # total_sample = len(filenames)
    # al_iter_sample_start = int(total_sample * 0)
    # al_iter_sample_end =  int(total_sample * 12/15)
    # sample_filenames = filenames[: al_iter_sample] 
    for filename in filenames:
        label_index = os.path.splitext(filename)[0]
        index = label_index.split('-')[-1]
        if len(label_index.split('-')) > 2:
            label = '-'.join(label_index.split('-')[:-1])
        else:
            label = label_index.split('-')[0]
        # if int(index) >= al_iter_sample_start and int(index) < al_iter_sample_end:
        image_path = RawImagePath + "/" + filename
        print(image_path)
        f.write(image_path + " " + label)
        f.write("\n")
    f.close()

    createDataset("",SaveTextPath ,"train_data_lmdb/TRBA_uncali_conf_iter12", checkValid=True)