""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split(' ')
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    # RawImagePath = "raw_data/TRBC/Uncali_Conf_p10"
    # RawImagePath = "raw_data/TRBA/cali_p15"
    # RawImagePath = "raw_data/TRBA/margin_p15"
    # RawImagePath = "raw_data/TRBA/bald_p15"
    # RawImagePath = "raw_data/TRBA/random_p15"
    RawImagePath = "raw_data/TRBA/uncali_conf_p15"
    # SaveTextPath = "al_data_list_txt/TRBC_uncali_conf_iter6.txt"
    # RawImagePath = "/home/mdisk1/luoyu/Calibration/deep-text-recognition-benchmark_from_four_card/raw_data/TRBC_cali_p10"
    # RawImagePath = "/home/mdisk1/luoyu/Calibration/deep-text-recognition-benchmark_from_four_card/raw_data/TRBA_cali_p15"
    SaveTextPath = "al_data_list_txt/TRBA_uncali_conf_iter12.txt"
    f = open(SaveTextPath, 'w')
    filenames = os.listdir(RawImagePath)
    total_sample = len(filenames)
    al_iter_sample_start = int(total_sample * 0)
    al_iter_sample_end =  int(total_sample * 12/15)
    # sample_filenames = filenames[: al_iter_sample] 3
    for filename in filenames:
        label_index = os.path.splitext(filename)[0]
        index = label_index.split('-')[-1]
        if len(label_index.split('-')) > 2:
            label = '-'.join(label_index.split('-')[:-1])
        else:
            label = label_index.split('-')[0]
        if int(index) >= al_iter_sample_start and int(index) < al_iter_sample_end:
            image_path = RawImagePath + "/" + filename
            print(image_path)
            f.write(image_path + " " + label)
            f.write("\n")
    f.close()

    createDataset("",SaveTextPath ,"train_data_lmdb/TRBA_uncali_conf_iter12", checkValid=True)
    # fire.Fire(createDataset)
