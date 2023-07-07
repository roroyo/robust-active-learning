# robust-active-learning

## Prerequisites 
you can download the environmental dependencies by using
```
conda env create -f environment.yaml
```

## Training base model
```
  python train.py \
      --train_data train_data_lmdb \
      --valid_data test_data_lmdb \
      --select_data CVPR2016-NIPS2014 \
      --batch_ratio 0.5-0.5 \
      --Transformation TPS \
      --FeatureExtraction ResNet \
      --SequenceModeling BiLSTM \
      --Prediction Attn \
      --valInterva 200 \
      --lr 0.1 \
      --manualSeed 1111 \
      --alpha 0 \
      --batch_size 192 \
      --calibrator CE \
      --smooth_tail 0.1 \
      --total_data_usage_ratio 0.1 \
      --mode base \
      --patience 1000000 \
      --query_iter random_iter_0
```

## Iterative sampling
```
  python data_sampling.py \
      --train_data train_data_lmdb \
      --valid_data test_data_lmdb \
      --select_data CVPR2016-NIPS2014 \
      --batch_ratio 0.5-0.5 \
      --saved_model [path_to_the_model_of_last_round] \
      --Transformation TPS \
      --FeatureExtraction ResNet \
      --SequenceModeling BiLSTM \
      --Prediction Attn \
      --valInterva 200 \
      --lr 0.1 \
      --manualSeed 1111 \
      --alpha 0 \
      --batch_size 192 \
      --calibrator CE \
      --smooth_tail 0.1 \
      --total_data_usage_ratio 0.1 \
      --mode al \
      --query random 
```
please set the path "RawImagePath" and "SaveTextPath" for saving filtered images, and the path for saving the lmdb data of filtered images

## Iterative training
```
  python train.py \
      --train_data train_data_lmdb \
      --valid_data test_data_lmdb \
      --select_data CVPR2016-NIPS2014-[lmdb data_for_iter_training] \
      --batch_ratio 0.25-0.25-0.5 \
      --saved_model [path_to_the_model_of_last_round] \
      --Transformation TPS \
      --FeatureExtraction ResNet \
      --SequenceModeling BiLSTM \
      --Prediction Attn \
      --valInterva 200 \
      --lr 0.1 \
      --manualSeed 1111 \
      --alpha 0 \
      --batch_size 192 \
      --calibrator boost_soft \
      --smooth_tail 0.1 \
      --total_data_usage_ratio 0.1 \
      --mode al \
      --patience 1000000 \
      --query_iter random_iter_1 \
      --noise_ratio 0
```
Notice that you can choose whether to add noise to label by setting the value of noise_ratio.
