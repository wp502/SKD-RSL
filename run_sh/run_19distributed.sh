#!/bin/bash

basicBasic='train_ours_ddp.py --model_s ResNet18_cifar_another --model_t ResNet34_cifar_another --path_t ./save/base/cifar100/ResNet34_cifar_another_base/ResNet34_cifar_another_teacher_best.pth.tar --method 1 --feedback_time 0 --learning_rate 0.1 --lr_decay_rate 0.1 --aux_method_t Basic --aux_method_s Basic --self_method_t bi_directional --self_method_s bi_directional'

#multi-node/single-node multi-GPU
python $basicBasic --epochs 200 --weight_decay 5e-4 --lr_decay_epochs 60,120,160 --loss2csv -p 100 --dataset cifar100 --num_class 100 --batch_size 128 --dist-url 'tcp://127.0.0.1:1257' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall --seed 2022403 -t 19-01_43_dist

python $basicBasic --epochs 200 --weight_decay 5e-4 --lr_decay_epochs 60,120,160 --loss2csv -p 100 --dataset cifar100 --num_class 100 --batch_size 256 --dist-url 'tcp://127.0.0.1:1257' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall --seed 2022403 -t 19-01_45_dist

#Single GPU
python $basicBasic --epochs 200 --weight_decay 5e-4 --lr_decay_epochs 60,120,160 --loss2csv -p 100 --dataset cifar100 --num_class 100 --batch_size 128 --in_criterion KL_softmax --blocks_amount_t 4 --blocks_amount_s 4 --infwd_loss 1 --learning_rate_t 5e-3 --scheduler_method MultiStepLR --fusion_method_AE_t ADPAELinear --fusion_method_AE_s ADPAELinear --fusion_size littleSmall --seed 2022403 -t t_name --gpu [id]

#想要选定哪些卡跑，只需要在前面加 CUDA_VISIBLE_DEVICES=idx1,idx2

# test in IMAGENET
#python train_ours_ddp.py --model_s ResNet18_another --model_t ResNet34_another --path_t ./save/base/imagenet/ResNet34_another_imagenet_base/ResNet34_another_teacher_best.pth.tar -p 100 --dist-url 'tcp://127.0.0.1:1257' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --dataset_dir /data1/mazc/gjp/xxm/NKD/data/imagenet/ -t test

