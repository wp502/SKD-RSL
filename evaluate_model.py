from json.tool import main
import utils
import argparse
import logging
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import data_loader
from models import model_dict
import train_one_epoch
from zoo import DistillKL, CCSLoss, ICKDLoss, SPKDLoss, CRDLoss, KDLossv2, CDLoss
from tensorboardX import SummaryWriter
from dataloader import fetch_dataloader_1 as fetch_dataloader
import tensorboard_logger


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')

    parser.add_argument('--seed', type=int, default=2022403, help='random seed')

    # model
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--model_pth', type=str, default=None, help='evaluate model snapshot')
    parser.add_argument('--distill', type=str, default='evaluate')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10', 'tiny_imagenet', 'cub200', 'cars196', 'imagenet', 'caltech256',
                                 'food101'],
                        help='dataset')
    parser.add_argument('--dataset_test_percent', type=float,
                        default=0.25, help='[for Caltech256] test dataset percent')
    parser.add_argument('--pin_memory', action='store_false', default=True,
                        help="flag for whether use pin_memory in dataloader")
    parser.add_argument('--dataset_dir', type=str, default=None, help='whether appoint dataset dir path')
    parser.add_argument('--num_class', default=100,
                        type=int, help="number of classes")
    parser.add_argument('--augmentation', type=str,
                        default='yes', help='dataset augmentation')
    parser.add_argument('--subset_percent', type=float,
                        default=1.0, help='subset_percent')
    parser.add_argument('--metric_method', type=str, default='acc', help='metric method')
    parser.add_argument('-t', '--trial', type=str,
                        default='test', help='the experiment id')

    args = parser.parse_args()

    args.record_txt_folder = './save/accToRecall/'
    if not os.path.isdir(args.record_txt_folder):
        os.makedirs(args.record_txt_folder)
    args.record_txt_name = str(args.trial) + '.txt'
    args.record_txt_pth = os.path.join(args.record_txt_folder, args.record_txt_name)

    return args


def load_teacher(model_path, n_cls, model_t):
    print('==> loading model')
    # model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls).cuda()
    # model.load_state_dict(torch.load(model_path)['state_dict'])  # ['state_dict']
    model.load_state_dict(torch.load(model_path)['state_dict'])  # ['state_dict']
    print('==> done')
    return model


def main():
    args = parse_option()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    warnings.filterwarnings("ignore")

    model_pth = args.model_pth
    # model_pth_split = model_pth.split('/')[-1]
    model_pth_split = model_pth.split('/', 9)[-1]
    # method = model_pth.split('/')[-3]

    url_start = "==>Evaluate Model:" + args.model + "\tPath:" + model_pth_split + "\tDataset:" + args.dataset + "\tMetric:" + args.metric_method
    print(url_start)
    # set_logger
    # logger_name = args.model_name + ".log"
    # utils.set_logger(os.path.join(args.save_folder, logger_name))

    # dataloader
    # logging.info("Loading the datasets...")
    print("==>Loading the datasets...")

    # fetch dataloaders, considering full-set vs. sub-set scenarios
    # if args.subset_percent < 1.0:
    #     train_dl = data_loader.fetch_subset_dataloader('train', args)
    # else:
    #     train_dl = data_loader.fetch_dataloader('train', args)

    train_dl = fetch_dataloader('train', args)
    # train_dl = data_loader.fetch_dataloader('train', args)

    dev_dl = fetch_dataloader('dev', args)
    # dev_dl = data_loader.fetch_dataloader('dev', args)

    # logging.info("- done.")
    print("==>- done.")

    # tb_logger = SummaryWriter(log_dir=args.tb_folder)

    # load model
    # logging.info('Distill Method:' + str(args.distill) + '_T:' + str(args.model_t) + '-S:' + str(
    #     args.model_s) + '-Dataset:' + str(args.dataset) + '-Trial:' + str(args.trial))
    # logging.info('==> student model: ' + args.model_s)
    # logging.info('==> loading teacher model: ' + args.model_t)
    model = load_teacher(args.model_pth, args.num_class, args.model).cuda()
    # logging.info('==> done')
    # model_s = model_dict[args.model_s](num_classes=args.num_class).cuda()

    criterion_cls = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True

    # validate teacher accuracy
    # teacher_acc, _, _ = train_one_epoch.evaluate(dev_dl, model_t, criterion_cls, args)
    val_metrics = train_one_epoch.evaluate_2(model, criterion_cls, dev_dl, args, metric_method=args.metric_method)
    # logging.info('teacher accuracy: ' + str(val_metrics['top1']))
    # print('teacher accuracy ' + str(args.metric_method) + '@1: ' + str(val_metrics['top1'].item()) + '\t' + str(args.metric_method) + '@5: ' + str(val_metrics['top5'].item()))
    url_output = 'model accuracy ' + str(args.metric_method) + '@1_custom: ' + str(val_metrics['top1_custom']) + '\nmodel accuracy ' + str(args.metric_method) + '@1_api: ' + str(100.0 * val_metrics['top1_api'])
    print(url_output)

    with open(args.record_txt_pth, 'a') as f:
        f.write(url_start + '\n')
        f.write(url_output + '\n')
        f.write('\n')


if __name__ == '__main__':
    main()
