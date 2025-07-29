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
from tensorboardX import SummaryWriter
from dataloader import fetch_dataloader_1 as fetch_dataloader
import tensorboard_logger
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import time
import datetime


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float,
                        default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='60,120,160', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--seed', type=int, default=2022403, help='random seed')
    parser.add_argument('--scheduler_method', type=str, default='MultiStepLR')
    parser.add_argument('--distill', type=str, default='base')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10', 'tiny_imagenet', 'cub200', 'cars196', 'imagenet', 'caltech256',
                                 'food101'],
                        help='dataset')
    parser.add_argument('--dataset_lmdb', action='store_true', default=False,
                        help="flag for whether use lmdb to read dataset, especially for ImageNet")
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

    # For Resume Training
    parser.add_argument('--is_resume', action='store_true', default=False, help="flag for whether Resume training")
    parser.add_argument('--resume_checkpoint', default='./save/distill/', help="student model resume checkpoint path")

    parser.add_argument('-t', '--trial', type=str,
                        default='test', help='the experiment id')
    parser.add_argument('--is_checkpoint', action='store_true', default=False, help="flag for whether save checkpoint")
    parser.add_argument('--save_checkpoint_amount', default=50,
                        type=int, help="number of epoch checkpoint to save")
    parser.add_argument('--metric_method', type=str, default='acc', help='metric method')

    args = parser.parse_args()

    args.model_path = './save/base'
    args.model_path += '/' + str(args.dataset)
    # args.tb_path = os.path.join(args.model_path, 'tensorboard')

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(args.model, args.dataset, args.learning_rate,
                                                             args.weight_decay, args.trial)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    # args.save_folder = args.save_folder.replace('\\', '/')  # WINDOWS
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.is_checkpoint:
        args.checkpoint_save_pth = os.path.join(args.save_folder, 'checkpoint')
        if not os.path.isdir(args.checkpoint_save_pth):
            os.makedirs(args.checkpoint_save_pth)

    # args.tb_folder = os.path.join(args.tb_path, args.model_name)
    args.tb_folder = os.path.join(args.save_folder, str(args.trial) + '_tensorboard')
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    return args


def select_scheduler(optimizer, args):
    if args.scheduler_method == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    elif args.scheduler_method == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    return scheduler

def select_scheduler_resume(optimizer, args, start_epoch):
    if args.scheduler_method == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate, last_epoch=start_epoch)
    elif args.scheduler_method == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=start_epoch)

    return scheduler

def resume_model(model, optimizer, args):
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_dict'])
    start_epoch = checkpoint['epoch'] + 1

    return model, optimizer, start_epoch


def main():
    start_time = time.time()
    args = parse_option()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    warnings.filterwarnings("ignore")

    # set_logger
    logger_name = args.model_name + ".log"
    utils.set_logger(os.path.join(args.save_folder, logger_name))

    # dataloader
    logging.info("Loading the datasets...")

    # fetch dataloaders, considering full-set vs. sub-set scenarios
    # if args.subset_percent < 1.0:
    #     train_dl = data_loader.fetch_subset_dataloader('train', args)
    # else:
    #     train_dl = data_loader.fetch_dataloader('train', args)

    train_dl = fetch_dataloader('train', args)
    # train_dl = data_loader.fetch_dataloader('train', args)

    dev_dl = fetch_dataloader('dev', args)
    # dev_dl = data_loader.fetch_dataloader('dev', args)

    logging.info("- done.")

    # model
    model = model_dict[args.model](num_classes=args.num_class).cuda()
    utils.statis_params_amount(model)
    logging.info("==> train teacher model, teacher model:" + str(args.model) + " --Dataset:" + str(
        args.dataset) + " Trial:" + str(args.trial))

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if args.is_resume:
        logging.info('==> Resume Training')
        logging.info('==> Resume Base Model: ' + str(args.model))
        logging.info('==> Resume From: ' + str(args.resume_checkpoint))
        model, optimizer, start_epoch = resume_model(model, optimizer, args)
    else:
        start_epoch = 0

    if args.is_resume:
        scheduler = select_scheduler_resume(optimizer, args, start_epoch)
    else:
        scheduler = select_scheduler(optimizer, args)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True

    # tb_logger = tensorboard_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    tb_logger = SummaryWriter(log_dir=args.tb_folder)

    best_val_acc_top1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        # utils.adjust_learning_rate(epoch, args, optimizer)
        # scheduler.step()
        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1,
                                                 args.epochs, optimizer.param_groups[0]['lr']))

        logging.info("==> training...")

        train_acc, train_loss = train_one_epoch.train_vanilla(
            model, optimizer, criterion, train_dl, scheduler, epoch, args)

        # Evaluate for one epoch on validation set
        val_metrics = train_one_epoch.evaluate(model, criterion, dev_dl, args, metric_method=args.metric_method)

        val_metrics['epoch'] = epoch + 1
        val_acc_top1 = val_metrics['top1']
        val_acc_top5 = val_metrics['top5']
        is_best = val_acc_top1 >= best_val_acc_top1

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'acc': val_acc_top1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              args=args,
                              save_folder=args.save_folder,
                              is_teacher=True,
                              name=args.model)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc_top1 = val_acc_top1

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                args.save_folder, "eval_best_results.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            args.save_folder, "eval_last_results.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # Tensorboard
        tb_logger.add_scalar('Train_accuracy', train_acc, epoch)
        tb_logger.add_scalar('Train_loss', train_loss, epoch)
        tb_logger.add_scalar('Test_accuracy_top1', val_metrics['top1'], epoch)
        tb_logger.add_scalar('Test_accuracy_top5', val_metrics['top5'], epoch)
        tb_logger.add_scalar('Test_loss', val_metrics['loss'], epoch)

    tb_logger.close()
    logging.info('best accuracy top1: ' + str(best_val_acc_top1))

    end_time = time.time()
    time_consuming = end_time - start_time
    time_consuming = datetime.timedelta(seconds=time_consuming)
    logging.info('Dataset:\t' + str(args.dataset) + '\tTrial:\t' + str(args.trial))
    logging.info('Time Consuming:\t' + str(time_consuming))


if __name__ == '__main__':
    main()
