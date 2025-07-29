from json.tool import main
from pickletools import optimize
from unicodedata import name
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
from zoo import DistillKL, DMLLoss, KDCLLoss, SOKDLoss, Fusion_module, KLLoss, auxiliary_forward, kl_div, dist_s_label, dist_s_t
from tensorboardX import SummaryWriter
from dataloader import fetch_dataloader_1 as fetch_dataloader
import tensorboard_logger
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
    # SOKD's auxiliary lr 
    parser.add_argument('--lr_sokd_aux', type=float,
                        default=0.1, help='SOKD auxiliary lr')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='60,120,160', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--seed', type=int, default=2022403, help='random seed')

    # model
    # --models_list ResNet18 ResNet18
    parser.add_argument('--models_list', type=str, nargs='+', default=['ResNet18', 'ResNet18'], help="MUST be T S")
    parser.add_argument('--distill', type=str, default='kd', choices=['dml', 'kdcl', 'sokd', 'ffl', 'switokd'])
    # SOKD
    parser.add_argument('--model_t', type=str, default=None, help='[FOR SOKD]ResNet18_sokd')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # hyper-parameters
    parser.add_argument('-a', '--alpha', type=float, default=1, help='weight balance for CE')
    parser.add_argument('-b', '--beta', type=float, default=1, help='weight balance for KD')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight balance for other losses')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    # SOKD's parameters
    parser.add_argument('--sokd_aux_t', type=float, default=1, help='weight balance for SOKD auxiliary_lambda_kd_t')
    parser.add_argument('--sokd_aux_s', type=float, default=1, help='weight balance for SOKD auxiliary_lambda_kd_s')
    parser.add_argument('--sokd_kd', type=float, default=1, help='weight balance for SOKD lambda_kd')

    parser.add_argument('--is_resume', action='store_true', default=False, help="flag for whether Resume training")
    parser.add_argument('--checkpoint_1', default='./save/distill_online/', help="teacher model resume checkpoint path")
    parser.add_argument('--checkpoint_2', default='./save/distill_online/', help="student model resume checkpoint path")

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10', 'tiny_imagenet', 'cub200', 'cars196', 'imagenet', 'caltech256',
                                 'food101'],
                        help='dataset')
    parser.add_argument('--dataset_lmdb', action='store_true', default=False,
                        help="flag for whether use lmdb to read dataset, especially for ImageNet")
    parser.add_argument('--dataset_test_percent', type=float,
                        default=0.25, help='[for Caltech256] test dataset percent')
    parser.add_argument('--dataset_dir', type=str, default=None, help='whether appoint dataset dir path')
    parser.add_argument('--num_class', default=100,
                        type=int, help="number of classes")
    parser.add_argument('--augmentation', type=str,
                        default='yes', help='dataset augmentation')
    parser.add_argument('--subset_percent', type=float,
                        default=1.0, help='subset_percent')
    parser.add_argument('--pin_memory', action='store_false', default=True,
                        help="flag for whether use pin_memory in dataloader")

    parser.add_argument('-t', '--trial', type=str,
                        default='test', help='the experiment id')
    parser.add_argument('--is_checkpoint', action='store_true', default=False, help="flag for whether save checkpoint")
    parser.add_argument('--save_checkpoint_amount', default=50,
                        type=int, help="number of epoch checkpoint to save")

    parser.add_argument('--is_cka', action='store_true', default=False,
                        help="[NO USE]flag for whether calculate CKA Score for each block")
    parser.add_argument('--cka_score_name', type=str,
                        default='[NO USE]cka_score.csv', help='the cka score record file name')
    parser.add_argument('--metric_method', type=str, default='acc', help='metric method')

    args = parser.parse_args()

    args.model_path = './save/distill_online'
    args.model_path += '/' + str(args.dataset)
    # args.tb_path = os.path.join(args.model_path, 'tensorboard')

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # args.model_t = utils.get_teacher_name(args.path_t)
    # if args.distill == 'sokd':
    #     args.model_t = utils.get_teacher_name(args.path_t)

    if args.distill == 'sokd':
        args.model_name = "semi-online_"
        args.model_name += 'S-{}_T-{}_{}_{}_r-{}_a-{}_b-{}_{}'.format(args.models_list[0], args.model_t,
                                                                      args.dataset, args.distill, args.gamma,
                                                                      args.alpha, args.beta, args.trial)
    else:
        args.model_name = "online_"
        for idx, model_ in enumerate(args.models_list):
            args.model_name += 'M' + str(idx + 1) + "-" + str(model_) + "_"
        args.model_name += '{}_{}_r-{}_a-{}_b-{}_{}'.format(args.dataset, args.distill,
                                                            args.gamma, args.alpha, args.beta, args.trial)

    args.cka_score_name = args.trial + '_' + args.cka_score_name
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


def load_teacher(model_path, model_t, args):
    logging.info('==> loading teacher model')
    # model_t = get_teacher_name(model_path)
    # model = model_dict[model_t](num_classes=n_cls)
    model_t.load_state_dict(torch.load(model_path)['state_dict'])
    if args.distill == 'sokd':
        for k, v in model_t.named_parameters():
            if 'auxiliary' not in k:
                v.requires_grad = False
    logging.info('==> done')
    return model_t

def load_teacher_sokd(model_path, model_t, args):
    logging.info('==> loading teacher model')
    # model_t = get_teacher_name(model_path)
    # model = model_dict[model_t](num_classes=n_cls)
    model_t.load_state_dict(torch.load(model_path)['state_dict'])
    # if args.distill == 'sokd':
    #     for k, v in model_t.named_parameters():
    #         if 'auxiliary' not in k:
    #             v.requires_grad = False
    logging.info('==> done')
    return model_t


def resume_model(model_1, model_2, optimizer_list, args):
    checkpoint_1 = torch.load(args.checkpoint_1)
    model_1.load_state_dict(checkpoint_1['state_dict'])
    optimizer_list[0].load_state_dict(checkpoint_1['optim_dict'])
    start_epoch = checkpoint_1['epoch'] + 1

    checkpoint_2 = torch.load(args.checkpoint_2)
    model_2.load_state_dict(checkpoint_2['state_dict'])
    if args.distill != 'sokd':
        optimizer_list[1].load_state_dict(checkpoint_2['optim_dict'])

    return model_1, model_2, optimizer_list, start_epoch

def resume_model_sokd(module_list, aux_module, optimizer_list, args):
    checkpoint_1 = torch.load(args.checkpoint_1)
    module_list[0].load_state_dict(checkpoint_1['state_dict'])
    start_epoch = checkpoint_1['epoch'] + 1

    checkpoint_2 = torch.load(args.checkpoint_2)
    module_list[1].load_state_dict(checkpoint_2['state_dict'])
    aux_module.load_state_dict(checkpoint_2['state_dict_auxmodule'])
    optimizer_list[0].load_state_dict(checkpoint_2['optim_dict'])

    return module_list, aux_module, optimizer_list, start_epoch

def resume_model_ffl(module_list, fusion_module, optimizer_list, args):
    checkpoint_1 = torch.load(args.checkpoint_1)
    module_list[0].load_state_dict(checkpoint_1['state_dict'])
    optimizer_list[0].load_state_dict(checkpoint_1['optim_dict'])
    start_epoch = checkpoint_1['epoch'] + 1

    checkpoint_2 = torch.load(args.checkpoint_2)
    module_list[1].load_state_dict(checkpoint_2['state_dict'])
    optimizer_list[1].load_state_dict(checkpoint_2['optim_dict'])

    fusion_module.load_state_dict(checkpoint_1['state_dict_fmodule'])
    optimizer_list[2].load_state_dict(checkpoint_1['optim_dict_fmodule'])

    return module_list, fusion_module, optimizer_list, start_epoch


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

    if args.distill == 'sokd':
        assert len(args.models_list) == 1
        assert args.model_t is not None
    else:
        assert len(args.models_list) == 2

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

    tb_logger = SummaryWriter(log_dir=args.tb_folder)

    # load model
    # 0 表示 T， 1 表示 S
    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])
    if args.distill == 'sokd':
        load_model_str = '==>Semi-Online Distill Method:\t' + str(args.distill) + '\tT: ' + str(args.model_t) + '\t'
        # logging.info('==> loading teacher model: ' + args.model_t)
        model_t = model_dict[args.model_t](num_classes=args.num_class).cuda()
        model_t = load_teacher_sokd(args.path_t, model_t, args)
        # model_t = load_teacher(args.path_t, args.num_class, args.model_t, args)
        # logging.info('==> done')

        model_t.eval()
        module_list.append(model_t)
        # 加载 SOKD 的辅助块
        aux_module = auxiliary_forward(args.model_t, model_t)
        aux_module.train()
        trainable_list.append(aux_module)
        for k, v in model_t.named_parameters():
            v.requires_grad = False
    else:
        load_model_str = '==>Online Distill Method:\t' + str(args.distill) + '\t'
    # logging.info('Online Distill Method:' + str(args.distill) + '_Model1:' + str(args.model_t) + '-Model2:' + str(args.model_s))

    for idx, model_ in enumerate(args.models_list):
        if args.distill == 'sokd':
            load_model_str += 'S' + str(idx + 1) + ': ' + str(model_) + '\t'
            args.model_s = args.models_list[0]
        else:
            load_model_str += 'Model' + str(idx + 1) + ': ' + str(model_) + '\t'
            if idx == 0:
                args.model_t = args.models_list[0]
            else:
                args.model_s = args.models_list[idx]
        # logging.info('==> model_' + str(idx+1) + ': ' + str(model_))
        model_idx = model_dict[model_](num_classes=args.num_class).cuda()
        module_list.append(model_idx)
        trainable_list.append(model_idx)


    load_model_str += '_Dataset:' + str(args.dataset) + '-Trial:' + str(args.trial)
    logging.info(load_model_str)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T) # NO USE FOR FFL

    if args.dataset == 'cifar100' or args.dataset == 'cifar10':
        data = torch.randn(2, 3, 32, 32).cuda()
    else:
        data = torch.randn(2, 3, 224, 224).cuda()
    module_list[0].eval()
    module_list[1].eval()
    feat_0, _ = module_list[0](data, is_feat=True)
    feat_1, _ = module_list[1](data, is_feat=True)

    if args.distill == 'dml':
        criterion_kd = DMLLoss()
    elif args.distill == 'kdcl':
        criterion_kd = KDCLLoss()
    elif args.distill == 'sokd':
        criterion_kd = SOKDLoss(args.kd_T, args.sokd_aux_t, args.sokd_aux_s, args.sokd_kd)
    elif args.distill == 'ffl':
        input_channels = feat_0[-2].shape[1] + feat_1[-2].shape[1]
        # nn.AdaptiveAvgPool2d((4, 4))
        # 不加入 module_list，而是单独使用，但需要更新。
        fusion_module = Fusion_module(channel=input_channels, numclass=args.num_class).cuda()
        # module_list.append(fusion_module) # module_list[2]
        trainable_list.append(fusion_module) # trainable_list[2]
        criterion_kd = KLLoss()
    elif args.distill == 'switokd':
        criterion_kd = None

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)

    # optimizer
    optimizer_list = []
    if args.distill == 'sokd':
        optimizer = optim.SGD([{'params': trainable_list[1].parameters()},
                               {'params': trainable_list[0].parameters(), 'lr': args.lr_sokd_aux}],
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_list.append(optimizer)
    else:
        for each_trainable in trainable_list:
            optimizer = optim.SGD(each_trainable.parameters(),
                                  lr=args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            optimizer_list.append(optimizer)

    # SOKD module_list: 0 : T, 1: S    trainable_list: 0: aux_module, 1: S  optimizer_list 0: aux和S
    # FFL module_list: 0 : S1, 1: S2    trainable_list: 0: S1, 1: S2, 2: fusion_module  optimizer_list 0: S1, 1: S2, 2: fusion
    # Swit-OKD module_list: 0 : S1, 1: S2    trainable_list: 0: S1, 1: S2  optimizer_list 0: S1, 1: S2
    # KDCL, DML module_list: 0: S1, 1:S2    trainable_list: 0: S1, 1:S2 optimizer_list 0: S1, 1:S2
    if args.is_resume:
        logging.info('==> Resume training')
        if args.distill == 'ffl':
            module_list, fusion_module, optimizer_list, start_epoch = resume_model_ffl(module_list, fusion_module, optimizer_list, args)
        elif args.distill == 'sokd':
            module_list, aux_module, optimizer_list, start_epoch = resume_model_sokd(module_list, aux_module, optimizer_list, args)
        else:
            module_list[0], module_list[1], optimizer_list, start_epoch = resume_model(module_list[0], module_list[1],
                                                                                       optimizer_list, args)
    else:
        # if args.distill == 'sokd':
        #     module_list[0] = load_teacher(args.path_t, module_list[0], args)
            # module_list[0] = load_teacher_sokd(args.path_t, module_list[0], args)
        start_epoch = 0

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        trainable_list.cuda()
        # cudnn.enabled = False
        cudnn.benchmark = True
        cudnn.enabled = True

    if args.distill == 'ffl':
        best_val_acc_top1 = [0.0] * len(args.models_list)
    else:
        best_val_acc_top1 = [0.0] * len(trainable_list)

    # for i in range(len(args.models_list)):
    #     best_val_acc_top1_each = 0.0
    #     best_val_acc_top1.append(best_val_acc_top1_each)

    for epoch in range(start_epoch, args.epochs):
        for each_optimizer in optimizer_list:
            utils.adjust_learning_rate(epoch, args, each_optimizer)
        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1, args.epochs, optimizer_list[0].param_groups[0]['lr']))

        logging.info("==> training...")

        if args.distill == 'ffl':
            train_acc_list, train_loss_list = train_one_epoch.train_online_distill_ffl(
                optimizer_list, module_list, fusion_module, criterion_list, train_dl, epoch, args)
        elif args.distill == 'sokd':
            train_acc_list, train_loss_list = train_one_epoch.train_online_distill_sokd(
                optimizer_list, module_list, aux_module, criterion_list, train_dl, epoch, args)
        elif args.distill == 'switokd':
            train_acc_list, train_loss_list = train_one_epoch.train_online_distill_switokd(
                optimizer_list, module_list, criterion_list, train_dl, epoch, args)
        else:
            train_acc_list, train_loss_list = train_one_epoch.train_online_distill(
                optimizer_list, module_list, criterion_list, train_dl, epoch, args)

        # Evaluate for one epoch on validation set
        val_metrics_list = []
        val_acc_top1 = []
        val_acc_top5 = []
        # is_best = []
        for idx, model_ in enumerate(module_list):
            if args.distill == 'sokd' and idx == 0:
                val_metrics_ = train_one_epoch.evaluate_sokd_aux(model_, aux_module, criterion_cls, dev_dl, args,
                                                        metric_method=args.metric_method)
            else:
                val_metrics_ = train_one_epoch.evaluate(model_, criterion_cls, dev_dl, args, metric_method=args.metric_method)
            val_metrics_list.append(val_metrics_)

            val_metrics_['epoch'] = epoch + 1
            top1_ = val_metrics_['top1']
            top5_ = val_metrics_['top5']
            val_acc_top1.append(top1_)
            val_acc_top5.append(top5_)
            is_best_ = top1_ > best_val_acc_top1[idx]
            # is_best.append()

            # Save weights
            if args.distill == 'sokd':
                model_state_dict = {'epoch': epoch,
                                    'acc': top1_,
                                    'state_dict': model_.state_dict()}
                if idx == 0:
                    utils.save_checkpoint(model_state_dict,
                                          is_best=is_best_, args=args,
                                          save_folder=args.save_folder,
                                          is_teacher=True,
                                          name=args.model_t)
                else:
                    model_state_dict.update({'state_dict_auxmodule': aux_module.state_dict()})
                    model_state_dict.update({'optim_dict': optimizer_list[0].state_dict()})
                    utils.save_checkpoint(model_state_dict,
                                          is_best=is_best_, args=args,
                                          save_folder=args.save_folder,
                                          is_teacher=False,
                                          name=args.models_list[0])
            elif args.distill == 'ffl':
                model_state_dict = {'epoch': epoch,
                                    'acc': top1_,
                                    'state_dict': model_.state_dict(),
                                    'optim_dict': optimizer_list[idx].state_dict()}
                if idx == 0:
                    model_state_dict.update({'state_dict_fmodule': fusion_module.state_dict()})
                    model_state_dict.update({'optim_dict_fmodule': optimizer_list[2].state_dict()})
                utils.save_checkpoint(model_state_dict,
                                      is_best=is_best_, args=args,
                                      save_folder=args.save_folder,
                                      is_teacher=False,
                                      name="Model" + str(idx))
            else:
                model_state_dict = {'epoch': epoch,
                                    'acc': top1_,
                                    'state_dict': model_.state_dict(),
                                    'optim_dict': optimizer_list[idx].state_dict()}
                utils.save_checkpoint(model_state_dict,
                                      is_best=is_best_, args=args,
                                      save_folder=args.save_folder,
                                      is_teacher=False,
                                      name="Model" + str(idx))

            # If best_eval, best_save_path
            if is_best_:
                logging.info("- Found Model" + str(idx + 1) + " new best accuracy")
                best_val_acc_top1[idx] = top1_

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(
                    args.save_folder, "eval_Model" + str(idx + 1) + "_best_results.json")
                utils.save_dict_to_json(val_metrics_, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(
                args.save_folder, "eval_Model" + str(idx + 1) + "_last_results.json")
            utils.save_dict_to_json(val_metrics_, last_json_path)

            # Tensorboard
            if args.distill != 'sokd':
                tb_logger.add_scalar('Model' + str(idx + 1) + '_Train_accuracy', train_acc_list[idx], epoch)
                tb_logger.add_scalar('Model' + str(idx + 1) + '_Train_loss', train_loss_list[idx], epoch)
                tb_logger.add_scalar('Model' + str(idx + 1) + '_Test_accuracy_top1', top1_, epoch)
                tb_logger.add_scalar('Model' + str(idx + 1) + '_Test_accuracy_top5', top5_, epoch)
                tb_logger.add_scalar('Model' + str(idx + 1) + '_Test_loss', val_metrics_['loss'], epoch)

        if args.distill == 'sokd':
            tb_logger.add_scalar('Teacher_Train_accuracy', train_acc_list[0], epoch)
            # tb_logger.add_scalar('Teacher_Train_loss', train_loss_list[0], epoch)
            tb_logger.add_scalar('Train_loss', train_loss_list[0], epoch)
            tb_logger.add_scalar('Teacher_Test_accuracy_top1', val_acc_top1[0], epoch)
            tb_logger.add_scalar('Teacher_Test_accuracy_top5', val_acc_top1[0], epoch)
            tb_logger.add_scalar('Teacher_Test_loss', val_metrics_list[0]['loss'], epoch)
            tb_logger.add_scalar('Student_Train_accuracy', train_acc_list[1], epoch)
            # tb_logger.add_scalar('Student_Train_loss', train_loss_list[0], epoch)
            tb_logger.add_scalar('Student_Test_accuracy_top1', val_acc_top1[1], epoch)
            tb_logger.add_scalar('Student_Test_accuracy_top5', val_acc_top1[1], epoch)
            tb_logger.add_scalar('Student_Test_loss', val_metrics_list[1]['loss'], epoch)
        elif args.distill == 'ffl':
            tb_logger.add_scalar('Fusion_Model_Train_accuracy', train_acc_list[2], epoch)
            tb_logger.add_scalar('Fusion_Model_Train_loss', train_loss_list[2], epoch)


    tb_logger.close()

    for i in range(len(module_list)):
        logging.info('Model' + str(i + 1) + '_best accuracy top1: ' + str(best_val_acc_top1[i]))
    # if args.distill == 'sokd':
    #     logging.info('Teacher best accuracy top1: ' + str(best_val_acc_top1[0]))
    #     logging.info('Student best accuracy top1: ' + str(best_val_acc_top1[1]))
    # else:
    #     for i in range(len(args.models_list)):
    #         logging.info('Model' + str(i+1) + '_best accuracy top1: ' + str(best_val_acc_top1[i]))

    end_time = time.time()
    time_consuming = end_time - start_time
    time_consuming = datetime.timedelta(seconds=time_consuming)
    logging.info('Distill:\t' + str(args.distill) + '\tDataset:\t' + str(args.dataset) + '\tTrial:\t' + str(args.trial))
    logging.info('Time Consuming:\t' + str(time_consuming))


if __name__ == '__main__':
    main()
