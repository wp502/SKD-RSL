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
from zoo import DistillKL, CCSLoss, CCSLoss_2, ICKDLoss, SPKDLoss, CRDLoss, KDLossv2, CDLoss, Attention, HintLoss, ConvReg, build_review_kd2, hcl
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
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='60,120,160', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--seed', type=int, default=2022403, help='random seed')

    # model
    parser.add_argument('--model_s', type=str, default='ResNet18')
    parser.add_argument('--model_t', type=str, default='ResNet18')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--distill', type=str, default='kd', choices=['ickd', 'kd', 'ccs', 'spkd', 'crd', 'cd', 'fitnet', 'at', 'reviewkd'])

    # hyper-parameters
    parser.add_argument('-a', '--alpha', type=float, default=1, help='[in ReviewKD is kd_loss_weight]weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=1, help='[in ReviewKD is kl_loss_weight]weight balance for other losses')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='[in ReviewKD is ce_loss_weight]weight for classification')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # dataset
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

    parser.add_argument('-t', '--trial', type=str,
                        default='test', help='the experiment id')
    parser.add_argument('--is_checkpoint', action='store_true', default=False, help="flag for whether save checkpoint")
    parser.add_argument('--save_checkpoint_amount', default=50,
                        type=int, help="number of epoch checkpoint to save")

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    # parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_n', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.1, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # For Is Resume
    parser.add_argument('--is_resume', action='store_true', default=False, help="flag for whether Resume training")
    parser.add_argument('--checkpoint_s', default='./save/distill/', help="student model resume checkpoint path")

    # hint layer for FitNet
    parser.add_argument('--hint_layer', default=4, type=int, choices=[0, 1, 2, 3, 4, 5])

    # ReviewKD
    parser.add_argument('--use_kl_rekd', action='store_true', default=False,
                        help='use kl kd loss')
    parser.add_argument('--kd_warm_up_rekd', type=float, default=10.0,
                        help='feature konwledge distillation loss weight warm up epochs')

    # CD
    parser.add_argument('--cd_is_edt', action='store_true', default=False, help="flag for whether use EDT for CD method")
    # parser.add_argument('--cd_loss_rate', default=6, type=float, help='loss rate for CDLoss for CD method')
    parser.add_argument('--cd_loss_factor', default=0.9, type=float, help='loss factor for CDLoss for CD method')

    parser.add_argument('--is_cka', action='store_true', default=False,
                        help="[NO USE]flag for whether calculate CKA Score for each block")
    parser.add_argument('--cka_score_name', type=str,
                        default='[NO USE]cka_score.csv', help='the cka score record file name')
    parser.add_argument('--metric_method', type=str, default='acc', help='metric method')

    args = parser.parse_args()

    args.model_path = './save/distill'
    args.model_path += '/' + str(args.dataset)
    # args.tb_path = os.path.join(args.model_path, 'tensorboard')

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # args.model_t = utils.get_teacher_name(args.path_t)

    args.model_name = 'S-{}_T-{}_{}_{}_r-{}_a-{}_b-{}_{}'.format(args.model_s, args.model_t, args.dataset, args.distill,
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


def load_teacher(model_path, n_cls, model_t):
    print('==> loading teacher model')
    # model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls).cuda()
    model.load_state_dict(torch.load(model_path)['state_dict'])
    print('==> done')
    return model


def resume_model(model_s, optimizer, args):
    checkpoint_s = torch.load(args.checkpoint_s)
    model_s.load_state_dict(checkpoint_s['state_dict'])
    optimizer.load_state_dict(checkpoint_s['optim_dict'])
    start_epoch = checkpoint_s['epoch'] + 1

    return model_s, optimizer, start_epoch

def select_interlayer_amount(model, feat, blocks_amount, args):
    # 连续切片，全部最多切出四块，再从中选择所需数量
    if 'ResNet' in model:
        feat_inter = feat[1:-1][-blocks_amount:]
    elif 'MobileNetV2' in model:
        feat_inter = feat[1:-1][-blocks_amount:]
    elif 'VGG' in model:
        feat_inter = feat[1:-1][-blocks_amount:]
    elif 'WRN' in model:
        feat_inter = feat[1:-1][-blocks_amount:]
    elif 'ShuffleV2' in model:
        feat_inter = feat[1:-1][-blocks_amount:]

    assert len(feat_inter) == blocks_amount

    # for each_ in feat_inter:
    #     dim.append(each_.shape)
    return feat_inter


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

    tb_logger = SummaryWriter(log_dir=args.tb_folder)

    # load model
    logging.info('Distill Method:\t' + str(args.distill) + '\tT:' + str(args.model_t) + '-S:' + str(
        args.model_s) + '-Dataset:' + str(args.dataset) + '-Trial:' + str(args.trial))
    # logging.info('==> student model: ' + args.model_s)
    logging.info('==> loading teacher model: ' + args.model_t)
    model_t = load_teacher(args.path_t, args.num_class, args.model_t).cuda()
    logging.info('==> done')
    model_s = model_dict[args.model_s](num_classes=args.num_class).cuda()

    # if 'cifar100' in args.dataset or 'cifar10' in args.dataset:
    #     data = torch.randn(2, 3, 32, 32).cuda()
    # elif 'tiny_imagenet' in args.dataset:
    #     if args.augmentation == 'yes':
    #         data = torch.randn(2, 3, 224, 224).cuda()
    #     elif args.augmentation == 'no':
    #         data = torch.randn(2, 3, 64, 64).cuda()
    # elif 'cars' in args.dataset or 'cub200' in args.dataset:
    #     data = torch.randn(2, 3, 224, 224).cuda()
    # elif 'imagenet' in args.dataset:
    #     data = torch.randn(2, 3, 224, 224).cuda()
    if args.dataset == 'cifar100' or args.dataset == 'cifar10':
        data = torch.randn(2, 3, 32, 32).cuda()
    else:
        data = torch.randn(2, 3, 224, 224).cuda()
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])
    if args.distill != 'reviewkd':
        module_list.append(model_s)
        trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    if args.distill == 'cd':
        criterion_div = KDLossv2(args.kd_T)
    else:
        criterion_div = DistillKL(args.kd_T)

    if args.distill == 'kd':
        criterion_kd = DistillKL(args.kd_T)
    elif args.distill == 'ickd':
        args.s_dim = feat_s[-2].shape[1]
        args.t_dim = feat_t[-2].shape[1]
        args.feat_dim = args.t_dim
        criterion_kd = ICKDLoss(args)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif args.distill == 'ccs':
        args.t_dim = []
        args.s_dim = []
        # if 'ResNet' in args.model_t:
        #     feat_t_ch = feat_t[1:-1]
        # elif 'WRN' in args.model_t:
        #     feat_t_ch = feat_t[1:-1]
        feat_t_ch = select_interlayer_amount(args.model_t, feat_t, 4, args)
        for each_ in feat_t_ch:
            args.t_dim.append(each_.shape[1])

        # if 'ResNet' in args.model_s:
        #     feat_s_ch = feat_s[1:-1]
        # elif 'WRN' in args.model_s:
        #     feat_s_ch = feat_s[1:-1]
        # elif 'ShuffleV2' in args.model_s:
        #     feat_s_ch = feat_s[:-1]
        # elif 'MobileNetV2' in args.model_s:
        #     feat_s_ch = feat_s[1:-1]
        feat_s_ch = select_interlayer_amount(args.model_s, feat_s, 4, args)
        for each_ in feat_s_ch:
            args.s_dim.append(each_.shape[1])
        criterion_kd = CCSLoss_2(args)
        module_list.append(criterion_kd.Embed_1)
        module_list.append(criterion_kd.Embed_2)
        module_list.append(criterion_kd.Embed_3)
        module_list.append(criterion_kd.Embed_4)
        trainable_list.append(criterion_kd.Embed_1)
        trainable_list.append(criterion_kd.Embed_2)
        trainable_list.append(criterion_kd.Embed_3)
        trainable_list.append(criterion_kd.Embed_4)

        # criterion_kd = CCSLoss()
    elif args.distill == 'spkd':
        criterion_kd = SPKDLoss()
    elif args.distill == 'cd':
        args.t_dim = []
        args.s_dim = []
        # if 'ResNet' in args.model_t:
        #     feat_t_ch = feat_t[1:-1]
        # elif 'WRN' in args.model_t:
        #     feat_t_ch = feat_t[1:-1]
        feat_t_ch = select_interlayer_amount(args.model_t, feat_t, 4, args)
        for each_ in feat_t_ch:
            args.t_dim.append(each_.shape[1])

        # if 'ResNet' in args.model_s:
        #     feat_s_ch = feat_s[1:-1]
        # elif 'WRN' in args.model_s:
        #     feat_s_ch = feat_s[1:-1]
        # elif 'ShuffleV2' in args.model_s:
        #     feat_s_ch = feat_s[:-1]
        # elif 'MobileNetV2' in args.model_s:
        #     feat_s_ch = feat_s[1:-1]
        feat_s_ch = select_interlayer_amount(args.model_s, feat_s, 4, args)
        for each_ in feat_s_ch:
            args.s_dim.append(each_.shape[1])
        criterion_kd = CDLoss(args)
        module_list.append(criterion_kd.Embed_1)
        module_list.append(criterion_kd.Embed_2)
        module_list.append(criterion_kd.Embed_3)
        module_list.append(criterion_kd.Embed_4)
        trainable_list.append(criterion_kd.Embed_1)
        trainable_list.append(criterion_kd.Embed_2)
        trainable_list.append(criterion_kd.Embed_3)
        trainable_list.append(criterion_kd.Embed_4)
    elif args.distill == 'crd':
        args.s_dim = feat_s[-1].shape[1]
        args.t_dim = feat_t[-1].shape[1]
        # opt.feat_dim = opt.s_dim
        args.n_data = len(train_dl.dataset)
        criterion_kd = CRDLoss(args.s_dim, args.t_dim, args.feat_dim, args.nce_n, args.nce_t, args.nce_m, args.n_data)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
        # criterion_fea = CCLoss()
    elif args.distill == 'at':
        criterion_kd = Attention()
    elif args.distill == 'fitnet':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[args.hint_layer].shape, feat_t[args.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif args.distill == 'reviewkd':
        criterion_kd = None
        cnn = build_review_kd2(model_s, feat_t, feat_s)
        module_list.append(cnn)
        trainable_list.append(cnn)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.is_resume:
        logging.info('==> Resume Training')
        logging.info('==> Resume Student Model: ' + str(args.model_s))
        module_list[0], optimizer, start_epoch = resume_model(module_list[0], optimizer, args)
    else:
        start_epoch = 0

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True

    # validate teacher accuracy
    # teacher_acc, _, _ = train_one_epoch.evaluate(dev_dl, model_t, criterion_cls, args)
    val_metrics = train_one_epoch.evaluate(model_t, criterion_cls, dev_dl, args, metric_method=args.metric_method)
    logging.info('teacher accuracy: ' + str(val_metrics['top1']))

    best_val_acc_top1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(epoch, args, optimizer)
        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        logging.info("==> training...")

        if args.distill == 'reviewkd':
            train_acc, train_loss = train_one_epoch.train_distill_reviewkd(
                optimizer, module_list, criterion_list, train_dl, epoch, args)
        else:
            train_acc, train_loss = train_one_epoch.train_distill(
                optimizer, module_list, criterion_list, train_dl, epoch, args)

        # Evaluate for one epoch on validation set
        # val_metrics = train_one_epoch.evaluate(model_s, criterion_cls, dev_dl, args, metric_method=args.metric_method)
        if args.distill == 'reviewkd':
            val_metrics = train_one_epoch.evaluate_reviewkd(module_list[0], criterion_cls, dev_dl, args, metric_method=args.metric_method)
        else:
            val_metrics = train_one_epoch.evaluate(module_list[0], criterion_cls, dev_dl, args, metric_method=args.metric_method)

        val_metrics['epoch'] = epoch + 1
        val_acc_top1 = val_metrics['top1']
        val_acc_top5 = val_metrics['top5']
        is_best = val_acc_top1 > best_val_acc_top1

        # Save weights
        # utils.save_checkpoint({'epoch': epoch,
        #                        'acc': val_acc_top1,
        #                        'state_dict': model_s.state_dict(),
        #                        'optim_dict': optimizer.state_dict()},
        #                       is_best=is_best,
        #                       args=args,
        #                       save_folder=args.save_folder,
        #                       is_teacher=False,
        #                       name=args.model_s)
        utils.save_checkpoint({'epoch': epoch,
                               'acc': val_acc_top1,
                               'state_dict': module_list[0].state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              args=args,
                              save_folder=args.save_folder,
                              is_teacher=False,
                              name=args.model_s)
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
    logging.info('Distill:\t' + str(args.distill) + '\tDataset:\t' + str(args.dataset) + '\tTrial:\t' + str(args.trial))
    logging.info('Time Consuming:\t' + str(time_consuming))


if __name__ == '__main__':
    main()
