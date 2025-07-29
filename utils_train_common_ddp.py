import argparse
import logging
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from dataloader import fetch_dataloader_1, fetch_dataloader_2, fetch_dataloader_split
from zoo import *
from dataloader import fetch_dataset as fetch_dataset_part
import torch.distributed as dist
from utils_ddp import is_main_process

def parse_option():
    parser = argparse.ArgumentParser(description='ImageNet Training')
    # parser.add_argument('data', metavar='DIR',
    #                     help='path to dataset')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
    #                     choices=model_names,
    #                     help='model architecture: ' +
    #                         ' | '.join(model_names) +
    #                         ' (default: resnet18)')
    # parser.add_argument('-at', '--arch_teacher', metavar='ARCH', default='densenet121',
    #                     choices=model_names,
    #                     help='model architecture: ' +
    #                         ' | '.join(model_names) +
    #                         ' (default: densenet121)')
    parser.add_argument('-j', '--num_workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-p', '--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    # parser.add_argument('--is_resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--model_s', type=str, default='ResNet18_another')
    parser.add_argument('--model_t', type=str, default='ResNet18_another')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--distill', type=str, default='our')
    parser.add_argument('--feedback_time', type=int, default=0)

    # 90[30, 60, 80] lr 0.1 decay 0.1 weightdecay 1e-4 batchsize256
    # optimization
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--learning_rate_t', type=float,
                        default=5e-3, help='initial learning rate for Teacher in Feedback process')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='30,60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')

    # parser.add_argument('--params_group_method', type=str, default='gradually')
    parser.add_argument('--scheduler_method', choices=['CosineAnnealingLR', 'MultiStepLR'], type=str, default='MultiStepLR')
    parser.add_argument('--CosineSch_Tmax', type=int, default=90, help="the 'T_max' parameter in CosineAnnealingLR Scheduler")
    parser.add_argument('--optimizer_method', choices=['sgd', 'adam'], type=str, default='sgd')

    # model
    parser.add_argument('--method', type=str, default='1', help='Train metohd, Data split method')

    # hyper-parameters
    parser.add_argument('--kl_loss', type=float, default=1, help='weight balance for forward_Logits')
    parser.add_argument('--ce_loss', type=float, default=1, help='weight balance for forward_CE')
    parser.add_argument('--infwd_loss', type=float, default=1, help='weight balance for forward_Interlayfeature')
    parser.add_argument('--infb_loss', type=float, default=1, help='weight balance for feedback_Interlayfeature')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    # AE hyper-parameters
    parser.add_argument('--aefwd_t_loss', type=float, default=1, help='weight balance for forward_AE_T')
    parser.add_argument('--aefwd_s_loss', type=float, default=1, help='weight balance for forward_AE_S')
    parser.add_argument('--aefb_t_loss', type=float, default=1, help='weight balance for feedback_AE_T')
    parser.add_argument('--aefb_s_loss', type=float, default=1, help='weight balance for feedback_AE_S')

    parser.add_argument('--fusion_size', type=str, default='littleSmall',
                        choices=['Big', 'Mean', 'littleSmall', 'Small', 'largeSmall', 'hugeSmall', 'numClass', 'ADP'],
                        help='[for 111] which method for AE FUSION FEATURE SIZE, Mean is base')
    parser.add_argument('--fusion_method_AE_t', type=str, default='ADPAELinear',
                        choices=['AELinear', 'AEConv', 'Mean', 'LinearSingle', 'AEConv3x3Single', 'AEConv3x3Linear',
                                 'ADPAELinear', 'ADPAEConv3x3Linear', 'ADPAEConv1x1Linear'],
                        help='[for 111] which AE method (Linear or Conv or Mean) used for Teacher Interlayer knowledge fusion')
    parser.add_argument('--fusion_method_AE_s', type=str, default='ADPAELinear',
                        choices=['AELinear', 'AEConv', 'Mean', 'LinearSingle', 'AEConv3x3Single', 'AEConv3x3Linear',
                                 'ADPAELinear', 'ADPAEConv3x3Linear', 'ADPAEConv1x1Linear'],
                        help='[for 111] which AE method (Linear or Conv or Mean) used for Student Interlayer knowledge fusion')
    parser.add_argument('--aux_method_t', type=str, default='Basic',
                        choices=['Bottle', 'Basic', 'Bottle_big', 'Basic_big', 'Basic_133', 'Basic_313', 'Basic_313_another'],
                        help='which method block used for Teacher aux classifiers')
    parser.add_argument('--aux_method_s', type=str, default='Basic',
                        choices=['Bottle', 'Basic', 'Bottle_big', 'Basic_big', 'Basic_133', 'Basic_313', 'Basic_313_another'],
                        help='which method block used for Student aux classifiers')
    parser.add_argument('--self_method_t', type=str, default='bi_directional',
                        choices=['deep_shallow', 'shallow_deep', 'bi_directional', 'deep_shallow_single', 'shallow_deep_single'],
                        help='which criterion used for Teacher SELF-SUPERVISED way')
    parser.add_argument('--self_method_s', type=str, default='bi_directional',
                        choices=['deep_shallow', 'shallow_deep', 'bi_directional', 'deep_shallow_single',
                                 'shallow_deep_single'],
                        help='which criterion used for Student SELF-SUPERVISED way')
    parser.add_argument('--in_criterion', type=str, default='KL_softmax',
                        choices=['MSE', 'MSE_normalize', 'MSE_softmax', 'MSE_softmax_T', 'KL_softmax', 'L1'],
                        help='which criterion (MSE or KL) used for calculating Interlayer loss')
    parser.add_argument('--blocks_amount_t', type=int, default=4, choices=[1, 2, 3, 4],
                        help='[for 111] choose how many Interlayer Block of Teacher Network to Distill, must >= 1 ')
    parser.add_argument('--blocks_amount_s', type=int, default=4, choices=[1, 2, 3, 4],
                        help='[for 111] choose how many Interlayer Block of Student Network to Distill, must >= 1 ')

    # resume
    parser.add_argument('--is_resume', action='store_true', default=False, help="flag for whether Resume training")
    parser.add_argument('--checkpoint_t', default='./save/distill_our/', help="teacher model resume checkpoint path")
    parser.add_argument('--checkpoint_s', default='./save/distill_our/', help="student model resume checkpoint path")

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['cifar100', 'cifar10', 'tiny_imagenet', 'cub200', 'cars196', 'imagenet', 'caltech256', 'food101', 'flowers102'],
                        help='dataset')
    parser.add_argument('--dataset_lmdb', action='store_true', default=False,
                        help="flag for whether use lmdb to read dataset, especially for ImageNet")
    parser.add_argument('--dataset_test_percent', type=float,
                        default=0.25, help='[for Caltech256] test dataset percent')
    parser.add_argument('--dataset_dir', type=str, default=None, help='whether appoint dataset dir path')
    parser.add_argument('--num_class', default=1000,
                        type=int, help="number of classes")
    parser.add_argument('--fb_set_percent', type=float,
                        default=0.3, help='[For args.method==3]feedback data percent')
    parser.add_argument('--dataset_droplast', action='store_true', default=False,
                        help="flag for whether use DropLast to make DataLoader")
    parser.add_argument('--augmentation', type=str,
                        default='yes', help='dataset augmentation')
    # parser.add_argument('--subset_percent', type=float,
    #                     default=1.0, help='subset_percent')

    parser.add_argument('-t', '--trial', type=str,
                        default='test', help='the experiment id')
    parser.add_argument('--is_checkpoint', action='store_true', default=False, help="flag for whether save checkpoint")
    parser.add_argument('--save_checkpoint_amount', default=50,
                        type=int, help="number of epoch checkpoint to save")
    parser.add_argument('--loss2csv', action='store_false', default=True, help="[for 111]flag for whether save loss to CSV file")

    parser.add_argument('--ende_use_relu', action='store_false', default=True,
                        help="flag for whether use relu in FCDecoder")

    # Ablation
    parser.add_argument('--NO_SELF', action='store_true', default=False,
                        help="flag for whether use relu in FCDecoder")
    parser.add_argument('--NO_FUSION', action='store_true', default=False,
                        help="flag for whether use relu in FCDecoder")
    parser.add_argument('--fbUseGradSim', action='store_false', default=True,
                        help="flag for whether use Grad Similarity switch during feedback process")

    parser.add_argument('--metric_method', type=str, default='acc', help='metric method')

    args = parser.parse_args()

    args.model_path = './save/distill_our'
    args.model_path += '/' + str(args.dataset)
    # args.tb_path = os.path.join(args.model_path, 'tensorboard')

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.model_name = "our_"
    args.model_name += 'S-{}_T-{}_{}_{}_{}'.format(args.model_s, args.model_t,
                                                   args.dataset,
                                                   args.scheduler_method, args.trial)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.is_checkpoint:
        args.checkpoint_save_pth = os.path.join(args.save_folder, 'checkpoint')
        if not os.path.isdir(args.checkpoint_save_pth):
            os.makedirs(args.checkpoint_save_pth)

    args.loss_txt_name = 'loss_' + str(args.trial) + '.txt'

    args.tb_folder = os.path.join(args.save_folder, str(args.trial) + '_tensorboard')
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    argsDict = args.__dict__
    with open(os.path.join(args.save_folder, 'config.txt'), 'w', encoding='utf-8') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    return args

def load_teacher(args):
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info('==> loading teacher model')
    # model_t = get_teacher_name(model_path)
    # model = model_dict[model_t](num_classes=n_cls)
    if args.gpu is None:
        checkpoint = torch.load(args.path_t)
    elif torch.cuda.is_available():
        # if args.distributed:
        #     dist.barrier()
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.path_t, map_location=loc)
    args.module_dict['model_t'].load_state_dict(checkpoint['state_dict'])

    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info('==> done')
    # return model_t

def select_scheduler(optimizer, args):
    if args.scheduler_method == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    elif args.scheduler_method == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.CosineSch_Tmax)

    return scheduler

def select_random_data(args):

    if args.dataset == 'cifar100' or args.dataset == 'cifar10':
        data = torch.randn(2, 3, 32, 32).to(args.device)
    else:
        data = torch.randn(2, 3, 224, 224).to(args.device)

    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info("Data shape: " + str(data.shape))

    return data

def select_interlayer(model, feat, dim, args):
    # 获得模型所选中间层的shape
    # 连续切片，全部最多切出四块，再从中选择所需数量
    if 'ResNet' in model:
        feat_inter = feat[1:-1][-args.blocks_amount:]
    elif 'MobileNetV2' in model:
        feat_inter = feat[1:-1][-args.blocks_amount:]
    elif 'VGG' in model:
        feat_inter = feat[1:-1][-args.blocks_amount:]
    elif 'WRN' in model:
        feat_inter = feat[1:-1][-args.blocks_amount:]
    elif 'ShuffleV2' in model:
        feat_inter = feat[1:-1][-args.blocks_amount:]

    assert len(feat_inter) == args.blocks_amount

    for each_ in feat_inter:
        dim.append(each_.shape)

def select_interlayer111(model, feat, dim, blocks_amount, args):
    # 获得模型所选中间层的shape for 111
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

    for each_ in feat_inter:
        dim.append(each_.shape)

def select_interlayer_cf(model, feat, dim, args):
    # 获得模型所选需要添加 辅助分类器 的 中间层 的shape
    # 连续切片，全部最多切出四块，再从中选择所需数量
    if 'ResNet' in model:
        feat_inter = feat[1:-1][-args.auxCF_blocks_amount:]
    elif 'MobileNetV2' in model:
        feat_inter = feat[1:-1][-args.auxCF_blocks_amount:]
    elif 'VGG' in model:
        feat_inter = feat[1:-1][-args.auxCF_blocks_amount:]
    elif 'WRN' in model:
        feat_inter = feat[1:-1][-args.auxCF_blocks_amount:]
    elif 'ShuffleV2' in model:
        feat_inter = feat[1:-1][-args.auxCF_blocks_amount:]

    assert len(feat_inter) == args.auxCF_blocks_amount

    for each_ in feat_inter:
        dim.append(each_.shape)


def fetch_dataset(args):
    trainset, devset = fetch_dataset_part(args)

    return trainset, devset

def fetch_dataloader(args):
    train_dl = {}
    if args.method == "1":
        train_dl_fwd, train_dl_fb = fetch_dataloader_1('train', args)
        train_dl.update({'fwd': train_dl_fwd})
        train_dl.update({'fb': train_dl_fb})
        dev_dl = fetch_dataloader_1('dev', args)
    elif args.method == "2":
        train_dl_all = fetch_dataloader_2('train', args)
        train_dl.update({'all': train_dl_all})
        dev_dl = fetch_dataloader_2('dev', args)
    elif args.method == "3":
        train_dl_fwd, train_dl_fb = fetch_dataloader_split('train', args)
        train_dl.update({'fwd': train_dl_fwd})
        train_dl.update({'fb': train_dl_fb})
        dev_dl = fetch_dataloader_split('dev', args)

    return train_dl, dev_dl