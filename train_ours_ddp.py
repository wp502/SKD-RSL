import argparse
import logging
import os
import random
import shutil
import time
import warnings
import utils_ddp
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from models import model_dict
import numpy as np
import utils_train_common_ddp
from zoo import DistillKL, Our_FWD_ddp, Our_FB_ddp, initAUXAndAE_ddp
import train_one_epoch_ddp
import datetime
from utils_ddp import is_main_process

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))


#parser.add_argument('--selfKD', default=1, type=int, help="parameter for self training or normal training")
#parser.add_argument('--regularization', action='store_true', help='parameter for regularization')
#parser.add_argument('--smoothing', action='store_true', help='parameter for label smoothing')
#parser.add_argument('--selfKD', action='store_true', help="parameter for self training or normal training")

# best_acc1 = 0

def resume_model(args):
    if args.gpu is None:
        checkpoint_t = torch.load(args.checkpoint_t)
        checkpoint_s = torch.load(args.checkpoint_s)
    elif torch.cuda.is_available():
        # if args.distributed:
        #     dist.barrier()
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint_t = torch.load(args.checkpoint_t, map_location=loc)
        checkpoint_s = torch.load(args.checkpoint_s, map_location=loc)
    # T
    # checkpoint_t = torch.load(args.checkpoint_t)
    args.module_dict['model_t'].load_state_dict(checkpoint_t['state_dict'])
    args.module_auxcfae_t_dict.load_state_dict(checkpoint_t['auxcfae_dict'])
    args.optimizer_dict['opt_fb'].load_state_dict(checkpoint_t['optim_dict'])
    args.scheduler_dict['sch_fb'].load_state_dict(checkpoint_t['scheduler_dict'])
    args.start_epoch = checkpoint_t['epoch'] + 1
    args.best_val_acc_top1['bestAcc_model_t'] = checkpoint_t['best_acc']
    args.optimizer_dict['opt_ae_t'].load_state_dict(checkpoint_t['optimizer_ae_dict'])
    args.scheduler_dict['sch_ae_t'].load_state_dict(checkpoint_t['scheduler_ae_dict'])
    args.optimizer_dict['opt_auxcf_t'].load_state_dict(checkpoint_t['optimizer_auxcf_dict'])
    args.scheduler_dict['sch_auxcf_t'].load_state_dict(checkpoint_t['scheduler_auxcf_dict'])

    # S

    args.module_dict['model_s'].load_state_dict(checkpoint_s['state_dict'])
    args.module_auxcfae_s_dict.load_state_dict(checkpoint_s['auxcfae_dict'])
    args.optimizer_dict['opt_fwd'].load_state_dict(checkpoint_s['optim_dict'])
    args.scheduler_dict['sch_fwd'].load_state_dict(checkpoint_s['scheduler_dict'])
    args.best_val_acc_top1['bestAcc_model_s'] = checkpoint_s['best_acc']
    args.optimizer_dict['opt_ae_s'].load_state_dict(checkpoint_s['optimizer_ae_dict'])
    args.scheduler_dict['sch_ae_s'].load_state_dict(checkpoint_s['scheduler_ae_dict'])
    args.optimizer_dict['opt_auxcf_s'].load_state_dict(checkpoint_s['optimizer_auxcf_dict'])
    args.scheduler_dict['sch_auxcf_s'].load_state_dict(checkpoint_s['scheduler_auxcf_dict'])

def trainable_to_cuda2(trainable, ngpus_per_node, args, model_name=None):
    # if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    #     logging.info('using CPU, this will be slow')
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                trainable.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                trainable = torch.nn.parallel.DistributedDataParallel(trainable, device_ids=[args.gpu])
            else:
                trainable.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                trainable = torch.nn.parallel.DistributedDataParallel(trainable)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        trainable = trainable.cuda(args.gpu)
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     trainable = trainable.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if model_name is not None:
            if model_name.startswith('AlexNet') or model_name.startswith('VGG'):
                trainable.features = torch.nn.DataParallel(trainable.features)
                trainable.cuda()
            else:
                trainable = torch.nn.DataParallel(trainable).cuda()
        else:
            trainable = torch.nn.DataParallel(trainable).cuda()

    return trainable

def select_optimizer(args):
    args.optimizer_dict = {}
    args.warmup_scheduler_dict = {}
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info('Add Param Groups Method ALL')
    # utils_train_common_imagenet.select_parameters_group(args)
    # args.param_groups_list = args.trainable_fb_dict.parameters()
    if args.optimizer_method == 'sgd':
        optimizer_ae_t = optim.SGD(args.trainable_ae_t_dict.parameters(), lr=args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        optimizer_ae_s = optim.SGD(args.trainable_ae_s_dict.parameters(), lr=args.learning_rate,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
        optimizer_auxcf_t = optim.SGD(args.trainable_auxcf_t_dict.parameters(), lr=args.learning_rate,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
        optimizer_auxcf_s = optim.SGD(args.trainable_auxcf_s_dict.parameters(), lr=args.learning_rate,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
        optimizer_fwd = optim.SGD(args.trainable_fwd_dict.parameters(), lr=args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        optimizer_fb = optim.SGD(args.trainable_fb_dict.parameters(),
            lr=args.learning_rate_t, momentum=args.momentum, weight_decay=args.weight_decay)


    args.optimizer_dict.update({'opt_fwd': optimizer_fwd})
    args.optimizer_dict.update({'opt_ae_t': optimizer_ae_t})
    args.optimizer_dict.update({'opt_ae_s': optimizer_ae_s})
    args.optimizer_dict.update({'opt_auxcf_t': optimizer_auxcf_t})
    args.optimizer_dict.update({'opt_auxcf_s': optimizer_auxcf_s})
    args.optimizer_dict.update({'opt_fb': optimizer_fb})

def sth_needTo_train(model_t, model_s, args):
    args.module_dict = nn.ModuleDict({})
    args.module_auxcfae_s_dict = nn.ModuleDict({})
    args.module_auxcfae_t_dict = nn.ModuleDict({})
    args.trainable_fb_dict = nn.ModuleDict({})  # feedback 中需要更新的项目，只包括 教师网络
    args.trainable_fwd_dict = nn.ModuleDict({})  # forward 中需要更新的项目，只包括 学生网络
    args.criterion_dict = nn.ModuleDict({})  # 存放所有 criterion

    args.trainable_ae_t_dict = nn.ModuleDict({})  # feedback 中需要更新的其他项，自编码器项
    args.trainable_ae_s_dict = nn.ModuleDict({})  # feedback 中需要更新的其他项，自编码器项
    args.trainable_auxcf_t_dict = nn.ModuleDict({})  # feedback 中需要更新的其他项，自编码器项
    args.trainable_auxcf_s_dict = nn.ModuleDict({})  # feedback 中需要更新的其他项，自编码器项

    data = utils_train_common_ddp.select_random_data(args)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    args.module_dict.update({'model_t': model_t})
    args.module_dict.update({'model_s': model_s})

    args.trainable_fb_dict.update({'opt_model_t': model_t})
    args.trainable_fwd_dict.update({'opt_model_s': model_s})

    criterion_cls = nn.CrossEntropyLoss().to(args.device)
    criterion_div = DistillKL(args.kd_T).to(args.device)

    args.t_dim = []
    args.s_dim = []
    utils_train_common_ddp.select_interlayer111(args.model_t, feat_t, args.t_dim, args.blocks_amount_t, args)
    utils_train_common_ddp.select_interlayer111(args.model_s, feat_s, args.s_dim, args.blocks_amount_s, args)

    # 获得 教师 和 学生 中间特征层 展开后的大小
    # utils_train_common_imagenet.tsInShape(args)

    # criterion_inter_fwd, criterion_inter_fb = utils_train_common_imagenet.choiceFusionMethod(args)
    criterion_inter_fwd = Our_FWD_ddp(args.kd_T, args).to(args.device)
    criterion_inter_fb = Our_FB_ddp(args.kd_T, args).to(args.device)
    initAUXAndAE_ddp(args)

    args.module_auxcfae_t_dict.update({'model_ae_t': args.ae_t})
    args.module_auxcfae_s_dict.update({'model_ae_s': args.ae_s})

    args.trainable_ae_t_dict.update({'opt_ae_t': args.ae_t})
    args.trainable_ae_s_dict.update({'opt_ae_s': args.ae_s})
    for i in range(args.blocks_amount_s):
        args.module_auxcfae_s_dict.update({'model_auxCF_s_b' + str(i + 1): args.auxCF_s[str(i + 1)]})
        args.trainable_auxcf_s_dict.update({'opt_auxCF_s_b'+str(i+1): args.auxCF_s[str(i+1)]})
    for i in range(args.blocks_amount_t):
        args.module_auxcfae_t_dict.update({'model_auxCF_t_b' + str(i + 1): args.auxCF_t[str(i + 1)]})
        args.trainable_auxcf_t_dict.update({'opt_auxCF_t_b'+str(i+1): args.auxCF_t[str(i+1)]})

    # for each_trainable_t_name in args.trainable_ae_t_dict.keys():
    #     args.trainable_ae_t_dict[each_trainable_t_name].train()
    # for each_trainable_s_name in args.trainable_ae_s_dict.keys():
    #     args.trainable_ae_s_dict[each_trainable_s_name].train()
    # for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
    #     args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].train()
    # for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
    #     args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].train()


    args.criterion_dict.update({'cri_cls': criterion_cls})
    args.criterion_dict.update({'cri_div': criterion_div})
    args.criterion_dict.update({'cri_infwd': criterion_inter_fwd})
    args.criterion_dict.update({'cri_infb': criterion_inter_fb})

def csv_sth(args):
    args.loss_csv_fwd_name = 'loss_csv_' + str(args.trial) + '_fwd.csv'
    args.loss_csv_fb_name = 'loss_csv_' + str(args.trial) + '_fb.csv'

    csv_title_fwd = ['Epoch_Step', 'SoftTargets_KL']
    csv_title_fb = ['Epoch_Step', 'SoftTargets_KL']
    for i in range(args.blocks_amount_s):
        csv_title_fwd.append('Block' + str(i) + '_CE')

    for i in range(args.blocks_amount_s - 1):
        # deep - shallow
        deep_sender_idx = args.blocks_amount_s - 1 - i
        for each in range(deep_sender_idx):
            csv_title_fwd.append('Block' + str(deep_sender_idx) + '->Block' + str(each) + '_KL')
        # shallow - deep
        shallow_sender_idx = i
        for each in range(i + 1, args.blocks_amount_s):
            csv_title_fwd.append('Block' + str(shallow_sender_idx) + '->Block' + str(each) + '_KL')

    for i in range(args.blocks_amount_t):
        csv_title_fb.append('Block' + str(i) + '_CE')

    for i in range(args.blocks_amount_t - 1):
        # deep - shallow
        deep_sender_idx = args.blocks_amount_t - 1 - i
        for each in range(deep_sender_idx):
            csv_title_fb.append('Block' + str(deep_sender_idx) + '->Block' + str(each) + '_KL')
        # shallow - deep
        shallow_sender_idx = i
        for each in range(i + 1, args.blocks_amount_t):
            csv_title_fb.append('Block' + str(shallow_sender_idx) + '->Block' + str(each) + '_KL')

    # for i in range(args.blocks_amount_t):
    #     csv_title_fb.append('Block'+str(i)+'_CE')
    #     for j in range(args.blocks_amount_t):
    #         if i is not j:
    #             csv_title_fb.append('Block'+str(i)+'->Block'+str(j)+'_KL')
    csv_title_fwd.extend(
        ['SelfSup_CE_ALL', 'SelfSup_KL_ALL', 'SelfSup_ALL', 'reconS_CE', 'fusion_KL', 'AE_ALL', 'Distill_ALL',
         'FWD_ALL'])
    csv_title_fb.extend(
        ['SelfSup_CE_ALL', 'SelfSup_KL_ALL', 'SelfSup_ALL', 'reconT_CE', 'fusion_KL', 'GradCos_Feedback',
         'GradCos_Self', 'GradCos_Ohter', 'AE_ALL', 'Distill_ALL', 'FB_ALL'])
    csv_title_sketch = [
        'T: %s iLR: %s Block_Method: %s, Blocks_Amount: %s, SelfSup_Method: %s, S: %s iLR: %s Block_Method: %s, Blocks_Amount: %s, SelfSup_Method: %s' % (
            str(args.model_t), str(args.learning_rate_t), str(args.aux_method_t), str(args.blocks_amount_t),
            str(args.self_method_t),
            str(args.model_s), str(args.learning_rate), str(args.aux_method_s), str(args.blocks_amount_s),
            str(args.self_method_s))]

    if args.loss2csv:
        with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_title_sketch)
            writer.writerow(csv_title_fwd)

        with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_title_sketch)
            writer.writerow(csv_title_fb)

def main():
    args = utils_train_common_ddp.parse_option()
    assert args.blocks_amount_t >= 2 and args.blocks_amount_s >= 2


    args.logger_name = args.model_name + ".log"


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    warnings.filterwarnings("ignore")
    tb_logger = SummaryWriter(log_dir=args.tb_folder)

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # global best_acc1
    start_time = time.time()

    best_acc1_dict = {}

    best_acc1_dict.update({'bestAcc_model_t': torch.Tensor([0.0])})
    best_acc1_dict.update({'bestAcc_model_s': torch.Tensor([0.0])})
    args.best_val_acc_top1 = best_acc1_dict

    args.gpu = gpu
    args.txt_f = open(os.path.join(args.save_folder, args.logger_name), 'a', encoding='utf-8')
    utils_ddp.set_logger(os.path.join(args.save_folder, args.logger_name))

    if args.gpu is not None:
        if is_main_process():
            str1 = "Use GPU: {} for training".format(args.gpu)
            logger = logging.getLogger()
            logger.parent = None
            logger.info(str1)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # print(dist.get_rank())
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    args.device = device

    # sth need to train
    model_t = model_dict[args.model_t](num_classes=args.num_class).to(args.device)
    model_s = model_dict[args.model_s](num_classes=args.num_class).to(args.device)

    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info('==>Our Distill Method:' + 'T: ' + str(args.model_t) + '_S:' + str(
        args.model_s) + '_Dataset:' + str(args.dataset) + '_Trial:' + str(args.trial) + args.model_name)

    # something need to train
    sth_needTo_train(model_t, model_s, args)

    args.batch_size = int(args.batch_size / ngpus_per_node)

    select_optimizer(args)

    # scheduler
    args.scheduler_dict = {}
    for optimizer_name_ in args.optimizer_dict.keys():
        scheduler_ = utils_train_common_ddp.select_scheduler(args.optimizer_dict[optimizer_name_], args)
        if 'ae_t' in optimizer_name_:
            args.scheduler_dict.update({'sch_ae_t': scheduler_})
        elif 'ae_s' in optimizer_name_:
            args.scheduler_dict.update({'sch_ae_s': scheduler_})
        elif 'fwd' in optimizer_name_:
            args.scheduler_dict.update({'sch_fwd': scheduler_})
        elif 'fb' in optimizer_name_:
            args.scheduler_dict.update({'sch_fb': scheduler_})
        elif 'auxcf_t' in optimizer_name_:
            args.scheduler_dict.update({'sch_auxcf_t': scheduler_})
        elif 'auxcf_s' in optimizer_name_:
            args.scheduler_dict.update({'sch_auxcf_s': scheduler_})


    # optionally resume from a checkpoint
    str1 = '==> loading pre-trained model'
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info(str1)
    if args.is_resume:
        str1 = '==> Resume Training'
        if is_main_process():
            logger = logging.getLogger()
            logger.parent = None
            logger.info(str1)

        resume_model(args)
        if args.gpu is not None:
            for each_best in args.best_val_acc_top1.keys():
                args.best_val_acc_top1[each_best] = torch.Tensor([args.best_val_acc_top1[each_best]])
                args.best_val_acc_top1[each_best] = args.best_val_acc_top1[each_best].to(args.gpu)
            # args.best_val_acc_top1 = args.best_val_acc_top1.to(args.gpu)
    else:
        args.start_epoch = 0
        utils_train_common_ddp.load_teacher(args)


    # Data loading code
    # dataloader
    str1 = "Loading the datasets..."
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info(str1)

    train_dataset, dev_dataset = utils_train_common_ddp.fetch_dataset(args)
    str1 = "- done."
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info(str1)

    if args.method == "1":
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            fb_train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=2022403)
            dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, shuffle=False, drop_last=args.dataset_droplast)
        else:
            train_sampler = None
            fb_train_sampler = None
            dev_sampler = None

        trainloader_fwd = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        trainloader_fb = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                           shuffle=(fb_train_sampler is None),
                                                           num_workers=args.num_workers, pin_memory=True,
                                                           sampler=fb_train_sampler)

        # train_dl = {}
        # train_dl.update({'fwd': trainloader_fwd})
        # train_dl.update({'fb': trainloader_fb})

        dev_dl = torch.utils.data.DataLoader(
            dev_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, sampler=dev_sampler)
    else:
        print("NO THIS METHOD")
        exit()

    # for idx, (fwd_data, fb_data) in enumerate(zip(trainloader_fwd, trainloader_fb)):
    #     if idx == 2:
    #         break
    #     str_fwd_data = 'fwd\t' + str(idx) + '\t' + str(fwd_data[0].shape) + '\t' + str(fwd_data[1][:10])
    #     str_fb_data = 'fb\t' + str(idx) + '\t' + str(fb_data[0].shape) + '\t' + str(fb_data[1][:10])
    #     if is_main_process():
    #         logger = logging.getLogger()
    #         logger.parent = None
    #         logger.info(str_fwd_data)
    #         logger.info(str_fb_data)
    #     args.txt_f.write(str_fwd_data + '\n')
    #     args.txt_f.write(str_fb_data + '\n')
    #     print(str_fwd_data)
    #     print(str_fb_data)

    str1 = '==> done'
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info(str1)

    model_auxcfae_t = [each for each in args.module_auxcfae_t_dict.values()]  # 0: ae, 1-last: auxcf
    model_auxcfae_s = [each for each in args.module_auxcfae_s_dict.values()]  # 0: ae, 1-last: auxcf
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']

    model_t = trainable_to_cuda2(model_t, ngpus_per_node, args, model_name=args.model_t)
    model_s = trainable_to_cuda2(model_s, ngpus_per_node, args, model_name=args.model_s)
    for i in range(len(model_auxcfae_t)-1):
        model_auxcfae_t[i] = trainable_to_cuda2(model_auxcfae_t[i], ngpus_per_node, args)

    for i in range(len(model_auxcfae_s)-1):
        model_auxcfae_s[i] = trainable_to_cuda2(model_auxcfae_s[i], ngpus_per_node, args)
    # args.module_auxcfae_s_dict = trainable_to_cuda2(args.module_auxcfae_s_dict, ngpus_per_node, args)
    # args.module_auxcfae_t_dict = trainable_to_cuda2(args.module_auxcfae_t_dict, ngpus_per_node, args)

    # args.module_dict.to(args.device)
    # args.module_auxcfae_s_dict.to(args.device)
    # args.module_auxcfae_t_dict.to(args.device)
    args.criterion_dict.to(args.device)

    cudnn.benchmark = True
    cudnn.enabled = True

    # Test the accuracy of teacher
    val_metrics = utils_ddp.validate(dev_dl, model_t, args.criterion_dict['cri_cls'], args)
    str1 = 'Teacher accuracy: ' + str(val_metrics['top1'])
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info(str1)

    if not args.is_resume:
        args.best_val_acc_top1['bestAcc_model_s'] = torch.Tensor([0.0])
        args.best_val_acc_top1['bestAcc_model_t'] = torch.Tensor([val_metrics['top1']])
        if args.gpu is not None:
            for each_best in args.best_val_acc_top1.keys():
                args.best_val_acc_top1[each_best] = args.best_val_acc_top1[each_best].to(args.gpu)
            # args.best_val_acc_top1 = args.best_val_acc_top1.to(args.gpu)

    # csv_sth(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            fb_train_sampler.set_epoch(epoch)

        str1 = "Epoch {}/{}, FWD_lr:{}, FB_lr:{}".format(epoch + 1, args.epochs,
                                                      args.optimizer_dict['opt_fwd'].param_groups[0]['lr'],
                                                      args.optimizer_dict['opt_fb'].param_groups[0]['lr'])
        if is_main_process():
            logger = logging.getLogger()
            logger.parent = None
            logger.info(str1)

        str1 = "==> Training..."
        if is_main_process():
            logger = logging.getLogger()
            logger.parent = None
            logger.info(str1)

        # args.trainable_ae_t_dict.train()
        # args.trainable_ae_s_dict.train()
        # args.trainable_auxcf_t_dict.train()
        # args.trainable_auxcf_s_dict.train()

        model_t.train()
        model_s.train()
        for each in model_auxcfae_t:
            each.train()
        for each in model_auxcfae_s:
            each.train()

        # train for one epoch
        if args.method == "1":
            train_acc_list, train_loss_list = train_one_epoch_ddp.train_our_distill_method12(model_t, model_s, model_auxcfae_t, model_auxcfae_s, trainloader_fwd, trainloader_fb, epoch, args)
        else:
            print("NO THIS METHOD")
            exit()

        if args.distributed:
            dist.barrier()

        # learning rate update
        for scheduler_name_ in args.scheduler_dict.keys():
            args.scheduler_dict[scheduler_name_].step()

        # Evaluate for one epoch on validation set
        val_metrics_dict = {}
        val_acc_top1 = {}
        val_acc_top5 = {}
        # is_best = {}

        # model_dict_temp = {}
        # model_dict_temp.update({'model_t': model_t})
        # model_dict_temp.update({'model_s': model_s})

        for idx, model_ in enumerate(args.module_dict.keys()):
            val_metrics_ = utils_ddp.validate(dev_dl, args.module_dict[model_], args.criterion_dict['cri_cls'], args)

            val_metrics_['epoch'] = epoch + 1
            top1_ = val_metrics_['top1']
            top5_ = val_metrics_['top5']
            val_acc_top1.update({model_: top1_})
            val_acc_top5.update({model_: top5_})
            val_metrics_dict.update({model_: val_metrics_})

            is_best_ = top1_ > args.best_val_acc_top1['bestAcc_' + model_]
            # If best_eval, best_save_path
            if is_best_:
                if idx == 0:
                    str1 = "- Found Teacher new best accuracy"
                    if is_main_process():
                        logger = logging.getLogger()
                        logger.parent = None
                        logger.info(str1)

                    best_json_path = os.path.join(
                        args.save_folder, "eval_Teacher_best_results.json")
                elif idx == 1:
                    str1 = "- Found Student new best accuracy"
                    if is_main_process():
                        logger = logging.getLogger()
                        logger.parent = None
                        logger.info(str1)

                    best_json_path = os.path.join(
                        args.save_folder, "eval_Student_best_results.json")
                args.best_val_acc_top1['bestAcc_' + model_] = top1_

                if is_main_process():
                    utils_ddp.save_dict_to_json(val_metrics_, best_json_path)
            # is_best.update({model_: is_best_})

            # Save latest val metrics in a json file in the model directory
            if idx == 0:
                last_json_path = os.path.join(
                    args.save_folder, "eval_Teacher_last_results.json")
            elif idx == 1:
                last_json_path = os.path.join(
                    args.save_folder, "eval_Student_last_results.json")
            if is_main_process():
                utils_ddp.save_dict_to_json(val_metrics_, last_json_path)

            if idx == 0:
                # T
                # 直接用这个module_auxcfae_dict，具有一定的歧义。但是在测试中，args.dict的参数会跟着，放到DDP上训练后的模型，一起更新参数。
                model_auxcfae_now = args.module_auxcfae_t_dict.state_dict()
                optimizer_now = args.optimizer_dict['opt_fb'].state_dict()
                scheduler_now = args.scheduler_dict['sch_fb'].state_dict()
                optimizer_ae_now = args.optimizer_dict['opt_ae_t'].state_dict()
                scheduler_ae_now = args.scheduler_dict['sch_ae_t'].state_dict()
                optimizer_auxcf_now = args.optimizer_dict['opt_auxcf_t'].state_dict()
                scheduler_auxcf_now = args.scheduler_dict['sch_auxcf_t'].state_dict()
            elif idx == 1:
                # S
                model_auxcfae_now = args.module_auxcfae_s_dict.state_dict()
                optimizer_now = args.optimizer_dict['opt_fwd'].state_dict()
                scheduler_now = args.scheduler_dict['sch_fwd'].state_dict()
                optimizer_ae_now = args.optimizer_dict['opt_ae_s'].state_dict()
                scheduler_ae_now = args.scheduler_dict['sch_ae_s'].state_dict()
                optimizer_auxcf_now = args.optimizer_dict['opt_auxcf_s'].state_dict()
                scheduler_auxcf_now = args.scheduler_dict['sch_auxcf_s'].state_dict()

            # Save weights
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                model_state_dict = {'name': model_,
                                    'epoch': epoch,
                                    'acc': top1_,
                                    'best_acc': args.best_val_acc_top1['bestAcc_' + model_],
                                    'state_dict': args.module_dict[model_].state_dict(),
                                    'auxcfae_dict': model_auxcfae_now,
                                    'optim_dict': optimizer_now,
                                    'optimizer_ae_dict': optimizer_ae_now,
                                    'optimizer_auxcf_dict': optimizer_auxcf_now,
                                    'scheduler_auxcf_dict': scheduler_auxcf_now,
                                    'scheduler_ae_dict': scheduler_ae_now,
                                    'scheduler_dict': scheduler_now}

                if idx == 0:
                    utils_ddp.save_checkpoint(model_state_dict,
                                          is_best=is_best_, args=args,
                                          save_folder=args.save_folder,
                                          is_teacher=True,
                                          name=args.model_t)
                elif idx == 1:
                    utils_ddp.save_checkpoint(model_state_dict,
                                          is_best=is_best_, args=args,
                                          save_folder=args.save_folder,
                                          is_teacher=False,
                                          name=args.model_s)

        # Tensorboard
    #     tb_logger.add_scalar('Teacher_Train_accuracy', train_acc_list[0], epoch)
    #     tb_logger.add_scalar('Forward_Train_loss', train_loss_list[0], epoch)
    #     tb_logger.add_scalar('Teacher_Test_accuracy_top1', val_acc_top1['model_t'], epoch)
    #     tb_logger.add_scalar('Teacher_Test_accuracy_top5', val_acc_top5['model_t'], epoch)
    #     tb_logger.add_scalar('Teacher_Test_loss', val_metrics_dict['model_t']['loss'], epoch)
    #     tb_logger.add_scalar('Student_Train_accuracy', train_acc_list[1], epoch)
    #     tb_logger.add_scalar('Feedback_Train_loss', train_loss_list[1], epoch)
    #     tb_logger.add_scalar('Student_Test_accuracy_top1', val_acc_top1['model_s'], epoch)
    #     tb_logger.add_scalar('Student_Test_accuracy_top5', val_acc_top5['model_s'], epoch)
    #     tb_logger.add_scalar('Student_Test_loss', val_metrics_dict['model_s']['loss'], epoch)
    #
    # tb_logger.close()
    strT = 'Teacher:' + str(args.model_t) + '_best accuracy top1: ' + str(args.best_val_acc_top1['bestAcc_model_t'])
    strS = 'Student:' + str(args.model_s) + '_best accuracy top1: ' + str(args.best_val_acc_top1['bestAcc_model_s'])
    end_time = time.time()
    time_consuming = end_time - start_time
    time_consuming = datetime.timedelta(seconds=time_consuming)
    str_end1 = 'Dataset:\t' + str(args.dataset) + '\tTrial:\t' + str(args.trial)
    str_end2 = 'Time Consuming:\t' + str(time_consuming)

    if is_main_process():
        # logger = logging.getLogger('train')
        logger = logging.getLogger()
        logger.parent = None
        logger.info(strT)
        logger.info(strS)
        logger.info(str_end1)
        logger.info(str_end2)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()



if __name__ == '__main__':
    main()