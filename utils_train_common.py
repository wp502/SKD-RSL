import argparse
import logging
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from dataloader import fetch_dataloader_1, fetch_dataloader_2, fetch_dataloader_split
from zoo import *

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--pin_memory', action='store_false', default=True, help="flag for whether use pin_memory in dataloader")
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float,
                        default=0.1, help='initial learning rate for Student and AE optimizer')
    parser.add_argument('--learning_rate_t', type=float,
                        default=5e-3, help='initial learning rate for Teacher in Feedback process')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='60,120,160', help='where to decay lr, can be a list')
    # parser.add_argument('--add_param_group_epochs', type=str,
    #                     default='60,120', help='when to add_param_group, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    # parser.add_argument('--lr_decay_paramgroup', type=float,
    #                     default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # parser.add_argument('--params_group_method', type=str, default='gradually')
    parser.add_argument('--scheduler_method', choices=['CosineAnnealingLR', 'MultiStepLR'], type=str, default='MultiStepLR')
    parser.add_argument('--CosineSch_Tmax', type=int, default=200, help="the 'T_max' parameter in CosineAnnealingLR Scheduler")
    parser.add_argument('--add_param_groups_method', choices=['1', '2', '3', '4', '5', '6', 'all', 'all_fc'], type=str, default='all', help="use for 'select_parameters_group' function to select Teacher model need to update part")
    parser.add_argument('--optimizer_method', choices=['sgd', 'adam'], type=str, default='sgd')

    parser.add_argument('--seed', type=int, default=2022403, help='random seed')

    # model
    parser.add_argument('--method', type=str, default='1', help='Train metohd, Data split method')
    parser.add_argument('--model_s', type=str, default='ResNet18_another')
    parser.add_argument('--model_t', type=str, default='ResNet18_another')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--distill', type=str, default='our')
    parser.add_argument('--feedback_time', type=int, default=0)

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

    parser.add_argument('--fusion_method', type=str, default='AUXCF',
                        choices=['Linear', 'Conv', 'AUXCF'],
                        help='which method (Linear or Conv or AUXCF) used for Interlayer knowledge fusion')
    parser.add_argument('--fusion_size', type=str, default='littleSmall',
                        choices=['Big', 'Mean', 'littleSmall', 'Small', 'largeSmall', 'hugeSmall', 'numClass', 'ADP'],
                        help='[for 111] which method for AE FUSION FEATURE SIZE, Mean is base')
    parser.add_argument('--fusion_method_AUXCF', type=str, default='AEConv',
                        choices=['AELinear', 'AEConv', 'Mean', 'LinearSingle', 'AEConv3x3Single', 'AEConv3x3Linear', 'ADPAELinear', 'ADPAEConv3x3Linear', 'ADPAEConv1x1Linear'],
                        help='which method (Linear or Conv or Mean) used for Interlayer knowledge fusion')
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
    parser.add_argument('--in_method', type=str, default='inAuxCF_logitsNToAE_OnlyOne-5',
                        choices=['Channel_MEAN', 'Attention', 'MEAN_STD', 'MEAN_STD_KDCL', 'MEAN_STD_allAE', 'AT_allAE',
                                 'CHMEAN_allAE', 'SELF_AT_HW', 'SELF_AT_C', 'MEAN_STD_FusionTS', 'MEAN_STD_aeFeatCF',
                                 'inAuxCF_logitsToAE', 'inAuxCF_logitsNToAE', 'inAuxCF_logitsNToAE_befLie',
                                 'inAuxCF_logitsNToAE_SgSw', 'inAuxCF_logitsNToAE_OnlyOne',
                                 'inAuxCF_logitsNToAE_OnlyOne-1', 'inAuxCF_logitsNToAE_OnlyOne-2',
                                 'inAuxCF_logitsNToAE_OnlyOne-4', 'inAuxCF_logitsNToAE_OnlyOne-5',
                                 'inAuxCF_logitsNToAE_BtF'],
                        help='which method (Channel or Attention) used for reduce the dimensions of Interlayer')
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
    parser.add_argument('--in_useOther', action='store_true', default=False, help="flag for whether use Softmax or Normalize for normalizing Interlayer")
    parser.add_argument('--fwd_in_TtoS', action='store_false', default=True,
                        help="In the process of feature transfer of middle layer, is the size of teacher compatible with students or the size of students expanding to teachers?")
    parser.add_argument('--blocks_amount', type=int, default=4, choices=[1, 2, 3, 4],
                        help='choose how many Interlayer Block to Distill, must >= 1 ')
    parser.add_argument('--blocks_amount_t', type=int, default=4, choices=[1, 2, 3, 4],
                        help='[for 111] choose how many Interlayer Block of Teacher Network to Distill, must >= 1 ')
    parser.add_argument('--blocks_amount_s', type=int, default=4, choices=[1, 2, 3, 4],
                        help='[for 111] choose how many Interlayer Block of Student Network to Distill, must >= 1 ')
    parser.add_argument('--auxCF_blocks_amount', type=int, default=4, choices=[1, 2, 3, 4],
                        help='[NO USE] choose how many Intermedia layer need aux classification block')
    parser.add_argument('--auxCFAmount', type=int, default=1, help='how many block for aux classificer')


    # resume
    parser.add_argument('--is_resume', action='store_true', default=False, help="flag for whether Resume training")
    parser.add_argument('--checkpoint_t', default='./save/distill_our/', help="teacher model resume checkpoint path")
    parser.add_argument('--checkpoint_t_best', default='./save/distill_our/', help="[NO USE]teacher best model resume checkpoint path")
    parser.add_argument('--checkpoint_s', default='./save/distill_our/', help="student model resume checkpoint path")
    parser.add_argument('--is_warmup', action='store_true', default=False, help="flag for whether warmup")

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10', 'tiny_imagenet', 'cub200', 'cars196', 'imagenet', 'caltech256', 'food101', 'flowers102'],
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
    parser.add_argument('--fb_set_percent', type=float,
                        default=0.3, help='[For args.method==3]feedback data percent')
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

    # parser.add_argument('--is_use_fwd_cka', action='store_false', default=True,
    #                     help="flag for whether use CKA method to Auxiliary training in Forward Knowledge Distillation Process")
    # parser.add_argument('--is_use_fdbk_cka', action='store_false', default=True,
    #                     help="flag for whether use CKA method to Auxiliary training in Feedback Knowledge Distillation Process")
    # parser.add_argument('--is_100cka', action='store_false', default=True,
    #                     help="flag for whether use 100 * CKA")
    parser.add_argument('--metric_method', type=str, default='acc', help='metric method')

    args = parser.parse_args()

    args.model_path = './save/distill_our'
    args.model_path += '/' + str(args.dataset)
    # args.tb_path = os.path.join(args.model_path, 'tensorboard')

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # add_param_group_epoch = args.add_param_group_epochs.split(',')
    # args.add_param_group_epoch = list([])
    # for it_epoch in add_param_group_epoch:
    #     args.add_param_group_epoch.append(int(it_epoch))

    # args.model_t = utils.get_teacher_name(args.path_t)
    # if args.distill == 'sokd':
    #     args.model_t = utils.get_teacher_name(args.path_t)

    args.model_name = "our_"
    args.model_name += 'S-{}_T-{}_{}_{}_{}'.format(args.model_s, args.model_t,
                                                   args.dataset,
                                                   args.scheduler_method, args.trial)

    args.save_folder = os.path.join(args.model_path, args.model_name)
    # args.save_folder = args.save_folder.replace('\\', '/')  # WINDOWS
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.is_checkpoint:
        args.checkpoint_save_pth = os.path.join(args.save_folder, 'checkpoint')
        if not os.path.isdir(args.checkpoint_save_pth):
            os.makedirs(args.checkpoint_save_pth)

    args.loss_txt_name = 'loss_' + str(args.trial) + '.txt'
    # args.cka_fwd_csv_name = args.trial + '-FWD_cka_score.csv'
    # args.cka_fdbk_csv_name = args.trial + '-FDBK_cka_score.csv'

    # args.tb_folder = os.path.join(args.tb_path, args.model_name)
    args.tb_folder = os.path.join(args.save_folder, str(args.trial) + '_tensorboard')
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    argsDict = args.__dict__
    with open(os.path.join(args.save_folder, 'config.txt'), 'w', encoding='utf-8') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    return args

def load_teacher(args):
    logging.info('==> loading teacher model')
    # model_t = get_teacher_name(model_path)
    # model = model_dict[model_t](num_classes=n_cls)
    args.module_dict['model_t'].load_state_dict(torch.load(args.path_t)['state_dict'])

    logging.info('==> done')
    # return model_t

def select_parameters_group(args):
    if 'ResNet' in args.model_t:
        args.param_groups_list = [
                {'params': args.trainable_fb_dict['opt_model_t'].layer4.parameters(), 'lr': args.learning_rate_t / 10},
                {'params': args.trainable_fb_dict['opt_model_t'].linear.parameters(), 'lr': args.learning_rate_t / 10}
            ]
    if args.add_param_groups_method == '1':
        # Best2
        logging.info('Add Param Groups Method 1')
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer2.parameters(), 'lr': args.learning_rate_t / 20})
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer3.parameters(), 'lr': args.learning_rate_t / 20})
    elif args.add_param_groups_method == '12':
        logging.info('Add Param Groups Method 12')
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer2.parameters(), 'lr': args.learning_rate_t / 10})
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer3.parameters(), 'lr': args.learning_rate_t / 10})
    elif args.add_param_groups_method == '2':
        # Best1
        logging.info('Add Param Groups Method 2')
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer3.parameters(), 'lr': args.learning_rate_t / 20})
    elif args.add_param_groups_method == '3':
        logging.info('Add Param Groups Method 3')
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer3.parameters(), 'lr': args.learning_rate_t / 10})
    elif args.add_param_groups_method == '4':
        logging.info('Add Param Groups Method 4')
        pass
    elif args.add_param_groups_method == '5':
        logging.info('Add Param Groups Method 5')
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer1.parameters(), 'lr': args.learning_rate_t / 20})
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer2.parameters(), 'lr': args.learning_rate_t / 20})
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer3.parameters(), 'lr': args.learning_rate_t / 20})
    elif args.add_param_groups_method == '6':
        logging.info('Add Param Groups Method 6')
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer1.parameters(), 'lr': args.learning_rate_t / 30})
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer2.parameters(), 'lr': args.learning_rate_t / 20})
        args.param_groups_list.append(
            {'params': args.trainable_fb_dict['opt_model_t'].layer3.parameters(), 'lr': args.learning_rate_t / 20})
    elif args.add_param_groups_method == 'all':
        logging.info('Add Param Groups Method ALL')
        args.param_groups_list = args.trainable_fb_dict.parameters()
    elif args.add_param_groups_method == 'all_fc':
        logging.info('Add Param Groups Method ALL FC')
        args.param_groups_list = []
        params_front = [param for name, param in args.trainable_fb_dict['opt_model_t'].named_parameters()
                     if name not in ["linear.weight", "linear.bias"]]
        args.param_groups_list.append({'params': params_front, 'lr': args.learning_rate_t})
        args.param_groups_list.append({'params': args.trainable_fb_dict['opt_model_t'].linear.parameters(), 'lr': args.learning_rate_t * 10})
    else:
        logging.info("Add Param Groups ERROR")
        exit()

def select_scheduler(optimizer, args):
    if args.scheduler_method == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    elif args.scheduler_method == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.CosineSch_Tmax)

    return scheduler

def select_random_data(args):

    if args.dataset == 'cifar100' or args.dataset == 'cifar10':
        data = torch.randn(2, 3, 32, 32).cuda()
    else:
        data = torch.randn(2, 3, 224, 224).cuda()

    logging.info("Data shape: " + str(data.shape))

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

def tsInShape(args):
    """ 针对不同的方法，计算AE的输入尺寸和特征尺寸大小 """
    if args.fusion_method == 'Conv':
        if 'MEAN_STD' in args.in_method:
            args.t_shape = sum(each_t[1] for each_t in args.t_dim)
            args.s_shape = sum(each_s[1] for each_s in args.s_dim)
        elif 'AT' in args.in_method:
            args.t_shape = sum(each_t[2] * each_t[3] for each_t in args.t_dim)
            args.s_shape = sum(each_s[2] * each_s[3] for each_s in args.s_dim)
        elif 'CHMEAN' in args.in_method:
            args.t_shape = sum(each_t[1] for each_t in args.t_dim)
            args.s_shape = sum(each_s[1] for each_s in args.s_dim)
    elif args.fusion_method == 'AUXCF':
        if 'logitsNToAE' in args.in_method:
            if 'befLie' in args.in_method:
                args.t_shape = sum(each_t[1] for each_t in args.t_dim)
                args.s_shape = sum(each_s[1] for each_s in args.s_dim)
            elif 'SgSw' in args.in_method:
                pass
    elif args.fusion_method == 'Linear':
        if args.in_method == 'Channel_MEAN':
            args.t_shape = sum(each_t[1] for each_t in args.t_dim)
            args.s_shape = sum(each_s[1] for each_s in args.s_dim)
        elif args.in_method == 'Attention':
            args.t_shape = sum(each_t[2] * each_t[3] for each_t in args.t_dim)
            args.s_shape = sum(each_s[2] * each_s[3] for each_s in args.s_dim)
        elif args.in_method == 'MEAN_STD':
            args.t_shape = sum(each_t[1] * 2 for each_t in args.t_dim)
            args.s_shape = sum(each_s[1] * 2 for each_s in args.s_dim)
        elif args.in_method == 'SELF_AT_HW':
            args.t_shape = sum((each_t[2] * each_t[3]) ** 2 for each_t in args.t_dim)
            args.s_shape = sum((each_s[2] * each_s[3]) ** 2 for each_s in args.s_dim)
        elif args.in_method == 'SELF_AT_C':
            args.t_shape = sum(each_t[1] ** 2 for each_t in args.t_dim)
            args.s_shape = sum(each_s[1] ** 2 for each_s in args.s_dim)


def choiceFusionMethod(args):
    if args.fusion_method == 'Linear':
        # 获得需要更新参数的项
        criterion_inter_fwd = Our_FWD(args.kd_T, args).cuda()
        criterion_inter_fb = Our_FB(args.kd_T, args).cuda()

    elif args.fusion_method == 'Conv':
        # if args.in_method == 'MEAN_STD':
        #     args.t_shape = sum(each_t[1] for each_t in args.t_dim)
        #     args.s_shape = sum(each_s[1] for each_s in args.s_dim)
        if args.in_method == 'MEAN_STD':
            criterion_inter_fwd = Our_FWD_Conv_12(args.kd_T, args).cuda()
            criterion_inter_fb = Our_FB_Conv_12(args.kd_T, args).cuda()
        elif args.in_method == 'MEAN_STD_KDCL':
            criterion_inter_fwd = Our_FWD_Conv_122(args.kd_T, args).cuda()
            criterion_inter_fb = Our_FB_Conv_122(args.kd_T, args).cuda()
        elif 'aeFeatCF' in args.in_method:
            criterion_inter_fwd = Our_FWD_Conv_15(args.kd_T, args).cuda()
            criterion_inter_fb = Our_FB_Conv_15(args.kd_T, args).cuda()
        elif 'allAE' in args.in_method:
            criterion_inter_fwd = Our_FWD_Conv_13(args.kd_T, args).cuda()
            criterion_inter_fb = Our_FB_Conv_13(args.kd_T, args).cuda()
        elif 'FusionTS' in args.in_method:
            pass
    elif args.fusion_method == 'AUXCF':
        if 'logitsToAE' in args.in_method:
            criterion_inter_fwd = Our_FWD_Conv_16(args.kd_T, args).cuda()
            criterion_inter_fb = Our_FB_Conv_16(args.kd_T, args).cuda()
        elif 'logitsNToAE' in args.in_method:
            if 'befLie' in args.in_method:
                criterion_inter_fwd = Our_FWD_Conv_16_3(args.kd_T, args).cuda()
                criterion_inter_fb = Our_FB_Conv_16_3(args.kd_T, args).cuda()
            elif 'SgSw' in args.in_method:
                criterion_inter_fwd = Our_FWD_Conv_18(args.kd_T, args).cuda()
                criterion_inter_fb = Our_FB_Conv_18(args.kd_T, args).cuda()
            elif 'OnlyOne' in args.in_method:
                if '-1' in args.in_method:
                    criterion_inter_fwd = Our_FWD_Conv_19_1(args.kd_T, args).cuda()
                    criterion_inter_fb = Our_FB_Conv_19_1(args.kd_T, args).cuda()
                elif '-2' in args.in_method:
                    criterion_inter_fwd = Our_FWD_Conv_19_2(args.kd_T, args).cuda()
                    criterion_inter_fb = Our_FB_Conv_19_2(args.kd_T, args).cuda()
                elif '-4' in args.in_method:
                    criterion_inter_fwd = Our_FWD_Conv_19_4(args.kd_T, args).cuda()
                    criterion_inter_fb = Our_FB_Conv_19_4(args.kd_T, args).cuda()
                elif '-5' in args.in_method:
                    criterion_inter_fwd = Our_FWD_Conv_111(args.kd_T, args).cuda()
                    criterion_inter_fb = Our_FB_Conv_111(args.kd_T, args).cuda()
                else:
                    criterion_inter_fwd = Our_FWD_Conv_19(args.kd_T, args).cuda()
                    criterion_inter_fb = Our_FB_Conv_19(args.kd_T, args).cuda()
            elif 'BtF' in args.in_method:
                criterion_inter_fwd = Our_FWD_Conv_110(args.kd_T, args).cuda()
                criterion_inter_fb = Our_FB_Conv_110(args.kd_T, args).cuda()
            else:
                criterion_inter_fwd = Our_FWD_Conv_16_2(args.kd_T, args).cuda()
                criterion_inter_fb = Our_FB_Conv_16_2(args.kd_T, args).cuda()
        # elif args.in_method == 'MEAN_STD_allAE':
        #     criterion_inter_fwd = Our_FWD_Conv_13(args.kd_T, args).cuda()
        #     criterion_inter_fb = Our_FB_Conv_13(args.kd_T, args).cuda()

    return criterion_inter_fwd, criterion_inter_fb

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