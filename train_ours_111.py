from json.tool import main
from pickletools import optimize
from sched import scheduler
from unicodedata import name
import utils
import utils_train_common
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
from zoo import DistillKL, initAUXCFAndAE111
from tensorboardX import SummaryWriter
import csv
import torchinfo
import tensorboard_logger
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import time
import datetime

# 考虑将自编码器模块单独放入一个optimizer中，并用recon loss单独更新


def resume_model(args):
    # T
    checkpoint_t = torch.load(args.checkpoint_t)
    args.module_dict['model_t'].load_state_dict(checkpoint_t['state_dict'])
    args.module_auxcfae_t_dict.load_state_dict(checkpoint_t['auxcfae_dict'])
    args.optimizer_dict['opt_fb'].load_state_dict(checkpoint_t['optim_dict'])
    args.scheduler_dict['sch_fb'].load_state_dict(
        checkpoint_t['scheduler_dict'])
    args.start_epoch = checkpoint_t['epoch'] + 1
    args.best_val_acc_top1['bestAcc_model_t'] = checkpoint_t['best_acc']
    args.optimizer_dict['opt_ae_t'].load_state_dict(
        checkpoint_t['optimizer_ae_dict'])
    args.scheduler_dict['sch_ae_t'].load_state_dict(
        checkpoint_t['scheduler_ae_dict'])
    args.optimizer_dict['opt_auxcf_t'].load_state_dict(
        checkpoint_t['optimizer_auxcf_dict'])
    args.scheduler_dict['sch_auxcf_t'].load_state_dict(
        checkpoint_t['scheduler_auxcf_dict'])

    # S
    checkpoint_s = torch.load(args.checkpoint_s)
    args.module_dict['model_s'].load_state_dict(checkpoint_s['state_dict'])
    args.module_auxcfae_s_dict.load_state_dict(checkpoint_s['auxcfae_dict'])
    args.optimizer_dict['opt_fwd'].load_state_dict(checkpoint_s['optim_dict'])
    args.scheduler_dict['sch_fwd'].load_state_dict(
        checkpoint_s['scheduler_dict'])
    args.best_val_acc_top1['bestAcc_model_s'] = checkpoint_s['best_acc']
    args.optimizer_dict['opt_ae_s'].load_state_dict(
        checkpoint_s['optimizer_ae_dict'])
    args.scheduler_dict['sch_ae_s'].load_state_dict(
        checkpoint_s['scheduler_ae_dict'])
    args.optimizer_dict['opt_auxcf_s'].load_state_dict(
        checkpoint_s['optimizer_auxcf_dict'])
    args.scheduler_dict['sch_auxcf_s'].load_state_dict(
        checkpoint_s['scheduler_auxcf_dict'])


def select_optimizer(args):
    """TODO
    考虑教师的更新是否需要用微调的方式进行"""
    args.optimizer_dict = {}
    args.warmup_scheduler_dict = {}
    utils_train_common.select_parameters_group(args)
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
        optimizer_fb = optim.SGD(args.param_groups_list,
                                 lr=args.learning_rate_t, momentum=args.momentum, weight_decay=args.weight_decay)
        # optimizer_fb_ae = optim.SGD(args.trainable_ae_fb_dict.parameters(), lr=args.learning_rate,
        #                              momentum=args.momentum,
        #                              weight_decay=args.weight_decay)

    elif args.optimizer_method == 'adam':
        optimizer_ae_t = optim.Adam(
            args.trainable_ae_t_dict.parameters(), lr=args.learning_rate)
        optimizer_ae_s = optim.Adam(
            args.trainable_ae_s_dict.parameters(), lr=args.learning_rate)
        optimizer_auxcf_t = optim.Adam(
            args.trainable_auxcf_t_dict.parameters(), lr=args.learning_rate)
        optimizer_auxcf_s = optim.Adam(
            args.trainable_auxcf_s_dict.parameters(), lr=args.learning_rate)
        optimizer_fwd = optim.Adam(
            args.trainable_fwd_dict.parameters(), lr=args.learning_rate)
        optimizer_fb = optim.Adam(
            args.param_groups_list, lr=args.learning_rate_t)
        # optimizer_fb_ae = optim.Adam(args.trainable_ae_fb_dict.parameters(), lr=args.learning_rate)

    # optimizer_fb.add_param_group(
    #     {'params': args.trainable_fb_dict['opt_fb_t'].parameters(), 'lr': args.learning_rate})

    args.optimizer_dict.update({'opt_fwd': optimizer_fwd})
    args.optimizer_dict.update({'opt_ae_t': optimizer_ae_t})
    args.optimizer_dict.update({'opt_ae_s': optimizer_ae_s})
    args.optimizer_dict.update({'opt_auxcf_t': optimizer_auxcf_t})
    args.optimizer_dict.update({'opt_auxcf_s': optimizer_auxcf_s})
    args.optimizer_dict.update({'opt_fb': optimizer_fb})
    # args.optimizer_dict.update({'opt_fb_ae': optimizer_fb_ae})


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
    # args.trainable_ae_fwd_dict = nn.ModuleDict({})  # forward 中需要更新的其他项，自编码器项

    data = utils_train_common.select_random_data(args)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    args.module_dict.update({'model_t': model_t})
    args.module_dict.update({'model_s': model_s})

    args.trainable_fb_dict.update({'opt_model_t': model_t})
    args.trainable_fwd_dict.update({'opt_model_s': model_s})

    criterion_cls = nn.CrossEntropyLoss().cuda()
    criterion_div = DistillKL(args.kd_T).cuda()

    args.t_dim = []
    args.s_dim = []
    utils_train_common.select_interlayer111(
        args.model_t, feat_t, args.t_dim, args.blocks_amount_t, args)
    utils_train_common.select_interlayer111(
        args.model_s, feat_s, args.s_dim, args.blocks_amount_s, args)

    # 获得 教师 和 学生 中间特征层 展开后的大小
    utils_train_common.tsInShape(args)

    # 获得需要更新参数的项
    # if args.fusion_method == 'Linear':
    #     # 获得需要更新参数的项
    #     criterion_inter_fwd = Our_FWD(args.kd_T, args).cuda()
    #     criterion_inter_fb = Our_FB(args.kd_T, args).cuda()
    #
    # elif args.fusion_method == 'Conv':
    #     if args.in_method == 'MEAN_STD':
    #         args.t_shape = sum(each_t[1] for each_t in args.t_dim)
    #         args.s_shape = sum(each_s[1] for each_s in args.s_dim)
    #     criterion_inter_fwd = Our_FWD_Conv_12(args.kd_T, args).cuda()
    #     criterion_inter_fb = Our_FB_Conv_12(args.kd_T, args).cuda()
    criterion_inter_fwd, criterion_inter_fb = utils_train_common.choiceFusionMethod(
        args)
    initAUXCFAndAE111(args)

    # args.trainable_fwd_dict.update({'opt_fwd_t': criterion_inter_fwd.fwd_t})
    # args.trainable_ae_fwd_dict.update({'opt_fwd_ae_T': criterion_inter_fwd.fwd_ae_t})
    # args.trainable_ae_fwd_dict.update({'opt_fwd_ae_S': criterion_inter_fwd.fwd_ae_s})
    # args.trainable_fwd_dict.update({'opt_fwd_auxCF': criterion_inter_fwd.fwd_auxcf})
    args.module_auxcfae_t_dict.update({'model_ae_t': args.ae_t})
    args.module_auxcfae_s_dict.update({'model_ae_s': args.ae_s})

    args.trainable_ae_t_dict.update({'opt_ae_t': args.ae_t})
    args.trainable_ae_s_dict.update({'opt_ae_s': args.ae_s})
    for i in range(args.blocks_amount_s):
        args.module_auxcfae_s_dict.update(
            {'model_auxCF_s_b'+str(i+1): args.auxCF_s[str(i+1)]})
        args.trainable_auxcf_s_dict.update(
            {'opt_auxCF_s_b'+str(i+1): args.auxCF_s[str(i+1)]})
        # args.trainable_fwd_dict.update({'opt_fwd_auxCF_s_b'+str(i+1): criterion_inter_fwd.fwd_auxCF_s[str(i+1)]})

    # args.trainable_ae_fb_dict.update({'opt_fb_ae_T': criterion_inter_fb.fb_ae_t})
    # args.trainable_ae_fb_dict.update({'opt_fb_ae_S': criterion_inter_fb.fb_ae_s})
    # args.trainable_fb_dict.update({'opt_ae_t': args.ae_t})
    # args.trainable_fb_dict.update({'opt_ae_s': args.ae_s})
    for i in range(args.blocks_amount_t):
        args.module_auxcfae_t_dict.update(
            {'model_auxCF_t_b' + str(i + 1): args.auxCF_t[str(i + 1)]})
        args.trainable_auxcf_t_dict.update(
            {'opt_auxCF_t_b'+str(i+1): args.auxCF_t[str(i+1)]})
        # args.trainable_fb_dict.update({'opt_fb_auxCF_s_b'+str(i+1): criterion_inter_fb.fb_auxCF_s[str(i+1)]})
    # args.trainable_fb_dict.update({'opt_fb_auxCF': criterion_inter_fb.fb_auxcf})
    # args.trainable_fb_dict.update({'opt_fb_s': criterion_inter_fb.fb_s})

    # args.trainable_ae_fb_dict.update({'opt_fb_ae_S': criterion_inter_fb.fb_s})

    args.criterion_dict.update({'cri_cls': criterion_cls})
    args.criterion_dict.update({'cri_div': criterion_div})
    args.criterion_dict.update({'cri_infwd': criterion_inter_fwd})
    args.criterion_dict.update({'cri_infb': criterion_inter_fb})


def train(args, train_dl, dev_dl):
    # args.module_dict.cuda()
    # args.criterion_dict.cuda()
    # args.trainable_fb_dict.cuda()
    # args.trainable_fwd_dict.cuda()
    # args.scheduler_dict = {}
    # args.optimizer_dict = {}

    args.loss_csv_fwd_name = 'loss_csv_' + str(args.trial) + '_fwd.csv'
    args.loss_csv_fb_name = 'loss_csv_' + str(args.trial) + '_fb.csv'

    csv_title_fwd = ['Epoch_Step', 'SoftTargets_KL']
    csv_title_fb = ['Epoch_Step', 'SoftTargets_KL']
    for i in range(args.blocks_amount_s):
        csv_title_fwd.append('Block'+str(i)+'_CE')

    for i in range(args.blocks_amount_s - 1):
        # deep - shallow
        deep_sender_idx = args.blocks_amount_s - 1 - i
        for each in range(deep_sender_idx):
            csv_title_fwd.append(
                'Block'+str(deep_sender_idx)+'->Block'+str(each)+'_KL')
        # shallow - deep
        shallow_sender_idx = i
        for each in range(i+1, args.blocks_amount_s):
            csv_title_fwd.append(
                'Block'+str(shallow_sender_idx)+'->Block'+str(each)+'_KL')

    for i in range(args.blocks_amount_t):
        csv_title_fb.append('Block'+str(i)+'_CE')

    for i in range(args.blocks_amount_t - 1):
        # deep - shallow
        deep_sender_idx = args.blocks_amount_t - 1 - i
        for each in range(deep_sender_idx):
            csv_title_fb.append(
                'Block'+str(deep_sender_idx)+'->Block'+str(each)+'_KL')
        # shallow - deep
        shallow_sender_idx = i
        for each in range(i+1, args.blocks_amount_t):
            csv_title_fb.append(
                'Block'+str(shallow_sender_idx)+'->Block'+str(each)+'_KL')

    # for i in range(args.blocks_amount_t):
    #     csv_title_fb.append('Block'+str(i)+'_CE')
    #     for j in range(args.blocks_amount_t):
    #         if i is not j:
    #             csv_title_fb.append('Block'+str(i)+'->Block'+str(j)+'_KL')
    csv_title_fwd.extend(['SelfSup_CE_ALL', 'SelfSup_KL_ALL', 'SelfSup_ALL',
                          'reconS_CE', 'fusion_KL', 'AE_ALL', 'Distill_ALL', 'FWD_ALL'])
    csv_title_fb.extend(['SelfSup_CE_ALL', 'SelfSup_KL_ALL', 'SelfSup_ALL', 'reconT_CE', 'fusion_KL',
                         'GradCos_Feedback', 'GradCos_Self', 'GradCos_Ohter', 'AE_ALL', 'Distill_ALL', 'FB_ALL'])
    csv_title_sketch = [
        'T: %s iLR: %s Block_Method: %s, Blocks_Amount: %s, SelfSup_Method: %s, S: %s iLR: %s Block_Method: %s, Blocks_Amount: %s, SelfSup_Method: %s' % (
            str(args.model_t), str(args.learning_rate_t), str(args.aux_method_t), str(
                args.blocks_amount_t), str(args.self_method_t),
            str(args.model_s), str(args.learning_rate), str(args.aux_method_s), str(args.blocks_amount_s), str(args.self_method_s))]

    if args.loss2csv:
        with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_title_sketch)
            writer.writerow(csv_title_fwd)

        with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_title_sketch)
            writer.writerow(csv_title_fb)

    for epoch in range(args.start_epoch, args.epochs):
        # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
        #     f.write("=======================Epoch " + str(epoch + 1) + "=======================" + "\n")

        for scheduler_name_ in args.scheduler_dict.keys():
            args.scheduler_dict[scheduler_name_].step()
        logging.info(
            "Epoch {}/{}, FWD_lr:{}, FB_lr:{}".format(epoch + 1, args.epochs,
                                                      args.optimizer_dict['opt_fwd'].param_groups[0]['lr'],
                                                      args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
        # args.learning_rate_now_t = optimizer_list[0].param_groups[0]['lr']

        # if args.params_group_method == "gradually":
        #     optimizer_list[0] = add_param_groups_gradually(optimizer_list[0], module_list[0], epoch, args)

        logging.info("==> Training...")

        # for each_trainable_fwd_name in args.trainable_fwd_dict.keys():
        #     if 'model' in each_trainable_fwd_name:
        #         continue
        #     else:
        #         args.trainable_fwd_dict[each_trainable_fwd_name].train()
        #
        # for each_trainable_fb_name in args.trainable_fb_dict.keys():
        # for each_trainable_fb_name in args.trainable_fb_dict.keys():
        #     if 'model' in each_trainable_fb_name:
        #         continue
        #     else:
        #         args.trainable_fb_dict[each_trainable_fb_name].train()
        # for each_trainable_fwd_name in args.trainable_ae_fwd_dict.keys():
        #     args.trainable_ae_fwd_dict[each_trainable_fwd_name].train()
        # for each_trainable_t_name in args.trainable_ae_t_dict.keys():
        #     args.trainable_ae_t_dict[each_trainable_t_name].train()
        # for each_trainable_s_name in args.trainable_ae_s_dict.keys():
        #     args.trainable_ae_s_dict[each_trainable_s_name].train()
        # for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
        #     args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].train()
        # for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
        #     args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].train()

        for each in args.module_auxcfae_t_dict.values():
            each.train()
        for each in args.module_auxcfae_s_dict.values():
            each.train()

        if args.method == "1":
            train_acc_list, train_loss_list = train_one_epoch.train_our_distill_111(
                train_dl, epoch, args)
        elif args.method == "2":
            exit()
        elif args.method == "3":
            # exit()
            train_acc_list, train_loss_list = train_one_epoch.train_our_distill_111_3(train_dl, epoch, args)

        # Evaluate for one epoch on validation set
        val_metrics_dict = {}
        val_acc_top1 = {}
        val_acc_top5 = {}
        # is_best = {}
        for idx, model_ in enumerate(args.module_dict.keys()):
            val_metrics_ = train_one_epoch.evaluate(args.module_dict[model_], args.criterion_dict['cri_cls'], dev_dl,
                                                    args,
                                                    metric_method=args.metric_method)
            val_metrics_['epoch'] = epoch + 1
            top1_ = val_metrics_['top1']
            top5_ = val_metrics_['top5']
            val_acc_top1.update({model_: top1_})
            val_acc_top5.update({model_: top5_})
            val_metrics_dict.update({model_: val_metrics_})

            is_best_ = top1_ > args.best_val_acc_top1['bestAcc_'+model_]
            # If best_eval, best_save_path
            if is_best_:
                if 't' in model_:
                    logging.info("- Found Teacher new best accuracy")
                    best_json_path = os.path.join(
                        args.save_folder, "eval_Teacher_best_results.json")
                elif 's' in model_:
                    logging.info("- Found Student new best accuracy")
                    best_json_path = os.path.join(
                        args.save_folder, "eval_Student_best_results.json")
                args.best_val_acc_top1['bestAcc_'+model_] = top1_

                utils.save_dict_to_json(val_metrics_, best_json_path)
            # is_best.update({model_: is_best_})

            # Save latest val metrics in a json file in the model directory
            if 't' in model_:
                last_json_path = os.path.join(
                    args.save_folder, "eval_Teacher_last_results.json")
            elif 's' in model_:
                last_json_path = os.path.join(
                    args.save_folder, "eval_Student_last_results.json")
            utils.save_dict_to_json(val_metrics_, last_json_path)

            if 't' in model_:
                model_auxcfae_now = args.module_auxcfae_t_dict.state_dict()
                optimizer_now = args.optimizer_dict['opt_fb'].state_dict()
                scheduler_now = args.scheduler_dict['sch_fb'].state_dict()
                optimizer_ae_now = args.optimizer_dict['opt_ae_t'].state_dict()
                scheduler_ae_now = args.scheduler_dict['sch_ae_t'].state_dict()
                optimizer_auxcf_now = args.optimizer_dict['opt_auxcf_t'].state_dict(
                )
                scheduler_auxcf_now = args.scheduler_dict['sch_auxcf_t'].state_dict(
                )
            elif 's' in model_:
                model_auxcfae_now = args.module_auxcfae_s_dict.state_dict()
                optimizer_now = args.optimizer_dict['opt_fwd'].state_dict()
                scheduler_now = args.scheduler_dict['sch_fwd'].state_dict()
                optimizer_ae_now = args.optimizer_dict['opt_ae_s'].state_dict()
                scheduler_ae_now = args.scheduler_dict['sch_ae_s'].state_dict()
                optimizer_auxcf_now = args.optimizer_dict['opt_auxcf_s'].state_dict(
                )
                scheduler_auxcf_now = args.scheduler_dict['sch_auxcf_s'].state_dict(
                )

            # Save weights
            model_state_dict = {'name': model_,
                                'epoch': epoch,
                                'acc': top1_,
                                'best_acc': args.best_val_acc_top1['bestAcc_'+model_],
                                'state_dict': args.module_dict[model_].state_dict(),
                                'auxcfae_dict': model_auxcfae_now,
                                'optim_dict': optimizer_now,
                                'optimizer_ae_dict': optimizer_ae_now,
                                'optimizer_auxcf_dict': optimizer_auxcf_now,
                                'scheduler_auxcf_dict': scheduler_auxcf_now,
                                'scheduler_ae_dict': scheduler_ae_now,
                                'scheduler_dict': scheduler_now}
            if 't' in model_:
                utils.save_checkpoint(model_state_dict,
                                      is_best=is_best_, args=args,
                                      save_folder=args.save_folder,
                                      is_teacher=True,
                                      name=args.model_t)
            elif 's' in model_:
                utils.save_checkpoint(model_state_dict,
                                      is_best=is_best_, args=args,
                                      save_folder=args.save_folder,
                                      is_teacher=False,
                                      name=args.model_s)

        if args.loss2csv:
            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([])

            with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([])

        args.tb_logger.add_scalar(
            'Teacher_Train_accuracy', train_acc_list[0], epoch)
        args.tb_logger.add_scalar(
            'Forward_Train_loss', train_loss_list[0], epoch)
        args.tb_logger.add_scalar(
            'Teacher_Test_accuracy_top1', val_acc_top1['model_t'], epoch)
        args.tb_logger.add_scalar(
            'Teacher_Test_accuracy_top5', val_acc_top5['model_t'], epoch)
        args.tb_logger.add_scalar(
            'Teacher_Test_loss', val_metrics_dict['model_t']['loss'], epoch)
        args.tb_logger.add_scalar(
            'Student_Train_accuracy', train_acc_list[1], epoch)
        args.tb_logger.add_scalar(
            'Feedback_Train_loss', train_loss_list[1], epoch)
        args.tb_logger.add_scalar(
            'Student_Test_accuracy_top1', val_acc_top1['model_s'], epoch)
        args.tb_logger.add_scalar(
            'Student_Test_accuracy_top5', val_acc_top5['model_s'], epoch)
        args.tb_logger.add_scalar(
            'Student_Test_loss', val_metrics_dict['model_s']['loss'], epoch)

        torch.cuda.empty_cache()


def main():
    start_time = time.time()
    args = utils_train_common.parse_option()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    warnings.filterwarnings("ignore")

    assert args.blocks_amount_t >= 2 and args.blocks_amount_s >= 2

    # set_logger
    logger_name = args.model_name + ".log"
    utils.set_logger(os.path.join(args.save_folder, logger_name))

    # dataloader
    logging.info("Loading the datasets...")
    train_dl, dev_dl = utils_train_common.fetch_dataloader(args)
    logging.info("- done.")

    # for idx, (fwd_data, fb_data) in enumerate(zip(train_dl['fwd'], train_dl['fb'])):
    #     if idx == 2:
    #         break
    #     str_fwd_data = 'fwd\t' + str(idx) + '\t' + str(fwd_data[0].shape) + '\t' + str(fwd_data[1][:10])
    #     str_fb_data = 'fb\t' + str(idx) + '\t' + str(fb_data[0].shape) + '\t' + str(fb_data[1][:10])
    #     # args.txt_f.write(str_fwd_data + '\n')
    #     # args.txt_f.write(str_fb_data + '\n')
    #     print(str_fwd_data)
    #     print(str_fb_data)

    # for i, data in enumerate(train_dl['fwd']):
    #     if i == 5:
    #         break
    #     print(i, data[0].shape, data[1][:10])

    args.tb_logger = SummaryWriter(log_dir=args.tb_folder)

    # load model frame
    model_t = model_dict[args.model_t](num_classes=args.num_class).cuda()
    model_s = model_dict[args.model_s](num_classes=args.num_class).cuda()

    load_model_str = '==>Our Distill Method:' + 'T: ' + str(args.model_t) + '_S:' + str(
        args.model_s) + '_Dataset:' + str(args.dataset) + '_Trial:' + str(args.trial) + '\n'
    logging.info(load_model_str + args.model_name)

    # something need to train
    sth_needTo_train(model_t, model_s, args)

    # optimizer
    args.iter_per_epoch = len(train_dl)
    select_optimizer(args)

    # scheduler
    args.scheduler_dict = {}
    for optimizer_name_ in args.optimizer_dict.keys():
        scheduler_ = utils_train_common.select_scheduler(
            args.optimizer_dict[optimizer_name_], args)
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

    args.best_val_acc_top1 = dict(bestAcc_model_t=0.0, bestAcc_model_s=0.0)
    # load pre-trained model
    logging.info('==> loading pre-trained model')
    if args.is_resume:
        logging.info('==> Resume Training')
        resume_model(args)
    else:
        args.start_epoch = 0
        utils_train_common.load_teacher(args)

    logging.info('==> done')

    if torch.cuda.is_available():
        args.module_dict.cuda()
        args.module_auxcfae_s_dict.cuda()
        args.module_auxcfae_t_dict.cuda()
        args.criterion_dict.cuda()
        args.trainable_fb_dict.cuda()
        args.trainable_fwd_dict.cuda()
        args.trainable_ae_t_dict.cuda()
        args.trainable_ae_s_dict.cuda()
        args.trainable_auxcf_t_dict.cuda()
        args.trainable_auxcf_s_dict.cuda()
        # args.trainable_ae_fwd_dict.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True

    val_metrics = train_one_epoch.evaluate(args.module_dict['model_t'], args.criterion_dict['cri_cls'], dev_dl, args,
                                           metric_method=args.metric_method)
    logging.info('Teacher accuracy: ' + str(val_metrics['top1']))
    if not args.is_resume:
        args.best_val_acc_top1['bestAcc_model_s'] = 0.0
        args.best_val_acc_top1['bestAcc_model_t'] = val_metrics['top1']

    # args.best_val_acc_top1 = [0.0] * len(args.module_dict)

    # auxnet1 = args.auxCF_t['1']
    # auxnet2 = args.auxCF_t['2']
    # auxnet3 = args.auxCF_t['3']
    # ae_t_net = args.ae_t
    # ae_s_net = args.ae_s
    #
    # torchinfo.summary(ae_t_net, (1, 2048, 1, 1))
    # torchinfo.summary(ae_s_net, (1, 2048, 1, 1))
    #
    # # total1 = 0.0
    # # for each in args.auxCF_t.keys():
    # #     total += sum([param.nelement() for param in args.auxCF_t[each].parameters()])
    # total1 = sum([param.nelement() for param in ae_t_net.parameters()])
    # total2 = sum([param.nelement() for param in ae_s_net.parameters()])
    #
    #
    # print("Parmeter: %.2fM" % (total1 / 1e6))
    # print("Parmeter: %.2fM" % (total2 / 1e6))

    train(args, train_dl, dev_dl)

    args.tb_logger.close()
    logging.info('Teacher:' + str(args.model_t) + '_best accuracy top1: ' +
                 str(args.best_val_acc_top1['bestAcc_model_t']))
    logging.info('Student:' + str(args.model_s) + '_best accuracy top1: ' +
                 str(args.best_val_acc_top1['bestAcc_model_s']))

    end_time = time.time()
    time_consuming = end_time - start_time
    time_consuming = datetime.timedelta(seconds=time_consuming)
    logging.info('Dataset:\t' + str(args.dataset) +
                 '\tTrial:\t' + str(args.trial))
    logging.info('Time Consuming:\t' + str(time_consuming))


if __name__ == '__main__':
    main()
