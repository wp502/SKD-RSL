import os
import time
import math
import utils
from tqdm import tqdm
import logging
from torch.autograd import Variable
import torch
import random
from zoo.CKA import linear_CKA_GPU
import csv
import time
import datetime
import torch.nn.functional as F
# import lpp_test 
# from data_prefetcher import DataPrefetcher
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from evaluate import evaluate, evaluate_kd
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from zoo import get_current_consistency_weight, kl_div, dist_s_label, dist_s_t, hcl
from copy import deepcopy


def train_vanilla_3(model, optimizer, loss_fn, dataloader, scheduler, epoch, args):
    """
    Noraml training, without KD
    """

    # set model to training mode
    model.train()
    # loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # total = 0
    # correct = 0

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            train_batch = Variable(train_batch).cuda()
            labels_batch = Variable(labels_batch).cuda()
            # train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # if epoch<=0:
            #     warmup_scheduler.step()
            # train_batch, labels_batch = Variable(
            #     train_batch), Variable(labels_batch)

            optimizer.zero_grad()
            output_batch = model(train_batch)
            # if args.regularization:
            #     loss = loss_fn(output_batch, labels_batch, params)
            # else:
            #     loss = loss_fn(output_batch, labels_batch)
            loss = loss_fn(output_batch, labels_batch)
            loss.backward()
            optimizer.step()

            # _, predicted = output_batch.max(1)
            # total += labels_batch.size(0)
            # correct += predicted.eq(labels_batch).sum().item()
            acc1, acc5 = utils.accuracy(
                output_batch, labels_batch, topk=(1, 5))
            losses.update(loss.item(), train_batch.size(0))
            top1.update(acc1[0], train_batch.size(0))
            top5.update(acc5[0], train_batch.size(0))

            # update the average loss
            # loss_avg.update(loss.data)
            # losses.update(loss.data, train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(
                losses.avg), top1='{:05.3f}'.format(top1.avg), top5='{:05.3f}'.format(top5.avg),
                lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    # acc = 100. * correct / total
    logging.info("- Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
        top1=top1, top5=top5, loss=losses))

    scheduler.step()

    return top1.avg, losses.avg


def train_vanilla_2(model, optimizer, loss_fn, dataloader, scheduler, epoch, args):
    """
    Noraml training, without KD
    """

    # set model to training mode
    model.train()
    # loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # total = 0
    # correct = 0

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        # iters = len(dataloader.dataset) // args.batch_size
        load_start_time = time.time()
        prefetcher = utils.DataPrefetcher(dataloader)
        train_batch, labels_batch = prefetcher.next()
        iter = 1
        load_end_time = time.time()
        load_time_consuming = load_end_time - load_start_time
        load_time_consuming = datetime.timedelta(seconds=load_time_consuming)
        logging.info('Load step data Time Consuming:\t' + str(load_time_consuming))

        # for i, (train_batch, labels_batch) in enumerate(dataloader):
        while train_batch is not None:
            train_batch = Variable(train_batch).cuda()
            labels_batch = Variable(labels_batch).cuda()
            # train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # if epoch<=0:
            #     warmup_scheduler.step()
            # train_batch, labels_batch = Variable(
            #     train_batch), Variable(labels_batch)

            optimizer.zero_grad()
            output_batch = model(train_batch)
            # if args.regularization:
            #     loss = loss_fn(output_batch, labels_batch, params)
            # else:
            #     loss = loss_fn(output_batch, labels_batch)
            backward_start_time = time.time()
            loss = loss_fn(output_batch, labels_batch)
            loss.backward()
            optimizer.step()
            backward_end_time = time.time()
            backward_time_consuming = backward_end_time - backward_start_time
            backward_time_consuming = datetime.timedelta(seconds=backward_time_consuming)
            logging.info('Backward Time Consuming:\t' + str(backward_time_consuming))

            # _, predicted = output_batch.max(1)
            # total += labels_batch.size(0)
            # correct += predicted.eq(labels_batch).sum().item()
            train_accuracy_start_time = time.time()
            acc1, acc5 = utils.accuracy(
                output_batch, labels_batch, topk=(1, 5))
            losses.update(loss.item(), train_batch.size(0))
            top1.update(acc1[0], train_batch.size(0))
            top5.update(acc5[0], train_batch.size(0))
            train_accuracy_end_time = time.time()
            train_accuracy_time_consuming = train_accuracy_end_time - train_accuracy_start_time
            train_accuracy_time_consuming = datetime.timedelta(seconds=train_accuracy_time_consuming)
            logging.info('train_accuracy Time Consuming:\t' + str(train_accuracy_time_consuming))

            # update the average loss
            # loss_avg.update(loss.data)
            # losses.update(loss.data, train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(
                losses.avg), top1='{:05.3f}'.format(top1.avg), top5='{:05.3f}'.format(top5.avg),
                lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

            next_data_start_time = time.time()
            train_batch, labels_batch = prefetcher.next()
            next_data_end_time = time.time()
            next_data_time_consuming = next_data_end_time - next_data_start_time
            next_data_time_consuming = datetime.timedelta(seconds=next_data_time_consuming)
            logging.info('next_data Time Consuming:\t' + str(next_data_time_consuming))
            # iter += 1

    # acc = 100. * correct / total
    logging.info("- Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
        top1=top1, top5=top5, loss=losses))

    scheduler.step()

    return top1.avg, losses.avg


def train_vanilla(model, optimizer, loss_fn, dataloader, scheduler, epoch, args):
    """
    Noraml training, without KD
    """

    # set model to training mode
    model.train()
    # loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # total = 0
    # correct = 0

    # model_s_weights = []
    # for name, parameters in model.state_dict().items():
    #     if "weight" in name:
    #         model_s_weights.append(parameters)
    #         # print(name, ':', parameters.detach().numpy())
    #
    # model_s_weights = [Variable(each, requires_grad=True) for each in model_s_weights]

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # train_batch = train_batch.float()
            train_batch = Variable(train_batch).cuda()
            labels_batch = Variable(labels_batch).cuda()

            # ===================forward=====================
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            acc1, acc5 = utils.accuracy(
                output_batch, labels_batch, topk=(1, 5))
            losses.update(loss.item(), train_batch.size(0))
            top1.update(acc1[0], train_batch.size(0))
            top5.update(acc5[0], train_batch.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:05.3f}'.format(
                losses.avg), top1='{:05.3f}'.format(top1.avg), top5='{:05.3f}'.format(top5.avg),
                lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    # acc = 100. * correct / total
    logging.info("- Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
        top1=top1, top5=top5, loss=losses))

    scheduler.step()

    return top1.avg, losses.avg


def train_distill(optimizer, module_list, criterion_list, dataloader, epoch, args):
    if args.distill == 'cd':
        args.gamma = utils.adjust_loss_alpha(args.gamma, epoch, args, factor=args.cd_loss_factor)
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    if args.is_cka:
        cka_score_recordcsv_list = []
        cka_score_recordcsv_title = ['Step', 'Block1', 'Block2', 'Block3', 'Block4']
        cka_score_record = []
        for i in range(4):
            cka_score_record.append(utils.AverageMeter())
        with open(os.path.join(args.save_folder, args.cka_score_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch " + str(epoch + 1)])
            writer.writerow(cka_score_recordcsv_title)

        # with open(os.path.join(args.save_folder, args.cka_score_name), 'a') as f:
        #     f.write("=======================Epoch " + str(epoch + 1) + "=======================" + "\n")

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            feat_s, output_s = model_s(train_batch, is_feat=True)

            with torch.no_grad():
                feat_t, output_t = model_t(train_batch, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                # feat_t = Variable(feat_t, requires_grad=False)

            if args.is_cka:
                each_cka_score_recordcsv_list = []
                cka_score_list = cal_contrast_cka(feat_t, feat_s, args)
                each_cka_score_recordcsv_list.append('Step' + str(i + 1))
                # cka_score_record_str = 'Step_' + str(i+1) + ': '
                for each_cka_score, each_record in zip(cka_score_list, cka_score_record):
                    each_record.update(each_cka_score.item(), 1)
                    # cka_score_record_str += str(each_cka_score.item()) + "\t"
                    each_cka_score_recordcsv_list.append(each_cka_score.item())

                # with open(os.path.join(args.save_folder, args.cka_score_name), 'a') as f:
                #     f.write("\t" + cka_score_record_str + "\n")
                cka_score_recordcsv_list.append(each_cka_score_recordcsv_list)

            # cls + kl div
            loss_cls = criterion_cls(output_s, labels_batch)
            if args.distill == 'cd':
                loss_div = criterion_div(output_s, output_t, labels_batch)
            else:
                loss_div = criterion_div(output_s, output_t)

            # other kd beyond KL divergence
            if args.distill == 'kd':
                loss_kd = 0
            elif args.distill == 'ickd':
                g_s = [feat_s[-2]]
                g_t = [feat_t[-2]]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif args.distill == 'ccs':
                # if 'ShuffleV2' in args.model_s:
                #     g_s = feat_s[:-1]
                # else:
                #     g_s = feat_s[1:-1]
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif args.distill == 'spkd':
                # if 'ShuffleV2' in args.model_s:
                #     g_s = feat_s[:-1]
                # else:
                #     g_s = feat_s[1:-1]
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_kd = criterion_kd(g_s, g_t, args)
            elif args.distill == 'cd':
                # if 'ShuffleV2' in args.model_s:
                #     g_s = feat_s[:-1]
                # else:
                #     g_s = feat_s[1:-1]
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_kd = criterion_kd(g_s, g_t)
            elif args.distill == 'crd':
                pass
            elif args.distill == 'fitnet':
                f_s = module_list[1](feat_s[args.hint_layer])
                f_t = feat_t[args.hint_layer]
                loss_kd = criterion_kd(f_s, f_t)
            elif args.distill == 'at':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            else:
                raise NotImplementedError(args.distill)

            loss = args.alpha * loss_cls + args.beta * loss_div + args.gamma * loss_kd

            acc1, acc5 = utils.accuracy(output_s, labels_batch, topk=(1, 5))
            losses.update(loss.item(), train_batch.size(0))
            top1.update(acc1[0], train_batch.size(0))
            top5.update(acc5[0], train_batch.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:05.3f}'.format(
                losses.avg), top1='{:05.3f}'.format(top1.avg), top5='{:05.3f}'.format(top5.avg),
                lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    if args.is_cka:
        # cka_score_record_str = "AVG: "
        # for each_record in cka_score_record:
        #     cka_score_record_str += str(each_record.avg) + "\t"
        # with open(os.path.join(args.save_folder, args.cka_score_name), 'a') as f:
        #     f.write(cka_score_record_str + "\n")

        with open(os.path.join(args.save_folder, args.cka_score_name), 'a', newline='') as f:
            writer = csv.writer(f)
            for each in cka_score_recordcsv_list:
                writer.writerow(each)
            writer.writerow(['AVG:'])
            writer.writerow([''] + [each_record.avg for each_record in cka_score_record])
            # writer.writerow(cka_score_recordcsv_title)

    logging.info("- Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
        top1=top1, top5=top5, loss=losses))
    return top1.avg, losses.avg


def train_distill_reviewkd(optimizer, module_list, criterion_list, dataloader, epoch, args):
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    cnn = module_list[0]
    model_t = module_list[-1]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            cnn.zero_grad()
            losses_all = {}

            s_features, pred = cnn(train_batch)

            with torch.no_grad():
                feat_t, output_t = model_t(train_batch, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                # feat_t = Variable(feat_t, requires_grad=False)

            t_features = feat_t[1:-1]
            # t_features[-1] = t_features[-1].view(t_features[-1].shape[0], t_features[-1].shape[1], 1, 1)
            t_features[-1] = nn.AdaptiveAvgPool2d((1, 1))(t_features[-1])
            feature_kd_loss = hcl(s_features, t_features)
            if args.kd_warm_up_rekd == 0.0:
                losses_all['review_kd_loss'] = feature_kd_loss * args.alpha
            else:
                losses_all['review_kd_loss'] = feature_kd_loss * min(1,
                                                                     epoch / args.kd_warm_up_rekd) * args.alpha  # args.kd_loss_weight
            if args.use_kl_rekd:
                losses_all['kl_kd_loss'] = criterion_div(pred, output_t) * args.beta  # args.kl_loss_weight

            xentropy_loss = criterion_cls(pred, labels_batch)

            losses_all['cls_loss'] = xentropy_loss * args.gamma  # args.ce_loss_weight
            loss = sum(losses_all.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = utils.accuracy(pred, labels_batch, topk=(1, 5))
            losses.update(loss.item(), train_batch.size(0))
            top1.update(acc1[0], train_batch.size(0))
            top5.update(acc5[0], train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(
                losses.avg), top1='{:05.3f}'.format(top1.avg), top5='{:05.3f}'.format(top5.avg),
                lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    logging.info("- Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
        top1=top1, top5=top5, loss=losses))
    return top1.avg, losses.avg


def train_online_distill(optimizer_list, module_list, criterion_list, dataloader, epoch, args):
    # set modules as train()
    losses_list = []
    top1_list = []
    top5_list = []
    for idx, module in enumerate(module_list):
        module.train()
        losses_list.append(utils.AverageMeter())
        top1_list.append(utils.AverageMeter())
        top5_list.append(utils.AverageMeter())

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    # model_s = module_list[0]
    # model_t = module_list[-1]

    if args.is_cka:
        cka_score_recordcsv_list = []
        cka_score_recordcsv_title = ['Step', 'Block1', 'Block2', 'Block3', 'Block4']
        cka_score_record = []
        for i in range(4):
            cka_score_record.append(utils.AverageMeter())
        with open(os.path.join(args.save_folder, args.cka_score_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch " + str(epoch + 1)])
            writer.writerow(cka_score_recordcsv_title)

        # with open(os.path.join(args.save_folder, args.cka_score_name), 'a') as f:
        #     f.write("=======================Epoch " + str(epoch + 1) + "=======================" + "\n")

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            feat_list = []
            output_list = []
            if args.distill == 'kdcl':
                outputs_kdcl = torch.zeros(
                    size=(len(module_list), train_batch.size(0), args.num_class), dtype=torch.float).cuda()

            for idx, model_ in enumerate(module_list):
                feat_, output_ = model_(train_batch, is_feat=True)
                feat_list.append(feat_)
                output_list.append(output_)
                if args.distill == 'kdcl':
                    outputs_kdcl[idx, ...] = output_

            if args.is_cka:
                each_cka_score_recordcsv_list = []
                cka_score_list = cal_contrast_cka(feat_list[0], feat_list[1], args)
                each_cka_score_recordcsv_list.append('Step' + str(i + 1))
                # cka_score_record_str = 'Step_' + str(i+1) + ': '
                for each_cka_score, each_record in zip(cka_score_list, cka_score_record):
                    each_record.update(each_cka_score.item(), 1)
                    # cka_score_record_str += str(each_cka_score.item()) + "\t"
                    each_cka_score_recordcsv_list.append(each_cka_score.item())

                # with open(os.path.join(args.save_folder, args.cka_score_name), 'a') as f:
                #     f.write("\t" + cka_score_record_str + "\n")
                cka_score_recordcsv_list.append(each_cka_score_recordcsv_list)

            if args.distill == 'kdcl':
                stable_out = outputs_kdcl.mean(dim=0)
                stable_out = stable_out.detach()

            if args.distill == 'sokd':
                args.beta = 0.0
                loss_div = 0.0
                if 'WRN' in args.model_t:
                    feat = feat_list[0][-4]
                else:
                    feat = feat_list[0][-3]
                feat_aux, output_aux = module_list[0].auxiliary_forward(feat.detach())
                feat_list.append(feat_aux)
                output_list.append(output_aux)
                loss_kd_T_aux, loss_kd_S_A, loss_kd = criterion_kd(output_list)
                loss_ce_aux = criterion_cls(output_list[2], labels_batch) * args.alpha
                loss_ce = criterion_cls(output_list[1], labels_batch) * args.gamma
                loss_aux = loss_ce_aux + loss_kd_T_aux + loss_kd_S_A
                loss = loss_ce + loss_kd
                loss_total = loss_aux + loss

                # output_list[0]: Teacher
                # output_list[1]: Student
                # output_list[2]: Teacher with AUX
                acc1, acc5 = utils.accuracy(output_list[1], labels_batch, topk=(1, 5))
                acc1_A, acc5_A = utils.accuracy(output_list[2], labels_batch, topk=(1, 5))

                losses_list[0].update(loss_aux.item(), train_batch.size(0))
                top1_list[0].update(acc1_A[0], train_batch.size(0))
                top5_list[0].update(acc5_A[0], train_batch.size(0))

                losses_list[1].update(loss.item(), train_batch.size(0))
                top1_list[1].update(acc1[0], train_batch.size(0))
                top5_list[1].update(acc5[0], train_batch.size(0))

                optimizer_list[0].zero_grad()
                loss_total.backward()
                optimizer_list[0].step()

            else:
                for idx, model_ in enumerate(module_list):
                    # cls + kl div
                    loss_cls = criterion_cls(output_list[idx], labels_batch)
                    # loss_div = criterion_div(output_s, output_t)
                    if args.distill == 'dml':
                        args.beta = 0.0
                        loss_div = 0.0
                        loss_kd = criterion_kd(output_list, idx, len(args.models_list))

                        loss = args.alpha * loss_cls + args.gamma * loss_kd / (len(args.models_list) - 1)
                    elif args.distill == 'kdcl':
                        args.beta = 0.0
                        loss_div = 0.0
                        loss_kd = criterion_kd(output_list, stable_out, args.kd_T, idx)

                        loss = args.alpha * loss_cls + args.gamma * loss_kd
                    else:
                        raise NotImplementedError(args.distill)

                    acc1, acc5 = utils.accuracy(output_list[idx], labels_batch, topk=(1, 5))
                    losses_list[idx].update(loss.item(), train_batch.size(0))
                    top1_list[idx].update(acc1[0], train_batch.size(0))
                    top5_list[idx].update(acc5[0], train_batch.size(0))

                    # ===================backward=====================
                    optimizer_list[idx].zero_grad()
                    loss.backward()
                    optimizer_list[idx].step()
            if args.distill == 'sokd':
                t.set_postfix(loss_Teacher='{:05.3f}'.format(
                    losses_list[0].avg), top1_Teacher='{:05.3f}'.format(top1_list[0].avg),
                    lr_Teacher='{:05.6f}'.format(optimizer_list[0].param_groups[0]['lr']),
                    loss_Student='{:05.3f}'.format(
                        losses_list[1].avg), top1_Student='{:05.3f}'.format(top1_list[1].avg),
                    lr_Student='{:05.6f}'.format(optimizer_list[0].param_groups[1]['lr']))
            else:
                t.set_postfix(loss_M1='{:05.3f}'.format(
                    losses_list[0].avg), top1_M1='{:05.3f}'.format(top1_list[0].avg),
                    top5_M1='{:05.3f}'.format(top5_list[0].avg),
                    lr_M1='{:05.6f}'.format(optimizer_list[0].param_groups[0]['lr']))
            t.update()

    if args.is_cka:
        # cka_score_record_str = "AVG: "
        # for each_record in cka_score_record:
        #     cka_score_record_str += str(each_record.avg) + "\t"
        # with open(os.path.join(args.save_folder, args.cka_score_name), 'a') as f:
        #     f.write(cka_score_record_str + "\n")

        with open(os.path.join(args.save_folder, args.cka_score_name), 'a', newline='') as f:
            writer = csv.writer(f)
            for each in cka_score_recordcsv_list:
                writer.writerow(each)
            writer.writerow(['AVG:'])
            writer.writerow([''] + [each_record.avg for each_record in cka_score_record])
            # writer.writerow(cka_score_recordcsv_title)

    info_str = "- "
    if args.distill == ' ':
        info_str += "Teacher_Branch_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
            top1=top1_list[0], top5=top5_list[0], loss=losses_list[0])
        info_str += "Student_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}".format(
            top1=top1_list[1], top5=top5_list[1])
    else:
        for i in range(len(args.models_list)):
            info_str += "Model" + str(
                i + 1) + "_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
                top1=top1_list[i], top5=top5_list[i], loss=losses_list[i])
    logging.info(info_str)
    return [top1.avg for top1 in top1_list], [losses.avg for losses in losses_list]


def train_online_distill_sokd(optimizer_list, module_list, aux_model, criterion_list, dataloader, epoch, args):                                                                                                                
    # set modules as train()
    losses_list = []
    top1_list = []
    top5_list = []
    module_list[0].eval()
    module_list[1].train()

    # for S
    losses_list.append(utils.AverageMeter())
    top1_list.append(utils.AverageMeter())
    top5_list.append(utils.AverageMeter())

    # for AUX_MODULE
    aux_model.train()
    losses_list.append(utils.AverageMeter())
    top1_list.append(utils.AverageMeter())
    top5_list.append(utils.AverageMeter())

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    # aux_module = aux_model

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            feat_list = []
            output_list = []

            for idx, model_ in enumerate(module_list):
                feat_, output_ = model_(train_batch, is_feat=True)
                feat_list.append(feat_)
                output_list.append(output_)

            if args.distill == 'sokd':
                args.beta = 0.0
                loss_div = 0.0
                if 'WRN' in args.model_t:
                    feat = feat_list[0][-4]
                else:
                    feat = feat_list[0][-3]
                feat_aux, output_aux = aux_model(feat.detach())
                feat_list.append(feat_aux)
                output_list.append(output_aux)
                loss_kd_T_aux, loss_kd_S_A, loss_kd = criterion_kd(output_list)
                loss_ce_aux = criterion_cls(output_list[2], labels_batch) * args.alpha
                loss_ce = criterion_cls(output_list[1], labels_batch) * args.gamma
                loss_aux = loss_ce_aux + loss_kd_T_aux + loss_kd_S_A
                loss = loss_ce + loss_kd
                loss_total = loss_aux + loss

                # output_list[0]: Teacher
                # output_list[1]: Student
                # output_list[2]: Teacher with AUX
                acc1, acc5 = utils.accuracy(output_list[1], labels_batch, topk=(1, 5))
                acc1_A, acc5_A = utils.accuracy(output_list[2], labels_batch, topk=(1, 5))

                losses_list[0].update(loss_aux.item(), train_batch.size(0))
                top1_list[0].update(acc1_A[0], train_batch.size(0))
                top5_list[0].update(acc5_A[0], train_batch.size(0))

                losses_list[1].update(loss.item(), train_batch.size(0))
                top1_list[1].update(acc1[0], train_batch.size(0))
                top5_list[1].update(acc5[0], train_batch.size(0))

                optimizer_list[0].zero_grad()
                loss_total.backward()
                optimizer_list[0].step()

            if args.distill == 'sokd':
                t.set_postfix(loss_TeacherAUX='{:05.3f}'.format(
                    losses_list[0].avg), top1_TeacherAUX='{:05.3f}'.format(top1_list[0].avg),
                    lr_TeacherAUX='{:05.6f}'.format(optimizer_list[0].param_groups[0]['lr']),
                    loss_Student='{:05.3f}'.format(
                        losses_list[1].avg), top1_Student='{:05.3f}'.format(top1_list[1].avg),
                    lr_Student='{:05.6f}'.format(optimizer_list[0].param_groups[1]['lr']))
            t.update()

    info_str = "- "
    if args.distill == 'sokd':
        info_str += "AUX_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
            top1=top1_list[0], top5=top5_list[0], loss=losses_list[0])
        info_str += "Student_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}".format(
            top1=top1_list[1], top5=top5_list[1])

    logging.info(info_str)
    return [top1.avg for top1 in top1_list], [losses.avg for losses in losses_list]


def train_online_distill_switokd(optimizer_list, module_list, criterion_list, dataloader, epoch, args):
    # set modules as train()
    losses_list = []
    top1_list = []
    top5_list = []
    for idx, module in enumerate(module_list):
        module.train()
        losses_list.append(utils.AverageMeter())
        top1_list.append(utils.AverageMeter())
        top5_list.append(utils.AverageMeter())

    criterion_cls = nn.CrossEntropyLoss().cuda()
    criterion_div = kl_div
    # dist_s_label = dist_s_label
    # dist_s_t = dist_s_t

    model = module_list[0]
    model_ks2 = module_list[1]

    optimizer = optimizer_list[0]
    optimizer_ks2 = optimizer_list[1]

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            target_onehot = Variable(
                (torch.zeros(train_batch.size()[0], args.num_class).cuda()).scatter_(1, labels_batch.view(
                    labels_batch.size()[0], 1).cuda(), 1))

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            optimizer.zero_grad()
            output = model(train_batch)

            optimizer_ks2.zero_grad()
            output_ks2 = model_ks2(train_batch)

            s_label = dist_s_label(target_onehot, output_ks2.detach())
            t_label = dist_s_label(target_onehot, output.detach())
            # s_t = dist_s_t(output.detach(), output_ks2.detach(), 1)

            ps_pt = dist_s_t(output.detach(), output_ks2.detach(), 1)

            epsilon = torch.exp(-1 * (t_label / (t_label + s_label)))
            delta = s_label - epsilon * t_label

            # backwarding
            if ps_pt > delta and t_label < s_label:

                loss_full = torch.Tensor([0.0]).cuda()

                loss_ks2 = criterion_cls(output_ks2, labels_batch) + \
                           criterion_div(output.detach(), output_ks2, 1)

                loss_ks2.backward()
                optimizer_ks2.step()

            else:
                loss_full = criterion_cls(output, labels_batch) + \
                            criterion_div(output_ks2.detach(), output, 1)

                loss_ks2 = criterion_cls(output_ks2, labels_batch) + \
                           criterion_div(output.detach(), output_ks2, 1)

                loss_full.backward()
                loss_ks2.backward()

                optimizer.step()
                optimizer_ks2.step()

            acc1, acc5 = utils.accuracy(output, labels_batch, topk=(1, 5))
            losses_list[0].update(loss_full.item(), train_batch.size(0))
            top1_list[0].update(acc1[0], train_batch.size(0))
            top5_list[0].update(acc5[0], train_batch.size(0))

            acc1_2, acc5_2 = utils.accuracy(output_ks2, labels_batch, topk=(1, 5))
            losses_list[1].update(loss_ks2.item(), train_batch.size(0))
            top1_list[1].update(acc1_2[0], train_batch.size(0))
            top5_list[1].update(acc5_2[0], train_batch.size(0))

            t.set_postfix(loss_M1='{:05.3f}'.format(
                losses_list[0].avg), top1_M1='{:05.3f}'.format(top1_list[0].avg),
                top5_M1='{:05.3f}'.format(top5_list[0].avg),
                loss_M2='{:05.3f}'.format(
                    losses_list[1].avg), top1_M2='{:05.3f}'.format(top1_list[1].avg),
                top5_M2='{:05.3f}'.format(top5_list[1].avg))
            t.update()

    info_str = "- "
    for i in range(len(args.models_list)):
        info_str += "Model" + str(
            i + 1) + "_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
            top1=top1_list[i], top5=top5_list[i], loss=losses_list[i])

    logging.info(info_str)
    return [top1.avg for top1 in top1_list], [losses.avg for losses in losses_list]


def train_online_distill_ffl(optimizer_list, module_list, fusion_model, criterion_list, dataloader, epoch, args):
    # set modules as train()
    losses_list = []
    top1_list = []
    top5_list = []
    for idx, module in enumerate(module_list):
        module.train()
        losses_list.append(utils.AverageMeter())
        top1_list.append(utils.AverageMeter())
        top5_list.append(utils.AverageMeter())
    # for FUSION_MODULE
    fusion_model.train()
    losses_list.append(utils.AverageMeter())
    top1_list.append(utils.AverageMeter())
    top5_list.append(utils.AverageMeter())

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_1 = module_list[0]
    model_2 = module_list[1]
    fusion_module = fusion_model

    consistency_weight = get_current_consistency_weight(epoch)
    # print(consistency_weight)

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            feat_1, output_1 = model_1(train_batch, is_feat=True)
            feat_2, output_2 = model_2(train_batch, is_feat=True)

            if args.dataset == 'cifar100' or args.dataset == 'cifar10':
                pre_express = nn.AdaptiveAvgPool2d((4, 4))
            else:
                pre_express = nn.AdaptiveAvgPool2d((7, 7))
            feat_1_fusion = pre_express(feat_1[-2])
            feat_2_fusion = pre_express(feat_2[-2])

            ensemble_logit = (output_1 + output_2) / 2
            fused_logit = fusion_module(feat_1_fusion, feat_2_fusion)

            loss_cross = criterion_cls(output_1, labels_batch) + criterion_cls(output_2, labels_batch) + criterion_cls(
                fused_logit, labels_batch)

            # MKD
            loss_kl = consistency_weight * (
                    criterion_kd(output_1, fused_logit) + criterion_kd(output_2, fused_logit) + criterion_kd(
                fused_logit,
                ensemble_logit))

            loss = loss_cross + loss_kl

            acc1_1, acc5_1 = utils.accuracy(output_1, labels_batch, topk=(1, 5))
            losses_list[0].update(loss_cross.item(), train_batch.size(0))
            top1_list[0].update(acc1_1[0], train_batch.size(0))
            top5_list[0].update(acc5_1[0], train_batch.size(0))

            acc1_2, acc5_2 = utils.accuracy(output_2, labels_batch, topk=(1, 5))
            losses_list[1].update(loss_kl.item(), train_batch.size(0))
            top1_list[1].update(acc1_2[0], train_batch.size(0))
            top5_list[1].update(acc5_2[0], train_batch.size(0))

            acc1_f, acc5_f = utils.accuracy(fused_logit, labels_batch, topk=(1, 5))
            losses_list[2].update(loss.item(), train_batch.size(0))
            top1_list[2].update(acc1_f[0], train_batch.size(0))
            top5_list[2].update(acc5_f[0], train_batch.size(0))

            optimizer_list[0].zero_grad()
            optimizer_list[1].zero_grad()
            optimizer_list[2].zero_grad()
            loss.backward()
            optimizer_list[0].step()
            optimizer_list[1].step()
            optimizer_list[2].step()

            t.set_postfix(loss_CE='{:05.3f}'.format(
                losses_list[0].avg), top1_M1='{:05.3f}'.format(top1_list[0].avg),
                loss_KL='{:05.3f}'.format(
                    losses_list[1].avg), top1_M2='{:05.3f}'.format(top1_list[1].avg),
                loss='{:05.3f}'.format(
                    losses_list[2].avg), top1_FM='{:05.3f}'.format(top1_list[2].avg))
            t.update()

    info_str = "- "
    # for i in range(len(args.models_list)):
    info_str += "Model1_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss_CE:{loss.avg: .4f}".format(
        top1=top1_list[0], top5=top5_list[0], loss=losses_list[0])
    info_str += "  Model2_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss_KL:{loss.avg: .4f}".format(
        top1=top1_list[1], top5=top5_list[1], loss=losses_list[1])
    info_str += "  Model_Fusion_Train accuracy Acc@1:{top1.avg: .4f}, Acc@5:{top5.avg: .4f}, training loss:{loss.avg: .4f}".format(
        top1=top1_list[2], top5=top5_list[2], loss=losses_list[2])
    logging.info(info_str)

    return [top1.avg for top1 in top1_list], [losses.avg for losses in losses_list]


def train_our_distill(optimizer_list, module_list, criterion_list, dataloader, epoch, args):
    # set teacher as eval
    module_list[0].eval()
    # set student as train
    module_list[1].train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_t = module_list[0]
    model_s = module_list[1]

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fdbk = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    if args.is_cka:
        cka_score_recordcsv_fwd_list = []
        cka_score_recordcsv_fdbk_list = []
        cka_score_recordcsv_title = ['Step', 'Block1', 'Block2', 'Block3', 'Block4']
        cka_score_fwd_record = []
        cka_score_fdbk_record = []
        for i in range(4):
            cka_score_fwd_record.append(utils.AverageMeter())
            cka_score_fdbk_record.append(utils.AverageMeter())
        with open(os.path.join(args.save_folder, args.cka_csv_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch " + str(epoch + 1)])
            writer.writerow(cka_score_recordcsv_title)

    with tqdm(total=len(dataloader)) as t:
        list_dataloader = list(dataloader)
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            train_batch_fwd, labels_batch_fwd = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch_fwd, labels_batch_fwd = Variable(train_batch_fwd), Variable(labels_batch_fwd)

            feat_s_fwd, output_s_fwd = model_s(train_batch_fwd, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_batch_fwd, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_batch_fwd)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss
            if 'ResNet' in args.model_t:
                feat_t_fwd_list = feat_t_fwd[1:-1]
            elif 'WRN' in args.model_t:
                feat_t_fwd_list = feat_t_fwd[1:-1]
            if 'ResNet' in args.model_s:
                feat_s_fwd_list = feat_s_fwd[1:-1]
            elif 'WRN' in args.model_s:
                feat_s_fwd_list = feat_s_fwd[1:-1]
            elif 'ShuffleV2' in args.model_s:
                feat_s_fwd_list = feat_s_fwd[:-1]
            elif 'MobileNetV2' in args.model_s:
                feat_s_fwd_list = feat_s_fwd[1:-1]
            loss_kd_fwd, cka_score_fwd_list = criterion_kd(feat_t_fwd_list, feat_s_fwd_list, args)

            if args.is_cka:
                each_cka_score_recordcsv_fwd_list = []
                each_cka_score_recordcsv_fwd_list.append('Step' + str(i + 1))
                for each_cka_fwd_score, each_record_fwd in zip(cka_score_fwd_list, cka_score_fwd_record):
                    each_record_fwd.update(each_cka_fwd_score.item(), 1)
                    each_cka_score_recordcsv_fwd_list.append(each_cka_fwd_score.item())

                cka_score_recordcsv_fwd_list.append(each_cka_score_recordcsv_fwd_list)

            loss_fwd = args.alpha * loss_cls_fwd + args.beta_fwd * loss_div_fwd + args.gamma_fwd * loss_kd_fwd

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                f.write("CH_fwd:" + str(loss_kd_fwd.item()) + '\t')
                f.write("LOSS_total_fws:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_batch_fwd, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_batch_fwd.size(0))
            top1_s.update(acc1_s[0], train_batch_fwd.size(0))
            top5_s.update(acc5_s[0], train_batch_fwd.size(0))

            # Forward distill backward
            optimizer_list[1].zero_grad()
            loss_fwd.backward()
            optimizer_list[1].step()

            # ===================Feedback Distill=====================
            if epoch >= args.feedback_time:
                model_t.train()
                model_s.eval()
                # 反馈用数据集index
                feedback_index = random.randint(0, len(dataloader) - 1)
                train_batch_fdbk, labels_batch_fdbk = list_dataloader[feedback_index]
                train_batch_fdbk, labels_batch_fdbk = train_batch_fdbk.cuda(), labels_batch_fdbk.cuda()
                # convert to torch Variables
                train_batch_fdbk, labels_batch_fdbk = Variable(train_batch_fdbk), Variable(labels_batch_fdbk)

                feat_t_fdbk, output_t_fdbk = model_t(train_batch_fdbk, is_feat=True)

                with torch.no_grad():
                    feat_s_fdbk, output_s_fdbk = model_s(train_batch_fdbk, is_feat=True)
                    output_s_fdbk = output_s_fdbk.detach()
                    feat_s_fdbk = [f.detach() for f in feat_s_fdbk]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fdbk = criterion_cls(output_t_fdbk, labels_batch_fdbk)
                loss_div_fdbk = criterion_div(output_t_fdbk, output_s_fdbk)
                # feedback channel loss
                if 'ResNet' in args.model_t:
                    feat_t_fdbk_list = feat_t_fdbk[1:-1]
                elif 'WRN' in args.model_t:
                    feat_t_fdbk_list = feat_t_fdbk[1:-1]
                if 'ResNet' in args.model_s:
                    feat_s_fdbk_list = feat_s_fdbk[1:-1]
                elif 'WRN' in args.model_s:
                    feat_s_fdbk_list = feat_s_fdbk[1:-1]
                elif 'ShuffleV2' in args.model_s:
                    feat_s_fdbk_list = feat_s_fdbk[:-1]
                elif 'MobileNetV2' in args.model_s:
                    feat_s_fdbk_list = feat_s_fdbk[1:-1]
                loss_kd_fdbk, cka_score_fdbk_list = criterion_kd(feat_t_fdbk_list, feat_s_fdbk_list, args,
                                                                 is_feedback=True)

                if args.is_cka:
                    each_cka_score_recordcsv_fdbk_list = []
                    each_cka_score_recordcsv_fdbk_list.append('Step' + str(i + 1))
                    for each_cka_fdbk_score, each_record_fdbk in zip(cka_score_fdbk_list, cka_score_fdbk_record):
                        each_record_fdbk.update(each_cka_fdbk_score.item(), 1)
                        each_cka_score_recordcsv_fdbk_list.append(each_cka_fdbk_score.item())

                    cka_score_recordcsv_fdbk_list.append(each_cka_score_recordcsv_fdbk_list)

                loss_fdbk = args.alpha * loss_cls_fdbk + args.beta_fdbk * loss_div_fdbk + args.gamma_fdbk * loss_kd_fdbk

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fdbk:" + str(loss_cls_fdbk.item()) + '\t')
                    f.write("KD_fdbk:" + str(loss_div_fdbk.item()) + '\t')
                    f.write("CH_fdbk:" + str(loss_kd_fdbk.item()) + '\t')
                    f.write("LOSS_total_fdbk:" + str(loss_fdbk.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fdbk, labels_batch_fdbk, topk=(1, 5))
                losses_fdbk.update(loss_fdbk.item(), train_batch_fdbk.size(0))
                top1_t.update(acc1_t[0], train_batch_fdbk.size(0))
                top5_t.update(acc5_t[0], train_batch_fdbk.size(0))

                # Feedback distill backward
                optimizer_list[0].zero_grad()
                loss_fdbk.backward()
                optimizer_list[0].step()

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_s='{:05.6f}'.format(optimizer_list[1].param_groups[0]['lr']),
                              loss_fdbk='{:05.3f}'.format(losses_fdbk.avg),
                              lr_t='{:05.6f}'.format(optimizer_list[0].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_s='{:05.6f}'.format(optimizer_list[1].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_batch_fwd, labels_batch_fwd, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_kd_fwd, loss_fwd
                del train_batch_fdbk, labels_batch_fdbk, feat_s_fdbk, output_s_fdbk, feat_t_fdbk, output_t_fdbk, loss_cls_fdbk, loss_div_fdbk, loss_kd_fdbk, loss_fdbk
            else:
                del train_batch_fwd, labels_batch_fwd, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_kd_fwd, loss_fwd

    if args.is_cka:
        with open(os.path.join(args.save_folder, args.cka_csv_name), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Forward CKA Score:'])
            for each_fwd in cka_score_recordcsv_fwd_list:
                writer.writerow(each_fwd)
            writer.writerow(['Forward AVG:'])
            writer.writerow([''] + [each_record_fwd.avg for each_record_fwd in cka_score_fwd_record])

            writer.writerow(['Feedback CKA Score:'])
            for each_fdbk in cka_score_recordcsv_fdbk_list:
                writer.writerow(each_fdbk)
            writer.writerow(['Feedback AVG:'])
            writer.writerow([''] + [each_record_fdbk.avg for each_record_fdbk in cka_score_fdbk_record])

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fdbk))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fdbk.avg]


def train_our_distill_2(train_dl, epoch, args):
    # args.module_dict.cuda()
    # args.criterion_dict.cuda()
    # args.trainable_fb_dict.cuda()
    # args.trainable_fwd_dict.cuda()
    # args.scheduler_dict = {}
    # args.optimizer_dict = {}

    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl)) as t:
        # list_dataloader = list(dataloader)
        # for i, ((train_fwd_batch, labels_fwd_batch), (train_fdbk_batch, labels_fdbk_batch)) in enumerate(zip(train_fwd_dl, train_fdbk_dl)):
        for i, (train_batch, labels_batch) in enumerate(train_dl):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            # 切分数据集，用于 feedback 的数据集为 fb_set_percent
            fb_count = int(args.batch_size * args.fb_set_percent)
            # fwd_count = args.batch_size - fb_count
            train_fwd_batch = train_batch[:-fb_count]
            labels_fwd_batch = labels_batch[:-fb_count]
            train_fb_batch = train_batch[-fb_count:]
            labels_fb_batch = labels_batch[-fb_count:]

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, args)

            loss_fwd = args.ce_loss * loss_cls_fwd + args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
                f.write("LOSS_total_fws:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, args)

                div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb,
                                                                            model_t_weights, args)

                if div_cosin_simility_all < 0:
                    kl_loss = 0.0
                else:
                    kl_loss = args.kl_loss

                if ch_cosin_simility_all < 0:
                    infb_loss = 0.0
                else:
                    infb_loss = args.infb_loss

                loss_fb = args.ce_loss * loss_cls_fb + kl_loss * loss_div_fb + infb_loss * loss_ch_fb

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                    f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                    f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                    f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                    f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                    f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_cls_fb, loss_div_fb, loss_ch_fb, loss_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_1(train_dl_fwd, train_dl_fb, epoch, args):
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            # # 切分数据集，用于 feedback 的数据集为 fb_set_percent
            # fb_count = int(args.batch_size * args.fb_set_percent)
            # # fwd_count = args.batch_size - fb_count
            # train_fwd_batch = train_batch[:-fb_count]
            # labels_fwd_batch = labels_batch[:-fb_count]
            # train_fb_batch = train_batch[-fb_count:]
            # labels_fb_batch = labels_batch[-fb_count:]

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, args)

            loss_fwd = args.ce_loss * loss_cls_fwd + args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
                f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, args)

                div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb,
                                                                            model_t_weights, args)

                if div_cosin_simility_all < 0:
                    kl_loss = 0.0
                else:
                    kl_loss = args.kl_loss

                if ch_cosin_simility_all < 0:
                    infb_loss = 0.0
                else:
                    infb_loss = args.infb_loss

                loss_fb = args.ce_loss * loss_cls_fb + kl_loss * loss_div_fb + infb_loss * loss_ch_fb

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                    f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                    f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                    f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                    f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                    f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_cls_fb, loss_div_fb, loss_ch_fb, loss_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_12(train_dl_fwd, train_dl_fb, epoch, args):
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            # # 切分数据集，用于 feedback 的数据集为 fb_set_percent
            # fb_count = int(args.batch_size * args.fb_set_percent)
            # # fwd_count = args.batch_size - fb_count
            # train_fwd_batch = train_batch[:-fb_count]
            # labels_fwd_batch = labels_batch[:-fb_count]
            # train_fb_batch = train_batch[-fb_count:]
            # labels_fb_batch = labels_batch[-fb_count:]

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, args)

            loss_fwd = args.ce_loss * loss_cls_fwd + args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd[
                'fusion_fwd']

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
                f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))
            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd_ae'].zero_grad()
            loss_ch_fwd['recon_fwd'].backward(retain_graph=True)

            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_fwd_ae'].step()

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, args)

                div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb,
                                                                            loss_ch_fb['fusion_fb'], model_t_weights,
                                                                            args)

                if div_cosin_simility_all < 0:
                    kl_loss = 0.0
                else:
                    kl_loss = args.kl_loss

                if ch_cosin_simility_all < 0:
                    infb_loss = 0.0
                else:
                    infb_loss = args.infb_loss

                loss_fb = args.ce_loss * loss_cls_fb + kl_loss * loss_div_fb + infb_loss * loss_ch_fb['fusion_fb']

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                    f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                    # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                    f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                    f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                    f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                args.optimizer_dict['opt_fb_ae'].zero_grad()
                loss_ch_fb['recon_fb'].backward(retain_graph=True)

                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_fb_ae'].step()

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_cls_fb, loss_div_fb, loss_ch_fb, loss_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_13(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, args)

            loss_fwd_ae = loss_ch_fwd['recon_T_fwd'] + loss_ch_fwd['recon_S_fwd']
            loss_fwd = args.ce_loss * loss_cls_fwd + args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd[
                'fusion_fwd']

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
                f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_fwd_ae'].step()

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, args)

                div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb,
                                                                            loss_ch_fb['fusion_fb'], model_t_weights,
                                                                            args)

                if div_cosin_simility_all < 0:
                    kl_loss = 0.0
                else:
                    kl_loss = args.kl_loss

                if ch_cosin_simility_all < 0:
                    infb_loss = 0.0
                else:
                    infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb'] + loss_ch_fb['recon_S_fb']
                loss_fb = args.ce_loss * loss_cls_fb + kl_loss * loss_div_fb + infb_loss * loss_ch_fb['fusion_fb']

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                    f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                    # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                    f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                    f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                    f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_fb_ae'].step()

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_cls_fb, loss_div_fb, loss_ch_fb, loss_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_14(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_t_aux = args.model_t_aux
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set teacher aux as eval
    model_t_aux.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            model_t_aux.eval()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t_aux(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, args)

            loss_fwd_ae = loss_ch_fwd['recon_T_fwd'] + loss_ch_fwd['recon_S_fwd']
            loss_fwd = args.ce_loss * loss_cls_fwd + args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd[
                'fusion_fwd']

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
                f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_fwd_ae'].step()

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()
                model_t_aux.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, args)

                div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb,
                                                                            loss_ch_fb['fusion_fb'], model_t_weights,
                                                                            args)

                if div_cosin_simility_all < 0:
                    kl_loss = 0.0
                else:
                    kl_loss = args.kl_loss

                if ch_cosin_simility_all < 0:
                    infb_loss = 0.0
                else:
                    infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb'] + loss_ch_fb['recon_S_fb']
                loss_fb = args.ce_loss * loss_cls_fb + kl_loss * loss_div_fb + infb_loss * loss_ch_fb['fusion_fb']

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                    f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                    # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                    f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                    f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                    f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_fb_ae'].step()

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_cls_fb, loss_div_fb, loss_ch_fb, loss_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_15(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_T_fwd'] + loss_ch_fwd['recon_S_fwd']
            loss_fwd = args.ce_loss * loss_cls_fwd + args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd[
                'fusion_fwd'] + loss_ch_fwd['auxCF_fwd']

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
                f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_fwd_ae'].step()

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, labels_fb_batch, args)

                div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb,
                                                                            loss_ch_fb['fusion_fb'], model_t_weights,
                                                                            args)

                if div_cosin_simility_all < 0:
                    kl_loss = 0.0
                else:
                    kl_loss = args.kl_loss

                if ch_cosin_simility_all < 0:
                    infb_loss = 0.0
                else:
                    infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb'] + loss_ch_fb['recon_S_fb']
                loss_fb = args.ce_loss * loss_cls_fb + kl_loss * loss_div_fb + infb_loss * loss_ch_fb['fusion_fb'] + \
                          loss_ch_fb['auxCF_fb']

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                    f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                    # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                    f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                    f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                    f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_fb_ae'].step()

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_cls_fb, loss_div_fb, loss_ch_fb, loss_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_16(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("---------------Forward loss-------------" + "\n")

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            # loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd,
                                              labels_fwd_batch, criterion_cls, criterion_div, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_T_fwd'] + loss_ch_fwd['recon_S_fwd']
            loss_fwd = args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd['self_supervised_fwd']
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
            #     f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
            #     # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
            #     f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
            #     f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_fwd_ae'].step()

            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(args.loss_csv_fwd)

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                # loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                # loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb,
                                                labels_fb_batch, criterion_cls, criterion_div, model_t_weights, args)

                # div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb['fusion_fb'], model_t_weights, args)
                #
                # if div_cosin_simility_all < 0:
                #     kl_loss = 0.0
                # else:
                #     kl_loss = args.kl_loss
                #
                # if ch_cosin_simility_all < 0:
                #     infb_loss = 0.0
                # else:
                #     infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb'] + loss_ch_fb['recon_S_fb']
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_ch_fb['self_supervised_fb']

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb.item())

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                #     f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                #     # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                #     f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                #     f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                #     f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                #     f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_fb_ae'].step()

                with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fb)

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_17(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("---------------Forward loss-------------" + "\n")

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd,
                                              labels_fwd_batch, criterion_cls, criterion_div, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_T_fwd'] + loss_ch_fwd['recon_S_fwd']
            loss_fwd = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd[
                'self_supervised_fwd']
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
            #     f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
            #     # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
            #     f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
            #     f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_fwd_ae'].step()

            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(args.loss_csv_fwd)

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                # loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                args.loss_csv_fb.append(loss_div_fb.item())
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb,
                                                labels_fb_batch, criterion_cls, criterion_div, loss_div_fb,
                                                model_t_weights, args)

                # div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb['fusion_fb'], model_t_weights, args)
                #
                # if div_cosin_simility_all < 0:
                #     kl_loss = 0.0
                # else:
                #     kl_loss = args.kl_loss
                #
                # if ch_cosin_simility_all < 0:
                #     infb_loss = 0.0
                # else:
                #     infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb'] + loss_ch_fb['recon_S_fb']
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_ch_fb['loss_other']

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb.item())

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                #     f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                #     # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                #     f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                #     f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                #     f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                #     f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_fb_ae'].step()

                with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fb)

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb, loss_div_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_19(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd_ae = utils.AverageMeter()
    losses_fwd_distill = utils.AverageMeter()
    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb_ae = utils.AverageMeter()
    losses_fb_distill = utils.AverageMeter()
    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("---------------Forward loss-------------" + "\n")

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd,
                                              labels_fwd_batch, criterion_cls, criterion_div, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_T_fwd'] + loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd[
                'self_supervised_fwd']
            loss_fwd = loss_fwd_distill + loss_fwd_ae
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd_distill.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
            #     f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
            #     # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
            #     f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
            #     f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd_ae.update(loss_fwd_ae.item(), train_fwd_batch.size(0))
            losses_fwd_distill.update(loss_fwd_distill.item(), train_fwd_batch.size(0))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            # args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            # loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            # args.optimizer_dict['opt_fwd_ae'].step()

            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(args.loss_csv_fwd)

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                # loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                args.loss_csv_fb.append(loss_div_fb.item())
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb,
                                                labels_fb_batch, criterion_cls, criterion_div, loss_div_fb,
                                                model_t_weights, args)

                # div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb['fusion_fb'], model_t_weights, args)
                #
                # if div_cosin_simility_all < 0:
                #     kl_loss = 0.0
                # else:
                #     kl_loss = args.kl_loss
                #
                # if ch_cosin_simility_all < 0:
                #     infb_loss = 0.0
                # else:
                #     infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb'] + loss_ch_fb['recon_S_fb']
                loss_fb_distill = loss_ch_fb['loss_other']
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_fb_distill + loss_fb_ae

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb_distill.item())
                args.loss_csv_fb.append(loss_fb.item())

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                #     f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                #     # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                #     f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                #     f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                #     f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                #     f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb_ae.update(loss_fb_ae.item(), train_fb_batch.size(0))
                losses_fb_distill.update(loss_fb_distill.item(), train_fb_batch.size(0))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                # args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                # loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                # args.optimizer_dict['opt_fb_ae'].step()

                with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fb)

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb_ae='{:05.3f}'.format(losses_fb_ae.avg),
                              loss_fb_distill='{:05.3f}'.format(losses_fb_distill.avg),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb, loss_div_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_19_1(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd_ae = utils.AverageMeter()
    losses_fwd_distill = utils.AverageMeter()
    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb_ae = utils.AverageMeter()
    losses_fb_distill = utils.AverageMeter()
    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("---------------Forward loss-------------" + "\n")

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd,
                                              labels_fwd_batch, criterion_cls, criterion_div, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_T_fwd'] + loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd[
                'self_supervised_fwd']
            loss_fwd = loss_fwd_distill + loss_fwd_ae
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd_distill.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
            #     f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
            #     # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
            #     f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
            #     f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd_ae.update(loss_fwd_ae.item(), train_fwd_batch.size(0))
            losses_fwd_distill.update(loss_fwd_distill.item(), train_fwd_batch.size(0))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            # args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            args.optimizer_dict['opt_ae'].zero_grad()
            # loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_ae'].step()
            # args.optimizer_dict['opt_fwd_ae'].step()

            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(args.loss_csv_fwd)

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                # loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                args.loss_csv_fb.append(loss_div_fb.item())
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb,
                                                labels_fb_batch, criterion_cls, criterion_div, loss_div_fb,
                                                model_t_weights, args)

                # div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb['fusion_fb'], model_t_weights, args)
                #
                # if div_cosin_simility_all < 0:
                #     kl_loss = 0.0
                # else:
                #     kl_loss = args.kl_loss
                #
                # if ch_cosin_simility_all < 0:
                #     infb_loss = 0.0
                # else:
                #     infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb'] + loss_ch_fb['recon_S_fb']
                loss_fb_distill = loss_ch_fb['loss_other']
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_fb_distill + loss_fb_ae

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb_distill.item())
                args.loss_csv_fb.append(loss_fb.item())

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                #     f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                #     # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                #     f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                #     f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                #     f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                #     f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb_ae.update(loss_fb_ae.item(), train_fb_batch.size(0))
                losses_fb_distill.update(loss_fb_distill.item(), train_fb_batch.size(0))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                # args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                args.optimizer_dict['opt_ae'].zero_grad()
                # loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_ae'].step()
                # args.optimizer_dict['opt_fb_ae'].step()

                with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fb)

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb_ae='{:05.3f}'.format(losses_fb_ae.avg),
                              loss_fb_distill='{:05.3f}'.format(losses_fb_distill.avg),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb, loss_div_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_19_2(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd_ae = utils.AverageMeter()
    losses_fwd_distill = utils.AverageMeter()
    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb_ae = utils.AverageMeter()
    losses_fb_distill = utils.AverageMeter()
    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            args.trainable_ae_t_dict['opt_ae_t'].eval()
            args.trainable_ae_s_dict['opt_ae_s'].train()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            # feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                # output_t_fwd = output_t_fwd.detach()
                # feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("---------------Forward loss-------------" + "\n")

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd,
                                              labels_fwd_batch, criterion_cls, criterion_div, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd[
                'self_supervised_fwd']
            loss_fwd = loss_fwd_distill + loss_fwd_ae
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd_distill.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
            #     f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
            #     # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
            #     f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
            #     f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd_ae.update(loss_fwd_ae.item(), train_fwd_batch.size(0))
            losses_fwd_distill.update(loss_fwd_distill.item(), train_fwd_batch.size(0))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            # args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            args.optimizer_dict['opt_ae_s'].zero_grad()
            # loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_ae_s'].step()
            # args.optimizer_dict['opt_fwd_ae'].step()

            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(args.loss_csv_fwd)

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()
                args.trainable_ae_t_dict['opt_ae_t'].train()
                args.trainable_ae_s_dict['opt_ae_s'].eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                # feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    # output_s_fb = output_s_fb.detach()
                    # feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                # loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                args.loss_csv_fb.append(loss_div_fb.item())
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb,
                                                labels_fb_batch, criterion_cls, criterion_div, loss_div_fb,
                                                model_t_weights, args)

                # div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb['fusion_fb'], model_t_weights, args)
                #
                # if div_cosin_simility_all < 0:
                #     kl_loss = 0.0
                # else:
                #     kl_loss = args.kl_loss
                #
                # if ch_cosin_simility_all < 0:
                #     infb_loss = 0.0
                # else:
                #     infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb']
                loss_fb_distill = loss_ch_fb['loss_other']
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_fb_distill + loss_fb_ae

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb_distill.item())
                args.loss_csv_fb.append(loss_fb.item())

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                #     f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                #     # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                #     f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                #     f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                #     f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                #     f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb_ae.update(loss_fb_ae.item(), train_fb_batch.size(0))
                losses_fb_distill.update(loss_fb_distill.item(), train_fb_batch.size(0))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                # args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                args.optimizer_dict['opt_ae_t'].zero_grad()
                # loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_ae_t'].step()
                # args.optimizer_dict['opt_fb_ae'].step()

                with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fb)

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd_ae_s='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb_ae_t='{:05.3f}'.format(losses_fb_ae.avg),
                              loss_fb_distill='{:05.3f}'.format(losses_fb_distill.avg),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb, loss_div_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_19_3(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd_ae = utils.AverageMeter()
    losses_fwd_distill = utils.AverageMeter()
    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb_ae = utils.AverageMeter()
    losses_fb_distill = utils.AverageMeter()
    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            args.trainable_fb_dict['opt_ae_t'].eval()
            args.trainable_fwd_dict['opt_ae_s'].train()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("---------------Forward loss-------------" + "\n")

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd,
                                              labels_fwd_batch, criterion_cls, criterion_div, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd[
                'self_supervised_fwd']
            loss_fwd = loss_fwd_distill + loss_fwd_ae
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd_distill.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
            #     f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
            #     # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
            #     f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
            #     f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd_ae.update(loss_fwd_ae.item(), train_fwd_batch.size(0))
            losses_fwd_distill.update(loss_fwd_distill.item(), train_fwd_batch.size(0))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            # args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            # args.optimizer_dict['opt_ae_s'].zero_grad()
            # loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            # args.optimizer_dict['opt_ae_s'].step()
            # args.optimizer_dict['opt_fwd_ae'].step()

            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(args.loss_csv_fwd)

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()
                args.trainable_fb_dict['opt_ae_t'].train()
                args.trainable_fwd_dict['opt_ae_s'].eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                # loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                args.loss_csv_fb.append(loss_div_fb.item())
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb,
                                                labels_fb_batch, criterion_cls, criterion_div, loss_div_fb,
                                                model_t_weights, args)

                # div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb['fusion_fb'], model_t_weights, args)
                #
                # if div_cosin_simility_all < 0:
                #     kl_loss = 0.0
                # else:
                #     kl_loss = args.kl_loss
                #
                # if ch_cosin_simility_all < 0:
                #     infb_loss = 0.0
                # else:
                #     infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb']
                loss_fb_distill = loss_ch_fb['loss_other']
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_fb_distill + loss_fb_ae

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb_distill.item())
                args.loss_csv_fb.append(loss_fb.item())

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                #     f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                #     # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                #     f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                #     f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                #     f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                #     f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb_ae.update(loss_fb_ae.item(), train_fb_batch.size(0))
                losses_fb_distill.update(loss_fb_distill.item(), train_fb_batch.size(0))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                # args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                # args.optimizer_dict['opt_ae_s'].zero_grad()
                # loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                # args.optimizer_dict['opt_ae_s'].step()
                # args.optimizer_dict['opt_fb_ae'].step()

                with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fb)

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd_ae_s='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb_ae_t='{:05.3f}'.format(losses_fb_ae.avg),
                              loss_fb_distill='{:05.3f}'.format(losses_fb_distill.avg),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb, loss_div_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_19_4(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd_ae = utils.AverageMeter()
    losses_fwd_distill = utils.AverageMeter()
    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb_ae = utils.AverageMeter()
    losses_fb_distill = utils.AverageMeter()
    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            args.trainable_fb_dict['opt_ae_t'].eval()
            args.trainable_fwd_dict['opt_ae_s'].train()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("---------------Forward loss-------------" + "\n")

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd,
                                              labels_fwd_batch, criterion_cls, criterion_div, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd[
                'self_supervised_fwd']
            loss_fwd = loss_fwd_distill + loss_fwd_ae
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd_distill.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
            #     f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
            #     f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
            #     # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
            #     f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
            #     f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd_ae.update(loss_fwd_ae.item(), train_fwd_batch.size(0))
            losses_fwd_distill.update(loss_fwd_distill.item(), train_fwd_batch.size(0))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            # args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            # args.optimizer_dict['opt_ae_s'].zero_grad()
            # loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            # args.optimizer_dict['opt_ae_s'].step()
            # args.optimizer_dict['opt_fwd_ae'].step()

            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(args.loss_csv_fwd)

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()
                args.trainable_fb_dict['opt_ae_t'].train()
                args.trainable_fwd_dict['opt_ae_s'].eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                # loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                args.loss_csv_fb.append(loss_div_fb.item())
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb,
                                                labels_fb_batch, criterion_cls, criterion_div, args)

                if args.fbUseGradSim is True:
                    # 将 损失 做梯度相似性比较。
                    loss_fb_kl = loss_ch_fb['loss_fusion'] + loss_div_fb
                    # cls_grad = torch.autograd.grad([loss_ce_all, loss_kl_all], model_t_weights, allow_unused=True, retain_graph=True)
                    cls_grad = torch.autograd.grad(loss_ch_fb['loss_ce'], model_t_weights, allow_unused=True,
                                                   retain_graph=True)
                    kl_all_grad = torch.autograd.grad(loss_ch_fb['loss_self_kd'], model_t_weights, allow_unused=True,
                                                      retain_graph=True)
                    div_kl_grad = torch.autograd.grad(loss_fb_kl, model_t_weights, allow_unused=True, retain_graph=True)
                    cos_similarity_temp = cal_each_grad_sim2(cls_grad, kl_all_grad, args)
                    # cos_similarity_fusion = cal_each_grad_sim(loss_ce_all, loss_fusion_fb, model_t_weights, args)
                    # cos_similarity_div = cal_each_grad_sim(loss_ce_all, loss_div_fb, model_t_weights, args)
                    # args.loss_csv_fb.append(cos_similarity_temp.item())
                    # args.loss_csv_fb.append(cos_similarity_fusion.item())
                    # args.loss_csv_fb.append(cos_similarity_div.item())
                    cos_similarity_div = cal_each_grad_sim2(cls_grad, div_kl_grad, args)
                    args.loss_csv_fb.append(cos_similarity_temp.item())
                    args.loss_csv_fb.append(cos_similarity_div.item())
                    args.loss_csv_fb.append('None')
                    self_kl_all_coe = 1.0
                    fusion_coe = 1.0
                    div_coe = 1.0
                    if cos_similarity_temp < 0:
                        self_kl_all_coe = 0
                    # if cos_similarity_fusion < 0:
                    #     fusion_coe = 0
                    if cos_similarity_div < 0:
                        div_coe = 0
                    # loss_fb_other = loss_ce_all + self_kl_all_coe * loss_kl_all + fusion_coe * loss_fusion_fb + div_coe * loss_div_fb
                    loss_fb_other = loss_ch_fb['loss_ce'] + self_kl_all_coe * loss_ch_fb[
                        'loss_self_kd'] + div_coe * loss_fb_kl
                    # if cos_similarity_temp < 0:
                    #     loss_self_supervised_fb = loss_ce_all
                    # else:
                    #     loss_self_supervised_fb = loss_ce_all + loss_kl_all
                else:
                    loss_fb_kl = loss_ch_fb['loss_fusion'] + loss_div_fb
                    args.loss_csv_fb.append('None')
                    args.loss_csv_fb.append('None')
                    args.loss_csv_fb.append('None')
                    # loss_fb_other = loss_ce_all + loss_kl_all + loss_fusion_fb + loss_div_fb
                    loss_fb_other = loss_ch_fb['loss_ce'] + loss_ch_fb['loss_self_kd'] + loss_fb_kl

                loss_fb_ae = loss_ch_fb['recon_T_fb']
                loss_fb_distill = loss_fb_other
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_fb_distill + loss_fb_ae

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb_distill.item())
                args.loss_csv_fb.append(loss_fb.item())

                # with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                #     f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                #     f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                #     # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                #     f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                #     f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                #     f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                #     f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb_ae.update(loss_fb_ae.item(), train_fb_batch.size(0))
                losses_fb_distill.update(loss_fb_distill.item(), train_fb_batch.size(0))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                # args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                # args.optimizer_dict['opt_ae_s'].zero_grad()
                # loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                # args.optimizer_dict['opt_ae_s'].step()
                # args.optimizer_dict['opt_fb_ae'].step()

                with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fb)

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd_ae_s='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb_ae_t='{:05.3f}'.format(losses_fb_ae.avg),
                              loss_fb_distill='{:05.3f}'.format(losses_fb_distill.avg),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb, loss_div_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_111(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    # model_ae_t = args.module_auxcfae_t_dict['model_ae_t']
    # model_ae_s = args.module_auxcfae_s_dict['model_ae_s']
    model_auxcfae_t = [each for each in args.module_auxcfae_t_dict.values()]  # 0: ae, 1-last: auxcf
    model_auxcfae_s = [each for each in args.module_auxcfae_s_dict.values()]  # 0: ae, 1-last: auxcf

    losses_fwd_ae = utils.AverageMeter()
    losses_fwd_distill = utils.AverageMeter()
    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb_ae = utils.AverageMeter()
    losses_fb_distill = utils.AverageMeter()
    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t.zero_grad()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            # param_temp = deepcopy(param)
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(
                zip(train_dl_fwd, train_dl_fb)):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            for each in model_auxcfae_t:
                each.eval()
            for each in model_auxcfae_s:
                each.train()
            # args.trainable_ae_t_dict['opt_ae_t'].eval()
            # args.trainable_ae_s_dict['opt_ae_s'].train()
            # for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
            #     args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].eval()
            # for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
            #     args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].train()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                # output_t_fwd = output_t_fwd.detach()
                # feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())
            # forward channel loss

            feat_t_fwd_list = selectInterlayer111(feat_t_fwd, args.blocks_amount_t, args)
            feat_s_fwd_list = selectInterlayer111(feat_s_fwd, args.blocks_amount_s, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_s_fwd, labels_fwd_batch,
                                              criterion_cls, criterion_div, model_auxcfae_t, model_auxcfae_s, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd[
                'self_supervised_fwd']
            loss_fwd = loss_fwd_distill + loss_fwd_ae
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd_distill.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd_ae.update(loss_fwd_ae.item(), train_fwd_batch.size(0))
            losses_fwd_distill.update(loss_fwd_distill.item(), train_fwd_batch.size(0))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            # args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            args.optimizer_dict['opt_ae_s'].zero_grad()
            args.optimizer_dict['opt_auxcf_s'].zero_grad()
            # loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_ae_s'].step()
            args.optimizer_dict['opt_auxcf_s'].step()
            # args.optimizer_dict['opt_fwd_ae'].step()

            if args.loss2csv:
                with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fwd)

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()
                for each in model_auxcfae_t:
                    each.train()
                for each in model_auxcfae_s:
                    each.eval()
                # args.trainable_ae_t_dict['opt_ae_t'].train()
                # args.trainable_ae_s_dict['opt_ae_s'].eval()
                # for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
                #     args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].train()
                # for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
                #     args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].eval()

                model_t.zero_grad()

                # model_t_weights = []
                # for name, param in model_t.named_parameters():
                #     param.requires_grad = True
                #     if "weight" in name:
                #         # param_temp = deepcopy(param)
                #         # param.grad = None
                #         model_t_weights.append(param)

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    # output_s_fb = output_s_fb.detach()
                    # feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                args.loss_csv_fb.append(loss_div_fb.item())
                # feedback channel loss

                feat_t_fb_list = selectInterlayer111(feat_t_fb, args.blocks_amount_t, args)
                feat_s_fb_list = selectInterlayer111(feat_s_fb, args.blocks_amount_s, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, labels_fb_batch,
                                                criterion_cls, criterion_div, loss_div_fb, model_t_weights,
                                                model_auxcfae_t, model_auxcfae_s, args)

                loss_fb_ae = loss_ch_fb['recon_T_fb']
                loss_fb_distill = loss_ch_fb['loss_other']
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_fb_distill + loss_fb_ae

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb_distill.item())
                args.loss_csv_fb.append(loss_fb.item())

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb_ae.update(loss_fb_ae.item(), train_fb_batch.size(0))
                losses_fb_distill.update(loss_fb_distill.item(), train_fb_batch.size(0))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                # args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                args.optimizer_dict['opt_ae_t'].zero_grad()
                args.optimizer_dict['opt_auxcf_t'].zero_grad()
                # loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_ae_t'].step()
                args.optimizer_dict['opt_auxcf_t'].step()
                # args.optimizer_dict['opt_fb_ae'].step()

                if args.loss2csv:
                    with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(args.loss_csv_fb)

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd_ae_s='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb_ae_t='{:05.3f}'.format(losses_fb_ae.avg),
                              loss_fb_distill='{:05.3f}'.format(losses_fb_distill.avg),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb, loss_div_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_111_3(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd_ae = utils.AverageMeter()
    losses_fwd_distill = utils.AverageMeter()
    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb_ae = utils.AverageMeter()
    losses_fb_distill = utils.AverageMeter()
    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    with tqdm(total=len(train_dl_fwd)) as t:
        # for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(zip(train_dl_fwd, train_dl_fb)):
        for i, ((train_fwd_batch, labels_fwd_batch)) in enumerate(train_dl_fwd):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):
            args.loss_csv_fwd = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            args.trainable_ae_t_dict['opt_ae_t'].eval()
            args.trainable_ae_s_dict['opt_ae_s'].train()
            for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
                args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].eval()
            for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
                args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].train()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                # output_t_fwd = output_t_fwd.detach()
                # feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())
            # forward channel loss

            feat_t_fwd_list = selectInterlayer111(feat_t_fwd, args.blocks_amount_t, args)
            feat_s_fwd_list = selectInterlayer111(feat_s_fwd, args.blocks_amount_s, args)

            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd,
                                              labels_fwd_batch, criterion_cls, criterion_div, args)

            loss_fwd_ae = loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd[
                'self_supervised_fwd']
            loss_fwd = loss_fwd_distill + loss_fwd_ae
            args.loss_csv_fwd.append(loss_fwd_ae.item())
            args.loss_csv_fwd.append(loss_fwd_distill.item())
            args.loss_csv_fwd.append(loss_fwd.item())

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd_ae.update(loss_fwd_ae.item(), train_fwd_batch.size(0))
            losses_fwd_distill.update(loss_fwd_distill.item(), train_fwd_batch.size(0))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            # args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            args.optimizer_dict['opt_ae_s'].zero_grad()
            args.optimizer_dict['opt_auxcf_s'].zero_grad()
            # loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_ae_s'].step()
            args.optimizer_dict['opt_auxcf_s'].step()
            # args.optimizer_dict['opt_fwd_ae'].step()

            with open(os.path.join(args.save_folder, args.loss_csv_fwd_name), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(args.loss_csv_fwd)

            t.set_postfix(loss_fwd_ae='{:05.3f}'.format(losses_fwd_ae.avg),
                          loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                          loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                          lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd_ae, loss_fwd_distill, loss_fwd, feat_t_fwd_list, feat_s_fwd_list

    if epoch >= (args.feedback_time - 1):
        # ===================Feedback Distill=====================
        # 获取模型的所有 weight
        model_t_weights = []
        for name, param in model_t.named_parameters():
            param.requires_grad = True
            if "weight" in name:
                model_t_weights.append(param)

        with tqdm(total=len(train_dl_fb)) as t:
            # list_dataloader = list(dataloader)
            for i, ((train_fb_batch, labels_fb_batch)) in enumerate(train_dl_fb):

                args.loss_csv_fb = []
                args.loss_csv_fb.append(str(epoch) + '_' + str(i))

                model_t.train()
                model_s.eval()
                args.trainable_ae_t_dict['opt_ae_t'].train()
                args.trainable_ae_s_dict['opt_ae_s'].eval()
                for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
                    args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].train()
                for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
                    args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    # output_s_fb = output_s_fb.detach()
                    # feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                # feedback cls + kl div
                # loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                args.loss_csv_fb.append(loss_div_fb.item())
                # feedback channel loss

                feat_t_fb_list = selectInterlayer111(feat_t_fb, args.blocks_amount_t, args)
                feat_s_fb_list = selectInterlayer111(feat_s_fb, args.blocks_amount_s, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb,
                                                labels_fb_batch, criterion_cls, criterion_div, loss_div_fb,
                                                model_t_weights, args)

                loss_fb_ae = loss_ch_fb['recon_T_fb']
                loss_fb_distill = loss_ch_fb['loss_other']
                # loss_fb = loss_ch_fb['fusion_fb'] + loss_ch_fb['self_supervised_fb']
                loss_fb = loss_fb_distill + loss_fb_ae

                args.loss_csv_fb.append(loss_fb_ae.item())
                args.loss_csv_fb.append(loss_fb_distill.item())
                args.loss_csv_fb.append(loss_fb.item())

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb_ae.update(loss_fb_ae.item(), train_fb_batch.size(0))
                losses_fb_distill.update(loss_fb_distill.item(), train_fb_batch.size(0))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                # args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                args.optimizer_dict['opt_ae_t'].zero_grad()
                args.optimizer_dict['opt_auxcf_t'].zero_grad()
                # loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_ae_t'].step()
                args.optimizer_dict['opt_auxcf_t'].step()
                # args.optimizer_dict['opt_fb_ae'].step()

                with open(os.path.join(args.save_folder, args.loss_csv_fb_name), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(args.loss_csv_fb)

                t.set_postfix(loss_fwd_ae_s='{:05.3f}'.format(losses_fwd_ae.avg),
                              loss_fwd_distill='{:05.3f}'.format(losses_fwd_distill.avg),
                              loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb_ae_t='{:05.3f}'.format(losses_fb_ae.avg),
                              loss_fb_distill='{:05.3f}'.format(losses_fb_distill.avg),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))

                t.update()

                del train_fb_batch, labels_fb_batch, feat_t_fb, output_t_fb, feat_s_fb, output_s_fb, loss_fb_distill, loss_div_fb, feat_t_fb_list, feat_s_fb_list, loss_ch_fb, loss_fb_ae, loss_fb

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_133(train_dl, epoch, args):
    train_dl_fwd = train_dl['fwd']
    train_dl_fb = train_dl['fb']
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    with tqdm(total=len(train_dl_fwd)) as t:
        # list_dataloader = list(dataloader)
        # for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(zip(train_dl_fwd, train_dl_fb)):
        for i, ((train_fwd_batch, labels_fwd_batch)) in enumerate(train_dl_fwd):
            # for i, (train_batch, labels_batch) in enumerate(train_dl):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, args)

            loss_fwd_ae = args.aefwd_t_loss * loss_ch_fwd['recon_T_fwd'] + args.aefwd_s_loss * loss_ch_fwd[
                'recon_S_fwd']
            loss_fwd = args.ce_loss * loss_cls_fwd + args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd[
                'fusion_fwd']

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
                f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_fwd_ae'].step()

            t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                          lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                          lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))

            t.update()

            del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd_ae, loss_fwd, feat_t_fwd_list, feat_s_fwd_list

    if epoch >= (args.feedback_time - 1):
        # ===================Feedback Distill=====================
        # 获取模型的所有 weight
        model_t_weights = []
        for name, param in model_t.named_parameters():
            param.requires_grad = True
            if "weight" in name:
                model_t_weights.append(param)

        with tqdm(total=len(train_dl_fb)) as t:
            # list_dataloader = list(dataloader)
            for i, ((train_fb_batch, labels_fb_batch)) in enumerate(train_dl_fb):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, args)

                div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb,
                                                                            loss_ch_fb['fusion_fb'],
                                                                            model_t_weights, args)

                if div_cosin_simility_all < 0:
                    kl_loss = 0.0
                else:
                    kl_loss = args.kl_loss

                if ch_cosin_simility_all < 0:
                    infb_loss = 0.0
                else:
                    infb_loss = args.infb_loss

                loss_fb_ae = args.aefb_t_loss * loss_ch_fb['recon_T_fb'] + args.aefb_s_loss * loss_ch_fb['recon_S_fb']
                loss_fb = args.ce_loss * loss_cls_fb + kl_loss * loss_div_fb + infb_loss * loss_ch_fb['fusion_fb']

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                    f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                    # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                    f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                    f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                    f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_fb_ae'].step()

                t.set_postfix(loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))

                t.update()

                del train_fb_batch, labels_fb_batch, feat_t_fb, output_t_fb, feat_s_fb, output_s_fb, loss_cls_fb, loss_div_fb, feat_t_fb_list, feat_s_fb_list, loss_ch_fb, div_cosin_simility_all, ch_cosin_simility_all, loss_fb_ae, loss_fb

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def train_our_distill_23(train_dl, epoch, args):
    model_t = args.module_dict['model_t']
    model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd = utils.AverageMeter()
    top1_s = utils.AverageMeter()
    top5_s = utils.AverageMeter()

    losses_fb = utils.AverageMeter()
    top1_t = utils.AverageMeter()
    top5_t = utils.AverageMeter()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl)) as t:
        # list_dataloader = list(dataloader)
        # for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(zip(train_dl_fwd, train_dl_fb)):
        for i, (train_batch, labels_batch) in enumerate(train_dl):

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()

            # 切分数据集，用于 feedback 的数据集，按照 fb_set_percent 比例来划分 batch_size。
            fb_count = int(args.batch_size * args.fb_set_percent)
            # fwd_count = args.batch_size - fb_count
            train_fwd_batch = train_batch[:-fb_count]
            labels_fwd_batch = labels_batch[:-fb_count]
            train_fb_batch = train_batch[-fb_count:]
            labels_fb_batch = labels_batch[-fb_count:]

            train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # convert to torch Variables
            train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                output_t_fwd = output_t_fwd.detach()
                feat_t_fwd = [f.detach() for f in feat_t_fwd]
                # feat_t = Variable(feat_t, requires_grad=False)

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("---------------Forward loss-------------" + "\n")
            # forward cls + kl div
            loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            # forward channel loss

            feat_t_fwd_list = selectInterlayer(feat_t_fwd, args)
            feat_s_fwd_list = selectInterlayer(feat_s_fwd, args)

            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, args)

            loss_fwd_ae = loss_ch_fwd['recon_T_fwd'] + loss_ch_fwd['recon_S_fwd']
            loss_fwd = args.ce_loss * loss_cls_fwd + args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd[
                'fusion_fwd']

            with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                f.write("CE_fwd:" + str(loss_cls_fwd.item()) + '\t')
                f.write("KD_fwd:" + str(loss_div_fwd.item()) + '\t')
                # f.write("CH_fwd:" + str(loss_ch_fwd.item()) + '\t')
                f.write("LOSS_total_fwd:" + str(loss_fwd.item()) + '\t')
                f.write("\n")

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd_ae'].zero_grad()
            # Forward distill backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_fwd_ae'].step()

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()

                train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # convert to torch Variables
                train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)
                    output_s_fb = output_s_fb.detach()
                    feat_s_fb = [f.detach() for f in feat_s_fb]
                    # feat_t = Variable(feat_t, requires_grad=False)

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("---------------Feedback loss-------------" + "\n")
                # feedback cls + kl div
                loss_cls_fb = criterion_cls(output_t_fb, labels_fb_batch)
                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
                # feedback channel loss

                feat_t_fb_list = selectInterlayer(feat_t_fb, args)
                feat_s_fb_list = selectInterlayer(feat_s_fb, args)

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, args)

                div_cosin_simility_all, ch_cosin_simility_all = cal_grad_fb(loss_cls_fb, loss_div_fb,
                                                                            loss_ch_fb['fusion_fb'], model_t_weights,
                                                                            args)

                if div_cosin_simility_all < 0:
                    kl_loss = 0.0
                else:
                    kl_loss = args.kl_loss

                if ch_cosin_simility_all < 0:
                    infb_loss = 0.0
                else:
                    infb_loss = args.infb_loss

                loss_fb_ae = loss_ch_fb['recon_T_fb'] + loss_ch_fb['recon_S_fb']
                loss_fb = args.ce_loss * loss_cls_fb + kl_loss * loss_div_fb + infb_loss * loss_ch_fb['fusion_fb']

                with open(os.path.join(args.save_folder, args.loss_txt_name), 'a') as f:
                    f.write("CE_fb:" + str(loss_cls_fb.item()) + '\t')
                    f.write("KD_fb:" + str(loss_div_fb.item()) + '\t')
                    # f.write("CH_fb:" + str(loss_ch_fb.item()) + '\t')
                    f.write("KD_cos_sim:" + str(div_cosin_simility_all.item()) + '\t')
                    f.write("CH_cos_sim:" + str(ch_cosin_simility_all.item()) + '\t')
                    f.write("LOSS_total_fb:" + str(loss_fb.item()) + '\t')
                    f.write("\n")

                acc1_t, acc5_t = utils.accuracy(output_t_fb, labels_fb_batch, topk=(1, 5))
                losses_fb.update(loss_fb.item(), train_fb_batch.size(0))
                top1_t.update(acc1_t[0], train_fb_batch.size(0))
                top5_t.update(acc5_t[0], train_fb_batch.size(0))

                # Feedback distill Auto-Encoder backward
                args.optimizer_dict['opt_fb_ae'].zero_grad()
                # Feedback distill backward
                args.optimizer_dict['opt_fb'].zero_grad()
                loss_fb_ae.backward(retain_graph=True)
                loss_fb.backward()
                args.optimizer_dict['opt_fb'].step()
                args.optimizer_dict['opt_fb_ae'].step()

            if epoch >= args.feedback_time:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']),
                              loss_fb='{:05.3f}'.format(losses_fb.avg),
                              lr_T='{:05.6f}'.format(args.optimizer_dict['opt_fb'].param_groups[0]['lr']))
            else:
                t.set_postfix(loss_fwd='{:05.3f}'.format(losses_fwd.avg),
                              lr_S='{:05.6f}'.format(args.optimizer_dict['opt_fwd'].param_groups[0]['lr']))

            t.update()

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_cls_fb, loss_div_fb, loss_ch_fb, loss_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_cls_fwd, loss_div_fwd, loss_ch_fwd, loss_fwd

    if epoch >= args.feedback_time:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb))
    else:
        logging.info(
            "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}."
                .format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd))

    return [top1_t.avg, top1_s.avg], [losses_fwd.avg, losses_fb.avg]


def selectInterlayer(feat, args):
    if 'ResNet' in args.model_t:
        feat_list = feat[1:-1][-args.blocks_amount:]
    elif 'MobileNetV2' in args.model_t:
        feat_list = feat[1:-1][-args.blocks_amount:]
    elif 'VGG' in args.model_t:
        feat_list = feat[1:-1][-args.blocks_amount:]
    elif 'WRN' in args.model_t:
        feat_list = feat[1:-1][-args.blocks_amount:]
    elif 'ShuffleV2' in args.model_t:
        feat_list = feat[1:-1][-args.blocks_amount:]

    assert len(feat_list) == args.blocks_amount
    # if 'ResNet' in args.model_t:
    #     feat_list = feat[2:-1]
    # elif 'MobileNetV2' in args.model_t:
    #     feat_list = feat[2:-1]
    # elif 'VGG' in args.model_t:
    #     feat_list = feat[2:-1]
    # elif 'WRN' in args.model_t:
    #     feat_list = feat[1:-2]
    # elif 'ShuffleV2' in args.model_t:
    #     feat_list = feat[1:-2]

    return feat_list


def selectInterlayer111(feat, blocks_amount, args):
    if 'ResNet' in args.model_t:
        feat_list = feat[1:-1][-blocks_amount:]
    elif 'MobileNetV2' in args.model_t:
        feat_list = feat[1:-1][-blocks_amount:]
    elif 'VGG' in args.model_t:
        feat_list = feat[1:-1][-blocks_amount:]
    elif 'WRN' in args.model_t:
        feat_list = feat[1:-1][-blocks_amount:]
    elif 'ShuffleV2' in args.model_t:
        feat_list = feat[1:-1][-blocks_amount:]

    assert len(feat_list) == blocks_amount
    # if 'ResNet' in args.model_t:
    #     feat_list = feat[2:-1]
    # elif 'MobileNetV2' in args.model_t:
    #     feat_list = feat[2:-1]
    # elif 'VGG' in args.model_t:
    #     feat_list = feat[2:-1]
    # elif 'WRN' in args.model_t:
    #     feat_list = feat[1:-2]
    # elif 'ShuffleV2' in args.model_t:
    #     feat_list = feat[1:-2]

    return feat_list


def cal_each_grad_sim_fb(loss_cls, loss_other, model_t_weights, args):
    # Note 有错
    cls_grad = torch.autograd.grad(loss_cls, model_t_weights, allow_unused=True, retain_graph=True)
    other_grad = torch.autograd.grad(loss_other, model_t_weights, allow_unused=True, retain_graph=True)

    cosin_simility = []
    for each_cls, each_other in zip(cls_grad, other_grad):
        cosin_simility.append(F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0))

    cosin_simility_all = torch.tensor(0.0).cuda()
    for each_sim in cosin_simility:
        cosin_simility_all += torch.mean(each_sim)

    return cosin_simility_all


def cal_grad_fb(loss_cls_fb, loss_div_fb, loss_ch_fb, model_t_weights, args):
    # Note 有错
    cls_fb_grad = torch.autograd.grad(loss_cls_fb, model_t_weights, allow_unused=True, retain_graph=True)
    div_fb_grad = torch.autograd.grad(loss_div_fb, model_t_weights, allow_unused=True, retain_graph=True)
    kd_fb_grad = torch.autograd.grad(loss_ch_fb, model_t_weights, allow_unused=True, retain_graph=True)

    div_cosin_simility = []
    kd_cosin_simility = []
    for each_cls, each_div in zip(cls_fb_grad, div_fb_grad):
        div_cosin_simility.append(F.cosine_similarity(each_cls.reshape(-1), each_div.reshape(-1), dim=0))
        # kd_cosin_simility.append(F.cosine_similarity(each_cls, each_kd, dim=0))

    for each_cls, each_kd in zip(cls_fb_grad[:-1], kd_fb_grad[:-1]):
        # div_cosin_simility.append(F.cosine_similarity(each_cls, each_div, dim=0))
        kd_cosin_simility.append(F.cosine_similarity(each_cls.reshape(-1), each_kd.reshape(-1), dim=0))

    div_cosin_simility_all = torch.tensor(0.0).cuda()
    ch_cosin_simility_all = torch.tensor(0.0).cuda()
    for each_div_sim in div_cosin_simility:
        div_cosin_simility_all += torch.mean(each_div_sim)
        # kd_cosin_simility_all += torch.mean(each_kd_sim)

    for each_kd_sim in kd_cosin_simility:
        # div_cosin_simility_all += torch.mean(each_div_sim)
        ch_cosin_simility_all += torch.mean(each_kd_sim)

    # print(div_cosin_simility_all)
    # print(kd_cosin_simility_all)

    return div_cosin_simility_all, ch_cosin_simility_all


def cal_each_grad_sim2(cls_grad, other_grad, args):
    nontype_idx_cls = len(cls_grad)
    nontype_idx_oth = len(other_grad)
    for idx, each in enumerate(cls_grad):
        if each is None:
            nontype_idx_cls = idx
            break

    for idx, each in enumerate(other_grad):
        if each is None:
            nontype_idx_oth = idx
            break

    split_idx = min(nontype_idx_cls, nontype_idx_oth)

    # print("cls")
    # print(nontype_cls)
    # print('other')
    # print(nontype_oth)

    cosin_simility_all = torch.mean(torch.Tensor(
        [F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0) for each_cls, each_other in
         zip(cls_grad[:split_idx], other_grad[:split_idx])]).cuda())

    # cosin_simility = []
    # for each_cls, each_other in zip(cls_grad[:split_idx], other_grad[:split_idx]):
    #     cosin_simility.append(F.cosine_similarity(each_cls.reshape(-1), each_other.reshape(-1), dim=0))
    #
    # cosin_simility_all = torch.tensor(0.0).cuda()
    # for each_sim in cosin_simility:
    #     cosin_simility_all += torch.mean(each_sim)

    return cosin_simility_all


def evaluate(model, criterion, dataloader, args, metric_method='acc'):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    total = 0
    correct = 0
    # recall = Recall(device='cuda')
    # accuracy = Accuracy()
    output_all, labels_all = [], []

    with torch.no_grad():
        # compute metrics over the dataset
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
            # data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            # compute model output
            output_batch = model(data_batch, is_feat=False)
            loss = criterion(output_batch, labels_batch)
            output_all.append(output_batch.data)
            labels_all.append(labels_batch.data)

            # print("[Iter %d]" % i)
            # if args.regularization:
            #     loss = loss_fn(output_batch, labels_batch, params)
            # else:
            #     loss = loss_fn(output_batch, labels_batch)

            # measure accuracy and record loss
            if metric_method == 'acc':
                acc1, acc5 = utils.accuracy(output_batch, labels_batch, topk=(1, 5))
                # accuracy.update((output_batch, labels_batch))
                top1.update(acc1[0], data_batch.size(0))
                top5.update(acc5[0], data_batch.size(0))

            losses.update(loss.item(), data_batch.size(0))

            # losses.update(loss.data, data_batch.size(0))
            # _, predicted = output_batch.max(1)
            # total += labels_batch.size(0)
            # correct += predicted.eq(labels_batch).sum().item()

        if metric_method == 'recall':
            output_all = torch.cat(output_all)
            labels_all = torch.cat(labels_all)
            rec = utils.recall(output_all, labels_all, K=[1, 5])
            for idx, each in enumerate(rec):
                rec[idx] = 100.0 * each

    # if metric_method == 'acc':
    #     total_accuracy = accuracy.compute()
    # elif metric_method == 'recall':
    #     total_recall = recall.compute()

    # loss_avg = losses.avg
    # acc = 100.*correct/total
    if metric_method == 'acc':
        logging.info("- Eval metrics, Acc@1:{top1.avg:.3f}, Acc@5:{top5.avg:.4f}, loss: {loss.avg:.4f}".format(
            top1=top1, top5=top5, loss=losses))
        my_metric = {'top1': top1.avg, 'top5': top5.avg, 'loss': losses.avg}
    elif metric_method == 'recall':
        logging.info("- Eval metrics, Recall@1:{recall1:.3f}, Recall@5:{recall5:.4f}, loss: {loss.avg:.4f}".format(
            recall1=rec[0], recall5=rec[1], loss=losses))
        my_metric = {'top1': rec[0], 'top5': rec[1], 'loss': losses.avg}
    return my_metric


def evaluate_reviewkd(model, criterion, dataloader, args, metric_method='acc'):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    total = 0
    correct = 0
    # recall = Recall(device='cuda')
    # accuracy = Accuracy()
    output_all, labels_all = [], []

    with torch.no_grad():
        # compute metrics over the dataset
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
            # data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            # compute model output
            feat, output_batch = model(data_batch)
            loss = criterion(output_batch, labels_batch)
            output_all.append(output_batch.data)
            labels_all.append(labels_batch.data)

            # print("[Iter %d]" % i)
            # if args.regularization:
            #     loss = loss_fn(output_batch, labels_batch, params)
            # else:
            #     loss = loss_fn(output_batch, labels_batch)

            # measure accuracy and record loss
            if metric_method == 'acc':
                acc1, acc5 = utils.accuracy(output_batch, labels_batch, topk=(1, 5))
                # accuracy.update((output_batch, labels_batch))
                top1.update(acc1[0], data_batch.size(0))
                top5.update(acc5[0], data_batch.size(0))

            losses.update(loss.item(), data_batch.size(0))

            # losses.update(loss.data, data_batch.size(0))
            # _, predicted = output_batch.max(1)
            # total += labels_batch.size(0)
            # correct += predicted.eq(labels_batch).sum().item()

        if metric_method == 'recall':
            output_all = torch.cat(output_all)
            labels_all = torch.cat(labels_all)
            rec = utils.recall(output_all, labels_all, K=[1, 5])
            for idx, each in enumerate(rec):
                rec[idx] = 100.0 * each

    # if metric_method == 'acc':
    #     total_accuracy = accuracy.compute()
    # elif metric_method == 'recall':
    #     total_recall = recall.compute()

    # loss_avg = losses.avg
    # acc = 100.*correct/total
    if metric_method == 'acc':
        logging.info("- Eval metrics, Acc@1:{top1.avg:.3f}, Acc@5:{top5.avg:.4f}, loss: {loss.avg:.4f}".format(
            top1=top1, top5=top5, loss=losses))
        my_metric = {'top1': top1.avg, 'top5': top5.avg, 'loss': losses.avg}
    elif metric_method == 'recall':
        logging.info("- Eval metrics, Recall@1:{recall1:.3f}, Recall@5:{recall5:.4f}, loss: {loss.avg:.4f}".format(
            recall1=rec[0], recall5=rec[1], loss=losses))
        my_metric = {'top1': rec[0], 'top5': rec[1], 'loss': losses.avg}
    return my_metric


def evaluate_sokd_aux(model_t, aux_model, criterion, dataloader, args, metric_method='acc'):
    """Evaluate the SOKD model. """

    # set model to evaluation mode
    model_t.eval()
    aux_model.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    total = 0
    correct = 0
    # recall = Recall(device='cuda')
    # accuracy = Accuracy()
    output_all, labels_all = [], []

    with torch.no_grad():
        # compute metrics over the dataset
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
            # data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            # compute model output
            feat_t, output_t = model_t(data_batch, is_feat=True)
            if 'WRN' in args.model_t:
                feat = feat_t[-4]
            else:
                feat = feat_t[-3]
            feat_aux, output_batch = aux_model(feat.detach())
            loss = criterion(output_batch, labels_batch)
            output_all.append(output_batch.data)
            labels_all.append(labels_batch.data)

            # print("[Iter %d]" % i)
            # if args.regularization:
            #     loss = loss_fn(output_batch, labels_batch, params)
            # else:
            #     loss = loss_fn(output_batch, labels_batch)

            # measure accuracy and record loss
            if metric_method == 'acc':
                acc1, acc5 = utils.accuracy(output_batch, labels_batch, topk=(1, 5))
                # accuracy.update((output_batch, labels_batch))
                top1.update(acc1[0], data_batch.size(0))
                top5.update(acc5[0], data_batch.size(0))

            losses.update(loss.item(), data_batch.size(0))

            # losses.update(loss.data, data_batch.size(0))
            # _, predicted = output_batch.max(1)
            # total += labels_batch.size(0)
            # correct += predicted.eq(labels_batch).sum().item()

        if metric_method == 'recall':
            output_all = torch.cat(output_all)
            labels_all = torch.cat(labels_all)
            rec = utils.recall(output_all, labels_all, K=[1, 5])
            for idx, each in enumerate(rec):
                rec[idx] = 100.0 * each

    # if metric_method == 'acc':
    #     total_accuracy = accuracy.compute()
    # elif metric_method == 'recall':
    #     total_recall = recall.compute()

    # loss_avg = losses.avg
    # acc = 100.*correct/total
    if metric_method == 'acc':
        logging.info("-AUX Eval metrics, Acc@1:{top1.avg:.3f}, Acc@5:{top5.avg:.4f}, AUX loss: {loss.avg:.4f}".format(
            top1=top1, top5=top5, loss=losses))
        my_metric = {'top1': top1.avg, 'top5': top5.avg, 'loss': losses.avg}
    elif metric_method == 'recall':
        logging.info(
            "-AUX Eval metrics, Recall@1:{recall1:.3f}, Recall@5:{recall5:.4f}, AUX loss: {loss.avg:.4f}".format(
                recall1=rec[0], recall5=rec[1], loss=losses))
        my_metric = {'top1': rec[0], 'top5': rec[1], 'loss': losses.avg}
    return my_metric


def evaluate_2(model, criterion, dataloader, args, metric_method='acc'):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    from ignite.metrics import Recall, Accuracy
    model.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    total = 0
    correct = 0
    recall = Recall(average=True, device='cuda')
    accuracy = Accuracy(device='cuda')
    output_all, labels_all = [], []

    with torch.no_grad():
        # compute metrics over the dataset
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
            # data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            # compute model output
            # output_batch,_ = model(data_batch)  # , is_feat=False
            output_batch = model(data_batch, is_feat=False)
            # loss = criterion(output_batch[0], labels_batch)
            loss = criterion(output_batch, labels_batch)
            output_all.append(output_batch.data)  # .data
            labels_all.append(labels_batch.data)  # .data

            # print("[Iter %d]" % i)
            # if args.regularization:
            #     loss = loss_fn(output_batch, labels_batch, params)
            # else:
            #     loss = loss_fn(output_batch, labels_batch)

            # measure accuracy and record loss
            if metric_method == 'acc':
                # way 1 custom
                acc1, acc5 = utils.accuracy(output_batch, labels_batch, topk=(1, 5))
                top1.update(acc1[0], data_batch.size(0))
                top5.update(acc5[0], data_batch.size(0))
                # accuracy.update((output_batch[0], labels_batch))
                # way 2 ignite.metrics
                accuracy.update((output_batch, labels_batch))
            elif metric_method == 'recall':
                # acc1, acc5 = utils.recall(output_batch, labels_batch, K=[1, 5])
                # top1.update(acc1[0], data_batch.size(0))
                # top5.update(acc5[0], data_batch.size(0))
                # recall.update((output_batch[0], labels_batch))
                # way 2 ignite.metrics
                recall.update((output_batch, labels_batch))

            # loss = 0
            losses.update(loss.item(), data_batch.size(0))

            # losses.update(loss.data, data_batch.size(0))
            # _, predicted = output_batch.max(1)
            # total += labels_batch.size(0)
            # correct += predicted.eq(labels_batch).sum().item()

        if metric_method == 'recall':
            # way 1 custom
            output_all = torch.cat(output_all)
            labels_all = torch.cat(labels_all)
            rec = utils.recall(output_all, labels_all, K=[1, 5])
            for idx, each in enumerate(rec):
                rec[idx] = 100.0 * each

        # if metric_method == 'recall':
        #     output_all = torch.cat(output_all)
        #     labels_all = torch.cat(labels_all)
        #     rec = utils.recall(output_all, labels_all, K=[1, 5])
        #     for idx, each in enumerate(rec):
        #         rec[idx] = 100.0 * each

    if metric_method == 'acc':
        total_accuracy = accuracy.compute()
    elif metric_method == 'recall':
        total_recall = recall.compute()

    # loss_avg = losses.avg
    # acc = 100.*correct/total

    # logging.info("- Eval metrics, Acc@1:{top1.avg:.3f}, Acc@5:{top5.avg:.4f}, loss: {loss.avg:.4f}".format(
    #     top1=top1, top5=top5, loss=losses))
    # my_metric = {'top1': top1.avg, 'top5': top5.avg, 'loss': losses.avg}
    # if metric_method == 'acc':
    #     logging.info("- Eval metrics, Acc@1:{top1.avg:.3f}, Acc@5:{top5.avg:.4f}, loss: {loss.avg:.4f}".format(
    #         top1=top1, top5=top5, loss=losses))
    #     my_metric = {'top1': top1.avg, 'top5': top5.avg, 'loss': losses.avg}
    # elif metric_method == 'recall':
    #     logging.info("- Eval metrics, Recall@1:{recall1:.3f}, Recall@5:{recall5:.4f}, loss: {loss.avg:.4f}".format(
    #         recall1=rec[0], recall5=rec[1], loss=losses))
    #     my_metric = {'top1': rec[0], 'top5': rec[1], 'loss': losses.avg}

    if metric_method == 'acc':
        my_metric = {'top1_custom': top1.avg, 'top1_api': total_accuracy, 'loss': losses.avg}
    elif metric_method == 'recall':
        my_metric = {'top1_custom': rec[0], 'top1_api': total_recall, 'loss': losses.avg}
    return my_metric


def cal_contrast_cka(feat_t, feat_s, args):
    cka_score_list = []
    if 'ResNet' in args.model_t:
        cka_t = feat_t[1:-1]
    elif 'WRN' in args.model_t:
        cka_t = feat_t[1:-1]

    if 'ResNet' in args.model_s:
        cka_s = feat_s[1:-1]
    elif 'WRN' in args.model_s:
        cka_s = feat_s[1:-1]
    elif 'ShuffleV2' in args.model_s:
        cka_s = feat_s[:-1]
    elif 'MobileNetV2' in args.model_s:
        cka_s = feat_s[1:-1]

    # if 'ShuffleV2' in args.model_s:
    #     cka_s = feat_s[:-1]
    # else:
    #     cka_s = feat_s[1:-1]
    # cka_t = feat_t[1:-1]

    for s_each, t_each in zip(cka_s, cka_t):
        cka_score_pair = linear_CKA_GPU(t_each.reshape((t_each.shape[0], -1)).detach(),
                                        s_each.reshape((s_each.shape[0], -1)).detach())
        cka_score_list.append(cka_score_pair)

    return cka_score_list
