import os
import time
import math
import utils
from tqdm import tqdm
import logging
from torch.autograd import Variable
import torch
import csv
import torch.distributed as dist
from utils_ddp import AverageMeter, Summary, ProgressMeter, is_main_process

def train_our_distill_method1(train_dl, epoch, args):
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

    model_t.zero_grad()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)

    with tqdm(total=len(train_dl_fwd)) as t:
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(zip(train_dl_fwd, train_dl_fb)):

            args.loss_csv_fwd = []
            args.loss_csv_fb = []
            args.loss_csv_fwd.append(str(epoch) + '_' + str(i))
            args.loss_csv_fb.append(str(epoch) + '_' + str(i))

            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            args.trainable_ae_t_dict.eval()
            args.trainable_ae_s_dict.train()
            args.trainable_auxcf_t_dict.eval()
            args.trainable_auxcf_s_dict.train()
            # for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
            #     args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].eval()
            # for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
            #     args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].train()

            # train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # # convert to torch Variables
            # train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)
            train_fwd_batch = train_fwd_batch.to(args.device, non_blocking=True)
            labels_fwd_batch = labels_fwd_batch.to(args.device, non_blocking=True)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)
                # output_t_fwd = output_t_fwd.detach()
                # feat_t_fwd = [f.detach() for f in feat_t_fwd]

            # forward cls + kl div
            # loss_cls_fwd = criterion_cls(output_s_fwd, labels_fwd_batch)
            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)
            args.loss_csv_fwd.append(loss_div_fwd.item())

            # forward channel loss
            feat_t_fwd_list = selectInterlayer111(feat_t_fwd, args.blocks_amount_t, args)
            feat_s_fwd_list = selectInterlayer111(feat_s_fwd, args.blocks_amount_s, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_s_fwd, labels_fwd_batch, criterion_cls, criterion_div,  args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd['self_supervised_fwd']
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
                args.trainable_ae_t_dict.train()
                args.trainable_ae_s_dict.eval()
                args.trainable_auxcf_t_dict.train()
                args.trainable_auxcf_s_dict.eval()
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

                # train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # # convert to torch Variables
                # train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)
                train_fb_batch = train_fb_batch.to(args.device, non_blocking=True)
                labels_fb_batch = labels_fb_batch.to(args.device, non_blocking=True)

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

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, labels_fb_batch, criterion_cls, criterion_div, loss_div_fb, model_t_weights, args)

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

def train_our_distill_method12(model_t, model_s, model_auxcfae_t, model_auxcfae_s, trainloader_fwd, trainloader_fb, epoch, args):
    each_epoch_start_time = time.time()
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)

    train_dl_fwd = trainloader_fwd
    train_dl_fb = trainloader_fb
    # model_t = args.module_dict['model_t']
    # model_s = args.module_dict['model_s']
    # set teacher as eval
    model_t.eval()
    # set student as train
    model_s.train()

    criterion_cls = args.criterion_dict['cri_cls']
    criterion_div = args.criterion_dict['cri_div']
    criterion_inter_fwd = args.criterion_dict['cri_infwd']
    criterion_inter_fb = args.criterion_dict['cri_infb']

    losses_fwd_ae = AverageMeter('Loss_fwd_ae', ':.4e', Summary.NONE)
    losses_fwd_distill = AverageMeter('Loss_fwd_dis', ':.4e', Summary.NONE)
    losses_fwd = AverageMeter('Loss_fwd_all', ':.4e', Summary.NONE)
    top1_s = AverageMeter('Acc@1_S', ':6.2f', Summary.AVERAGE)
    top5_s = AverageMeter('Acc@5_S', ':6.2f', Summary.AVERAGE)

    losses_fb_ae = AverageMeter('Loss_fb_ae', ':.4e', Summary.NONE)
    losses_fb_distill = AverageMeter('Loss_fb_dis', ':.4e', Summary.NONE)
    losses_fb = AverageMeter('Loss_fb_all', ':.4e', Summary.NONE)
    top1_t = AverageMeter('Acc@1_T', ':6.2f', Summary.AVERAGE)
    top5_t = AverageMeter('Acc@5_T', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(train_dl_fwd) + (args.distributed and (len(train_dl_fwd.sampler) * args.world_size < len(train_dl_fwd.dataset))),
        [batch_time, losses_fwd_ae, losses_fwd_distill, losses_fwd, top1_s, losses_fb_ae, losses_fb_distill, losses_fb, top1_t],
        prefix='Train: ')

    model_t.zero_grad()

    model_t_weights = []
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            model_t_weights.append(param)
    with tqdm(total=len(train_dl_fwd)) as t:
        for i, ((train_fwd_batch, labels_fwd_batch), (train_fb_batch, labels_fb_batch)) in enumerate(zip(train_dl_fwd, train_dl_fb)):
            # ===================Forward Distill=====================
            model_s.train()
            model_t.eval()
            [each.eval() for each in model_auxcfae_t]
            [each.train() for each in model_auxcfae_s]

            train_fwd_batch = train_fwd_batch.to(args.device, non_blocking=True)
            labels_fwd_batch = labels_fwd_batch.to(args.device, non_blocking=True)

            feat_s_fwd, output_s_fwd = model_s(train_fwd_batch, is_feat=True)

            with torch.no_grad():
                feat_t_fwd, output_t_fwd = model_t(train_fwd_batch, is_feat=True)

            loss_div_fwd = criterion_div(output_s_fwd, output_t_fwd)

            # forward channel loss
            feat_t_fwd_list = selectInterlayer111(feat_t_fwd, args.blocks_amount_t, args)
            feat_s_fwd_list = selectInterlayer111(feat_s_fwd, args.blocks_amount_s, args)

            # lpp_test.test(feat_t_fwd_list, feat_s_fwd_list, args)
            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_s_fwd, labels_fwd_batch,
                                              criterion_cls, criterion_div, model_auxcfae_t, model_auxcfae_s, args)
            # loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, labels_fwd_batch, args)

            loss_fwd_ae = loss_ch_fwd['recon_S_fwd']
            loss_fwd_distill = args.kl_loss * loss_div_fwd + args.infwd_loss * loss_ch_fwd['fusion_fwd'] + loss_ch_fwd['self_supervised_fwd']
            loss_fwd = loss_fwd_distill + loss_fwd_ae

            acc1_s, acc5_s = utils.accuracy(output_s_fwd, labels_fwd_batch, topk=(1, 5))
            losses_fwd_ae.update(loss_fwd_ae.item(), train_fwd_batch.size(0))
            losses_fwd_distill.update(loss_fwd_distill.item(), train_fwd_batch.size(0))
            losses_fwd.update(loss_fwd.item(), train_fwd_batch.size(0))
            top1_s.update(acc1_s[0], train_fwd_batch.size(0))
            top5_s.update(acc5_s[0], train_fwd_batch.size(0))

            # Forward distill Auto-Encoder backward
            args.optimizer_dict['opt_fwd'].zero_grad()
            args.optimizer_dict['opt_ae_s'].zero_grad()
            args.optimizer_dict['opt_auxcf_s'].zero_grad()
            # loss_fwd_ae.backward(retain_graph=True)
            loss_fwd.backward()
            args.optimizer_dict['opt_fwd'].step()
            args.optimizer_dict['opt_ae_s'].step()
            args.optimizer_dict['opt_auxcf_s'].step()
            # args.optimizer_dict['opt_fwd_ae'].step()

            # ===================Feedback Distill=====================
            if epoch >= (args.feedback_time - 1):
                model_t.train()
                model_s.eval()
                [each.train() for each in model_auxcfae_t]
                [each.eval() for each in model_auxcfae_s]
                model_t.zero_grad()

                train_fb_batch = train_fb_batch.to(args.device, non_blocking=True)
                labels_fb_batch = labels_fb_batch.to(args.device, non_blocking=True)

                feat_t_fb, output_t_fb = model_t(train_fb_batch, is_feat=True)

                with torch.no_grad():
                    feat_s_fb, output_s_fb = model_s(train_fb_batch, is_feat=True)

                loss_div_fb = criterion_div(output_t_fb, output_s_fb)
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

            # batch_time.update(time.time() - each_epoch_start_time)
            # each_epoch_start_time = time.time()
            #
            # if i % args.print_freq == 0:
            #     progress.display(i)

            if epoch >= args.feedback_time:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd
                del train_fb_batch, labels_fb_batch, feat_s_fb, output_s_fb, feat_t_fb, output_t_fb, loss_ch_fb, loss_fb, loss_div_fb
            else:
                del train_fwd_batch, labels_fwd_batch, feat_s_fwd, output_s_fwd, feat_t_fwd, output_t_fwd, loss_ch_fwd, loss_fwd, loss_div_fwd

    if args.distributed:
        dist.barrier()

    if args.distributed:
        losses_fwd_ae.all_reduce()
        losses_fwd_distill.all_reduce()
        losses_fwd.all_reduce()
        losses_fb_ae.all_reduce()
        losses_fb_distill.all_reduce()
        losses_fb.all_reduce()
        top1_t.all_reduce()
        top1_s.all_reduce()
        top5_t.all_reduce()
        top5_s.all_reduce()

    if epoch >= args.feedback_time:
        train_str = "- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, Forward training loss:{losses_fwd.avg:.4f}. \t Train Teacher accuracy Acc@1: {top1_t.avg:.4f}, Acc@5:{top5_t.avg:.4f}, Feedback training loss:{losses_fdbk.avg:.4f}.".format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd, top1_t=top1_t, top5_t=top5_t,
                        losses_fdbk=losses_fb)

    else:
        train_str ="- Train Student accuracy Acc@1: {top1_s.avg:.4f}, Acc@5:{top5_s.avg:.4f}, forward training loss:{losses_fwd.avg:.4f}.".format(top1_s=top1_s, top5_s=top5_s, losses_fwd=losses_fwd)
    if is_main_process():
        logger = logging.getLogger()
        logger.parent = None
        logger.info(train_str)
    # args.txt_f.write(train_str + '\n')
    # print(train_str)

    # progress.display_summary()

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
            args.trainable_ae_t_dict.eval()
            args.trainable_ae_s_dict.train()
            args.trainable_auxcf_t_dict.eval()
            args.trainable_auxcf_s_dict.train()
            # for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
            #     args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].eval()
            # for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
            #     args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].train()

            # train_fwd_batch, labels_fwd_batch = train_fwd_batch.cuda(), labels_fwd_batch.cuda()
            # # convert to torch Variables
            # train_fwd_batch, labels_fwd_batch = Variable(train_fwd_batch), Variable(labels_fwd_batch)
            train_fwd_batch = train_fwd_batch.to(args.device, non_blocking=True)
            labels_fwd_batch = labels_fwd_batch.to(args.device, non_blocking=True)

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

            loss_ch_fwd = criterion_inter_fwd(feat_t_fwd_list, feat_s_fwd_list, output_t_fwd, output_s_fwd, labels_fwd_batch, criterion_cls, criterion_div,  args)

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
                args.trainable_ae_t_dict.train()
                args.trainable_ae_s_dict.eval()
                args.trainable_auxcf_t_dict.train()
                args.trainable_auxcf_s_dict.eval()
                # for each_trainable__auxcf_t_name in args.trainable_auxcf_t_dict.keys():
                #     args.trainable_auxcf_t_dict[each_trainable__auxcf_t_name].train()
                # for each_trainable_auxcf_s_name in args.trainable_auxcf_s_dict.keys():
                #     args.trainable_auxcf_s_dict[each_trainable_auxcf_s_name].eval()

                # train_fb_batch, labels_fb_batch = train_fb_batch.cuda(), labels_fb_batch.cuda()
                # # convert to torch Variables
                # train_fb_batch, labels_fb_batch = Variable(train_fb_batch), Variable(labels_fb_batch)
                train_fb_batch = train_fb_batch.to(args.device, non_blocking=True)
                labels_fb_batch = labels_fb_batch.to(args.device, non_blocking=True)

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

                loss_ch_fb = criterion_inter_fb(feat_t_fb_list, feat_s_fb_list, output_t_fb, output_s_fb, labels_fb_batch, criterion_cls, criterion_div, loss_div_fb, model_t_weights, args)

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

    return feat_list
