import logging
import torch
import numpy as np
import os
import json
import shutil
import time
import torch.nn as nn
from enum import Enum
from torch.utils.data import Subset
import torch.distributed as dist

def set_logger(log_path):
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_path),
                                  logging.StreamHandler(os.sys.stdout)])

    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    #
    # if not logger.handlers:
    #     # Logging to a file
    #     file_handler = logging.FileHandler(log_path)
    #     file_handler.setFormatter(logging.Formatter(
    #         '%(asctime)s:%(levelname)s: %(message)s'))
    #     logger.addHandler(file_handler)
    #
    #     # Logging to console
    #     stream_handler = logging.StreamHandler()
    #     stream_handler.setFormatter(logging.Formatter('%(message)s'))
    #     logger.addHandler(stream_handler)
    #
    # return logger

def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                # if torch.backends.mps.is_available():
                #     images = images.to('mps')
                #     target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    if is_main_process():
                        progress.display(i + 1)

        if args.distributed:
            dist.barrier()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if is_main_process():
        progress.display_summary()
        logger = logging.getLogger()
        logger.parent = None
        logger.info("- Eval metrics, Acc@1:{top1.avg:.3f}, Acc@5:{top5.avg:.4f}, loss: {loss.avg:.4f}".format(
            top1=top1, top5=top5, loss=losses))

    my_metric = {'top1': top1.avg, 'top5': top5.avg, 'loss': losses.avg}

    return my_metric

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_checkpoint(state, is_best, args, save_folder = 'None', is_teacher = False, name=None):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """

    if is_teacher:
        last_filepath = os.path.join(save_folder, name + '_teacher_last.pth.tar')
    else:
        last_filepath = os.path.join(save_folder, name + '_last.pth.tar')

    torch.save(state, last_filepath)
    if is_best:
        if is_teacher:
            shutil.copyfile(last_filepath, os.path.join(save_folder, name + '_teacher_best.pth.tar'))
        else:
            shutil.copyfile(last_filepath, os.path.join(save_folder, name + '_best.pth.tar'))

    if args.is_checkpoint:
        epoch = state['epoch'] + 1
        acc = state['acc']
        acc = str('%.2f' % acc)
        if epoch >= (args.epochs-args.save_checkpoint_amount):
            # logging.info("Save " + str(epoch) + " epoch checkpoint")
            # 保存最后50个epoch的checkpoint
            if is_teacher:
                epoch_file = name + '_teacher_' + str(epoch) + '_checkpoint_' + str(acc) + '.pth.tar'
            else:
                epoch_file = name + '_' + str(epoch) + '_checkpoint_' + str(acc) + '.pth.tar'
            # shutil.copyfile(last_filepath, os.path.join(save_folder, epoch_file))
            shutil.copyfile(last_filepath, os.path.join(args.checkpoint_save_pth, epoch_file))
    # if epoch_checkpoint == True:
    #     if is_teacher:
    #         epoch_file = name + '_teacher_' + str(state['epoch']) + '.pth.tar'
    #     else:
    #         epoch_file = name + '_' + str(state['epoch']) + '.pth.tar'
    #     shutil.copyfile(last_filepath, os.path.join(save_folder, epoch_file))
        
def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'a') as f:
    # with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
