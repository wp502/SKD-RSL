import logging
import torch
import numpy as np
import os
import json
import shutil
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


def set_logger(log_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

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
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def recall(embeddings, labels, K=[]):
    batch_size = labels.size(0)
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    recall_k = []

    for k in K:
        # correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        # recall_k.append(correct_k)
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean()
        # correct_k = correct_labels[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        # correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().sum(0, keepdim=True)
        # recall_k.append(correct_k.mul_(100.0 / batch_size))
        recall_k.append(correct_k)
    return recall_k

# def save_checkpoint(state, is_best, args, save_folder = 'None', is_teacher = False, name=None):
#     """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
#     checkpoint + 'best.pth.tar'
#
#     Args:
#         state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
#         is_best: (bool) True if it is the best model seen till now
#         checkpoint: (string) folder where parameters are to be saved
#     """
#
#     if is_teacher:
#         last_filepath = os.path.join(save_folder, name + '_teacher_last.pth.tar')
#     else:
#         last_filepath = os.path.join(save_folder, name + '_last.pth.tar')
#
#     torch.save(state, last_filepath)
#     if is_best:
#         if is_teacher:
#             shutil.copyfile(last_filepath, os.path.join(save_folder, name + '_teacher_best.pth.tar'))
#         else:
#             shutil.copyfile(last_filepath, os.path.join(save_folder, name + '_best.pth.tar'))
#
#     if args.is_checkpoint:
#         epoch = state['epoch'] + 1
#         acc = state['acc']
#         acc = str('%.2f' % acc)
#         if epoch >= (args.epochs-args.save_checkpoint_amount):
#             # logging.info("Save " + str(epoch) + " epoch checkpoint")
#             # 保存最后50个epoch的checkpoint
#             if is_teacher:
#                 epoch_file = name + '_teacher_' + str(epoch) + '_checkpoint_' + str(acc) + '.pth.tar'
#             else:
#                 epoch_file = name + '_' + str(epoch) + '_checkpoint_' + str(acc) + '.pth.tar'
#             # shutil.copyfile(last_filepath, os.path.join(save_folder, epoch_file))
#             shutil.copyfile(last_filepath, os.path.join(args.checkpoint_save_pth, epoch_file))
#     # if epoch_checkpoint == True:
#     #     if is_teacher:
#     #         epoch_file = name + '_teacher_' + str(state['epoch']) + '.pth.tar'
#     #     else:
#     #         epoch_file = name + '_' + str(state['epoch']) + '.pth.tar'
#     #     shutil.copyfile(last_filepath, os.path.join(save_folder, epoch_file))

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

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def adjust_loss_alpha(alpha, epoch, args, factor=0.9):
    """Early Decay Teacher"""

    if "imagenet" in args.dataset:
        return alpha * (factor ** (epoch // 30))
        # if args.cd_is_edt:
        #     # use EDT
        #     if "ce" in loss_type or "kd" in loss_type:
        #         return 0 if epoch <= 30 else alpha * (factor ** (epoch // 30))
        #     else:
        #         return alpha * (factor ** (epoch // 30))
        # else:
        #     return alpha * (factor ** (epoch // 30))
    else:  # cifar
        if args.cd_is_edt:
            # use EDT
            if epoch >= 160:
                exponent = 2
            elif epoch >= 60:
                exponent = 1
            else:
                exponent = 0
            return alpha * (factor ** exponent)
            # if "ce" in loss_type or "kd" in loss_type:
            #     return 0 if epoch <= 60 else alpha * (factor**exponent)
            # else:
            #     return alpha * (factor**exponent)
        else:
            return alpha
    

class DataPrefetcher():

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_target = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

def statis_params_amount(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')