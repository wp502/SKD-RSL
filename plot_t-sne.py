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
from zoo import DistillKL, CCSLoss, ICKDLoss, SPKDLoss, CRDLoss, KDLossv2, CDLoss
from tensorboardX import SummaryWriter
# from dataloader import fetch_dataloader
# from dataloader import fetch_dataloader_1, fetch_dataloader_2
from dataloader import fetch_dataloader_2 as fetch_dataloader
import tensorboard_logger
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import csv


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')

    parser.add_argument('--seed', type=str, default=None, help='random seed')
    parser.add_argument('--classesList', type=str, default=None, help='[For STEP2 after completing Step1]Specify the classes list')
    parser.add_argument('--tsneWay', type=int, default=1, help='choice which tsne method')

    # model
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--model_pth', type=str, default=None, help='evaluate model snapshot')
    parser.add_argument('--distill', type=str, default='our')
    parser.add_argument('--pin_memory', action='store_false', default=True,
                        help="flag for whether use pin_memory in dataloader")

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10', 'tiny_imagenet', 'cub200', 'cars196'], help='dataset')
    parser.add_argument('--dataset_dir', type=str, default=None, help='whether appoint dataset dir path')
    parser.add_argument('--num_class', default=100,
                        type=int, help="number of classes")
    parser.add_argument('--augmentation', type=str,
                        default='yes', help='dataset augmentation')
    parser.add_argument('--subset_percent', type=float,
                        default=1.0, help='subset_percent')
    parser.add_argument('--num_extract', default=10,
                        type=int, help="number of extract classes")


    args = parser.parse_args()

    return args


def load_model(model_path, n_cls, model_t):
    print('==> loading model')
    # model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls).cuda()
    # model.load_state_dict(torch.load(model_path)['state_dict'])  # ['state_dict']
    model.load_state_dict(torch.load(model_path)['state_dict'])  # ['state_dict']
    print('==> done')
    return model


def getEmbedding(model, dev_dl, classes, args):
    if torch.cuda.is_available():
        model.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True

    model.eval()

    embeddings_list = []
    labels_list = []
    classes_list = []
    # class_name_list = []
    # for i in range(len(classes)):
    #     classes_dict[int(classes[i])] = dev_dl.dataset.classes[int(classes[i])]

    # for i, each in enumerate(classes):
    #     classes_list.append(dev_dl.dataset.classes[int(each)])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dev_dl):
            inputs, targets = inputs.cuda(), targets.cuda()
            # print(targets.item())
            if np.sum(classes == targets.item()) == 1:
                # features = model(inputs)
                # embedding是在linear层之前的一层。
                features, outputs = model(inputs, is_feat=True)
                embeddings_list.append(features[-1].squeeze(0).cpu().numpy())
                labels_list.append(targets.item())
                classes_list.append(dev_dl.dataset.classes[targets.item()])

    embeddings_list = np.asarray(embeddings_list)
    labels_list = np.asarray(labels_list)
    classes_list = np.asarray(classes_list)
    np.save(args.embedding_list_pth, embeddings_list)
    np.save(args.label_list_pth, labels_list)
    np.save(args.class_list_pth, classes_list)


def plt_tsne(embeddings_list, labels_list, classes_list, args, tsneWay=1):


    class_name_list = []
    for each in classes_list:
        if each not in class_name_list:
            class_name_list.append(each)
    embedding_dict = dict.fromkeys(class_name_list)
    # class_embedding_list = [[] for i in range(int(args.num_extract))]

    print(embeddings_list.shape)
    print(labels_list.shape)
    print(classes_list.shape)

    if tsneWay == 1:
        X_tsne = TSNE(n_components=2, perplexity=30, n_iter=800, learning_rate=80).fit_transform(embeddings_list)
    elif tsneWay == 2:
        X_tsne = TSNE(n_components=2, perplexity=30, n_iter=500, learning_rate=60).fit_transform(embeddings_list)  # 自己的
    elif tsneWay == 3:
        X_tsne = TSNE(n_components=2, perplexity=30, n_iter=500, learning_rate=40).fit_transform(embeddings_list)  # 对比算法
    # X_tsne = TSNE().fit_transform(embeddings_list)
    args.save_fig_pth = args.save_folder + '/tsne_' + str(
        args.dataset) + '_' + str(args.distill) + '_M-' + str(args.model) + '_' + str(
        args.model_folder) + '_Seed-' + str(
        args.seed) + '-' + str(tsneWay) + '.png'
    args.save_eps_pth = args.save_folder + '/tsne_' + str(
        args.dataset) + '_' + str(args.distill) + '_M-' + str(args.model) + '_' + str(
        args.model_folder) + '_Seed-' + str(
        args.seed) + '-' + str(tsneWay) + '.eps'

    for idx, each_X in enumerate(X_tsne):
        each_label = labels_list[idx]
        each_class = classes_list[idx]
        each_class_name = each_class
        if embedding_dict[each_class_name] is None:
            embedding_dict[each_class_name] = [each_X]
        else:
            embedding_dict[each_class_name].append(each_X)
    plt.figure(figsize=(10, 10))
    scatter_list = []
    for each_item in embedding_dict.items():
        each_item_values = np.asarray(each_item[1])
        scatter = plt.scatter(each_item_values[:, 0], each_item_values[:, 1], cmap='tab10', ec='w')
        scatter_list.append(scatter)
    # plt.title(str(args.distill).upper())
    plt.xticks([])
    plt.yticks([])
    # plt.legend(*scatter.legend_elements(), loc='best')
    plt.legend(scatter_list, class_name_list, loc='best')
    plt.savefig(args.save_fig_pth, format='png', dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(args.save_eps_pth, format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.show()

def load_data_model(args):
    model_pth = args.model_pth
    model_pth_split = model_pth.split('/')[-1]
    print("==>Evaluate Model:" + args.model + "\t" + model_pth_split + "\tDataset:" + args.dataset)
    print("==>Loading the datasets...")
    train_dl = fetch_dataloader('train', args)
    dev_dl = fetch_dataloader('dev', args)
    print("==>- done.")
    model = load_model(args.model_pth, args.num_class, args.model).cuda()
    return model, train_dl, dev_dl

def getACCForEachClass(model, dev_dl, args):
    # 获得模型在测试集上，每个类的精度
    if torch.cuda.is_available():
        model.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True

    # class_acc_record_str = []
    classes = dev_dl.dataset.classes
    class_to_idx = dev_dl.dataset.class_to_idx

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    accuracy_pred = {classname: 0 for classname in classes}

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dev_dl):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            _, predictions = torch.max(outputs, 1)
            for target, prediction in zip(targets, predictions):
                if target == prediction:
                    correct_pred[classes[target]] += 1
                total_pred[classes[target]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100.0 * float(correct_count / total_pred[classname])
        print_str = "Accuracy for class [{:5s}, idx:{:3d}] is {:.2f}%".format(classname, class_to_idx[classname], accuracy)
        # class_acc_record_str.append(print_str)
        accuracy_pred[classname] = accuracy
        print(print_str)

    with open(args.class_acc_csv_pth, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ClassName', 'Index', 'ACC@1'])
        # writer.writerow(class_acc_record_str)
        for classname in classes:
            writer.writerow([classname, 'IDX:'+str(class_to_idx[classname]), str(accuracy_pred[classname])+'%'])


def main():
    args = parse_option()
    if args.seed != 'None':
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.cuda.manual_seed(int(args.seed))

    warnings.filterwarnings("ignore")

    # classes = np.random.choice(np.arange(args.num_class), args.num_extract, replace=False)
    # classes = np.asarray([1, 24, 34, 39, 40, 46, 61, 90, 92, 96])
    # classes = np.asarray([94, 61, 5, 17, 29, 39, 76, 20, 0, 11]) # best_1 [FOR WORK1]
    # classes = np.asarray([94, 61, 59, 17, 29, 58, 76, 8, 0, 11]) # best_2 [FOR WORK1]
    # classes = np.asarray([94, 61, 5, 17, 29, 39, 76, 20, 57, 11]) # best_3 [FOR WORK1]
    # classes = np.asarray([94, 61, 0, 17, 8, 75, 76, 20, 79, 11])  # best_4 [FOR WORK1]

    # classes = np.asarray([13, 87, 96, 83, 97, 85, 18, 24, 33, 39]) # [FOR WORK2]
    # classes = np.asarray([87, 97, 85, 24, 39, 57, 6, 66, 79, 8]) # [FOR WORK2]
    classes = np.asarray([87, 85, 24, 39, 8, 23, 54, 75, 0, 89]) # [FOR WORK2]
    # print(classes)
    classes_str = ""
    for each in classes:
        classes_str = classes_str + str(each) + "_"

    classes_str = classes_str[:-1]

    if args.classesList != 'None':
        classes_str = args.classesList
        # classes = [int(each) for each in classes_str.split("_")]

    print(classes)
    print(classes_str)

    args.model_folder = args.model_pth.split('/')[-2] + "-C_" + classes_str
    args.save_folder = './visualizations/plt_tsne_save/' + str(args.distill) + '/' + str(args.model_folder) + '/'
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.embedding_list_pth = args.save_folder + '/embeddings_' + str(
        args.dataset) + '_' + str(args.distill) + '_M-' + str(args.model) + '_' + str(
        args.model_folder) + '_Seed-' + str(
        args.seed) + '.npy'
    args.label_list_pth = args.save_folder + '/labels_' + str(
        args.dataset) + '_' + str(args.distill) + '_M-' + str(args.model) + '_' + str(
        args.model_folder) + '_Seed-' + str(
        args.seed) + '.npy'
    args.class_list_pth = args.save_folder + '/classes_' + str(
        args.dataset) + '_' + str(args.distill) + '_M-' + str(args.model) + '_' + str(
        args.model_folder) + '_Seed-' + str(
        args.seed) + '.npy'
    args.class_acc_csv_pth = args.save_folder + '/class_acc_' + str(
        args.dataset) + '_' + str(args.distill) + '_M-' + str(args.model) + '_' + str(
        args.model_folder) + '_Seed-' + str(
        args.seed) + '.csv'

    # Step1和Step2只能运行其中一个
    # Step1
    # model, train_dl, dev_dl = load_data_model(args)
    # # getACCForEachClass(model, dev_dl, args)
    # dev_dl.dataset.class_to_idx['bicycle'] # 查找类名对应的idx
    # getEmbedding(model, dev_dl, classes, args)

    # Step2
    embeddings_list = np.load(args.embedding_list_pth)
    labels_list = np.load(args.label_list_pth)
    classes_list = np.load(args.class_list_pth)

    plt_tsne(embeddings_list, labels_list, classes_list, args, tsneWay=1)
    plt_tsne(embeddings_list, labels_list, classes_list, args, tsneWay=2)
    plt_tsne(embeddings_list, labels_list, classes_list, args, tsneWay=3)



if __name__ == '__main__':
    main()
