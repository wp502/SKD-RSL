# -*- coding: utf-8 -*

import os
import pickle

import PIL.Image
import torch
import scipy.io as io
import numpy as np
from pathlib import Path
import shutil


__all__ = ['CarstoFolder']


class CarstoFolder(torch.utils.data.Dataset):
    """Cars dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """
    def __init__(self, root, download=False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._anno_filename = 'cars_annos.mat'
        self._class_txt = 'classes.txt'
        self._img_pth_split_txt = 'img_pth_train_test_split.txt'
        self._anno_filename = 'cars_annos.mat'
        self._img_filename = 'car_ims.tgz'
        self._train_folder = 'train'
        self._test_folder = 'test'

        anno_file_pth = os.path.join(self._root, self._anno_filename)
        img_file_tgz_pth = os.path.join(self._root, self._img_filename)
        img_folder_pth = os.path.join(self._root, 'car_ims')

        if self._checkIntegrity():
            print('Files already downloaded and verified.')
        else:
            if os.path.exists(anno_file_pth) and os.path.exists(img_file_tgz_pth):
                if os.path.exists(img_folder_pth):
                    self._extract_folder()
                    # self._extract()
                else:
                    raise RuntimeError(
                        'Have found dataset, but not decompression')
            else:
                if download:
                    url = None
                    self._download(url)
                else:
                    raise RuntimeError(
                        'Dataset not found. You can use download=True to download it.')

    def _checkIntegrity(self):
        """Check whether we have already processed the data.

        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (
                os.path.exists(os.path.join(self._root, self._train_folder))
                and os.path.exists(os.path.join(self._root, self._test_folder)))

    def _download(self, url):
        raise NotImplementedError

    def _extract_folder(self):
        self._extract_basicinfo()
        image_path = os.path.join(self._root, 'car_ims')
        # Format of classes.txt: <class_num> <class_name>
        img_classes_name = np.genfromtxt(os.path.join(
            self._root, self._class_txt), dtype=str)
        # Format of img_pth_train_test_split.txt: <img_id> <img_pth> <img_class> <is_training_image>
        img_pth_split_info = np.genfromtxt(os.path.join(
            self._root, self._img_pth_split_txt), dtype=str)
        train_folder_pth = os.path.join(self._root, self._train_folder)
        test_folder_pth = os.path.join(self._root, self._test_folder)
        if not os.path.isdir(train_folder_pth):
            os.mkdir(train_folder_pth)
        if not os.path.isdir(test_folder_pth):
            os.mkdir(test_folder_pth)
        for each_name in img_classes_name:
            os.mkdir(os.path.join(train_folder_pth, each_name[1]))
            os.mkdir(os.path.join(test_folder_pth, each_name[1]))
        for idx, each_img in enumerate(img_pth_split_info):
            img_pth = each_img[1]
            img_class = each_img[2]
            img_is_test = each_img[3]
            img_class_name = img_classes_name[int(img_class)-1][1]
            source = os.path.join(self._root, img_pth)
            is_train_pth = train_folder_pth
            is_test_pth = test_folder_pth
            
            if int(img_is_test) == 1:
                target = os.path.join(is_test_pth, img_class_name)
                shutil.copy(source, target)
            else:
                target = os.path.join(is_train_pth, img_class_name)
                shutil.copy(source, target)

    def _extract_basicinfo(self):
        anno_data = io.loadmat(os.path.join(self._root, self._anno_filename))
        labels = anno_data['annotations']
        class_names = anno_data['class_names']

        class_num = 1
        with open(os.path.join(self._root, self._class_txt), 'w') as f:
            for i in range(class_names.shape[1]):
                class_name = str(class_names[0, i][0]).replace(' ', '_')
                if '/' in class_name:
                    class_name = class_name.replace('/', '_')
                f.write(str(class_num) + ' ' + class_name + '\n')
                class_num += 1

        img_num = 1
        with open(os.path.join(self._root, self._img_pth_split_txt), 'w') as f:
            for j in range(labels.shape[1]):
                pth = str(labels[0, j][0])[2: -2]
                test = int(labels[0, j][6])
                clas = int(labels[0, j][5])
                f.write(str(img_num) + ' ' + str(pth) + ' ' + str(clas) + ' ' + str(test) + '\n')
                img_num += 1

