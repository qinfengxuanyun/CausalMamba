import os
import torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import cv2
import csv
import random
from scipy import stats
import nibabel as nib
from collections import defaultdict

def split_age_groups(age_dict, sub_list, k):
    # 对年龄进行排序
    ages = [age_dict[i] for i in sub_list]
    sorted_ages = np.argsort(ages)

    # 创建空的二维列表
    age_groups = [[] for _ in range(k)]

    # 遍历排序后的年龄列表，将年龄添加到相应的分组中
    current_group = 0
    for age in sorted_ages:
        age_groups[current_group].append(sub_list[age])
        current_group = (current_group + 1) % k  # 循环填充每一列

    sub_list_age = []
    for group in age_groups:
        sub_list_age += group
    return sub_list_age

class MRIandGenedataset(Dataset):
    def __init__(self, label=-1, fold=0, k=5, phase="train", data_path="/data/qinfeng/datasets/",
                 gene_path="/data/qinfeng/datasets/ADNIALL_UKB/"):  # ADNIALL_SNPs2 ADNIALL_SNPs ADNIALL_PPMI_IMMUNO_fuse
        self.cls = label
        self.fold = fold
        self.path = data_path + "ADNIALL_2mm/AAL_ROISignals/"
        self.path2 = data_path + "ADNIALL/T1ImgMNI_cropwhite3/"
        self.gene_path = gene_path
        self.csv_path = data_path + "ADNIALL/label.csv"
        data = pd.read_csv(self.csv_path).values.tolist()
        self.subject = [str(i[0]) for i in data]
        self.cn_subject = [str(i[0]) for i in data if i[3] == 0]
        self.ad_subject = [str(i[0]) for i in data if i[3] == 1]
        self.emci_subject = [str(i[0]) for i in data if i[3] == 2]
        self.lmci_subject = [str(i[0]) for i in data if i[3] == 3]
        label_dict = {}
        age_dict = {}
        sex_dict = {}
        for index, sub in enumerate(self.subject):
            label_dict[sub] = data[index][3]
            age_dict[sub] = data[index][1]
            sex_dict[sub] = data[index][2]
        self.age_dict = age_dict
        self.sex_dict = sex_dict

        self.gene_dict = {}
        gene_sub_list = [file.split(".")[0] for file in os.listdir(gene_path)]

        self.cn_subject = list(set(self.cn_subject).intersection(set(gene_sub_list)))

        data = pd.read_csv(data_path + "Data_info_selected_all.csv", usecols=["Subj Name"]).values.tolist()
        sub_list_liu = ["_".join(row[0].split("_")[1:4]) for row in data]
        self.cn_subject = list(set(self.cn_subject).intersection(set(sub_list_liu)))
        self.cn_subject.sort()

        self.emci_subject = list(set(self.emci_subject).intersection(set(gene_sub_list)))
        self.emci_subject.sort()

        self.lmci_subject = list(set(self.lmci_subject).intersection(set(gene_sub_list)))
        self.lmci_subject.sort()

        # self.ad_subject = list(set(self.ad_subject).intersection(set(ad_list)))
        # self.ad_subject.sort()

        self.cn_subject = split_age_groups(age_dict, self.cn_subject, k)
        self.ad_subject = split_age_groups(age_dict, self.ad_subject, k)
        self.emci_subject = split_age_groups(age_dict, self.emci_subject, k)
        self.lmci_subject = split_age_groups(age_dict, self.lmci_subject, k)

        self.subject_list = []
        self.add_list = []
        self.add_list2 = []
        self.add_list3 = []
        self.add_list4 = []
        assert k > 1
        fold_size = len(self.cn_subject) // k  # 每份的个数:数据总条数/折数（组数）
        fold_size2 = len(self.ad_subject) // k  # 每份的个数:数据总条数/折数（组数）
        fold_size3 = len(self.emci_subject) // k  # 每份的个数:数据总条数/折数（组数）
        fold_size4 = len(self.lmci_subject) // k  # 每份的个数:数据总条数/折数（组数）

        for j in range(k):
            # idx = slice(j * fold_size, (j + 1) * fold_size)
            # idx2 = slice(j * fold_size2, (j + 1) * fold_size2)
            if j == k - 1:
                idx = slice(j * fold_size, len(self.cn_subject))
                idx2 = slice(j * fold_size2, len(self.ad_subject))
                idx3 = slice(j * fold_size3, len(self.emci_subject))
                idx4 = slice(j * fold_size4, len(self.lmci_subject))
            else:
                idx = slice(j * fold_size, (j + 1) * fold_size)
                idx2 = slice(j * fold_size2, (j + 1) * fold_size2)
                idx3 = slice(j * fold_size3, (j + 1) * fold_size3)
                idx4 = slice(j * fold_size4, (j + 1) * fold_size4)

            if phase == "train":
                if j is not fold:
                    add_list = self.cn_subject[idx]
                    add_list2 = self.ad_subject[idx2]
                    add_list3 = self.emci_subject[idx3]
                    add_list4 = self.lmci_subject[idx4]
                    self.add_list = self.add_list + add_list
                    self.add_list2 = self.add_list2 + add_list2
                    self.add_list3 = self.add_list3 + add_list3
                    self.add_list4 = self.add_list4 + add_list4
                    if self.cls == 0:
                        self.subject_list = self.subject_list + add_list
                    elif self.cls == 1:
                        self.subject_list = self.subject_list + add_list2
                    elif self.cls == 2:
                        self.subject_list = self.subject_list + add_list3
                    elif self.cls == 3:
                        self.subject_list = self.subject_list + add_list4
                    elif self.cls == [0, 1]:
                        self.subject_list = self.subject_list + add_list + add_list2
                    elif self.cls == [0, 2]:
                        self.subject_list = self.subject_list + add_list + add_list3
                    elif self.cls == [0, 3]:
                        self.subject_list = self.subject_list + add_list + add_list4
                    else:
                        self.subject_list = self.subject_list + add_list + add_list2 + add_list3 + add_list4
            elif phase == "all":
                add_list = self.cn_subject[idx]
                add_list2 = self.ad_subject[idx2]
                add_list3 = self.emci_subject[idx3]
                add_list4 = self.lmci_subject[idx4]
                self.add_list = self.add_list + add_list
                self.add_list2 = self.add_list2 + add_list2
                self.add_list3 = self.add_list3 + add_list3
                self.add_list4 = self.add_list4 + add_list4
                if self.cls == 0:
                    self.subject_list = self.subject_list + add_list
                elif self.cls == 1:
                    self.subject_list = self.subject_list + add_list2
                elif self.cls == 2:
                    self.subject_list = self.subject_list + add_list3
                elif self.cls == 3:
                    self.subject_list = self.subject_list + add_list4
                elif self.cls == [0, 1]:
                    self.subject_list = self.subject_list + add_list + add_list2
                elif self.cls == [0, 2]:
                    self.subject_list = self.subject_list + add_list + add_list3
                elif self.cls == [0, 3]:
                    self.subject_list = self.subject_list + add_list + add_list4
                else:
                    self.subject_list = self.subject_list + add_list + add_list2 + add_list3 + add_list4
            else:
                if j == fold:  ###第i折作valid
                    add_list = self.cn_subject[idx]
                    add_list2 = self.ad_subject[idx2]
                    add_list3 = self.emci_subject[idx3]
                    add_list4 = self.lmci_subject[idx4]
                    self.add_list = self.add_list + add_list
                    self.add_list2 = self.add_list2 + add_list2
                    self.add_list3 = self.add_list3 + add_list3
                    self.add_list4 = self.add_list4 + add_list4
                    if self.cls == 0:
                        self.subject_list = self.subject_list + add_list
                    elif self.cls == 1:
                        self.subject_list = self.subject_list + add_list2
                    elif self.cls == 2:
                        self.subject_list = self.subject_list + add_list3
                    elif self.cls == 3:
                        self.subject_list = self.subject_list + add_list4
                    elif self.cls == [0, 1]:
                        self.subject_list = self.subject_list + add_list + add_list2
                    elif self.cls == [0, 2]:
                        self.subject_list = self.subject_list + add_list + add_list3
                    elif self.cls == [0, 3]:
                        self.subject_list = self.subject_list + add_list + add_list4
                    else:
                        self.subject_list = self.subject_list + add_list + add_list2 + add_list3 + add_list4

        self.subject_list = list(set(self.subject_list).intersection(set(gene_sub_list)))

        self.files = []
        self.label = []
        for file in os.listdir(self.path2):
            sub = file.split('.')[0]
            if sub in self.subject_list:
                self.files.append(file)
                self.label.append(label_dict[sub])
        # The LabelEncoder encodes a sequence of bases as a sequence of integers.
        self.integer_encoder = LabelEncoder()
        # The OneHotEncoder converts an array of integers to a sparse matrix where
        self.one_hot_encoder = OneHotEncoder(categories='auto')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fid = self.files[index]
        sub = fid.split('.')[0]
        age_sex = np.array([self.age_dict[sub] / 100, self.sex_dict[sub]]).astype(np.float32)
        t1_data = self.get_img(sub + ".nii.gz")
        label = np.array(self.label[index]).astype(np.int64)
        label[label != 0] = 1
        data_range = np.load(f"adni_t1_range.npy").astype(np.float32)
        sequence = pd.read_csv(self.gene_path + sub + ".csv", usecols=['value']).values[:, 0].tolist()
        integer_encoded = self.integer_encoder.fit_transform(sequence)
        integer_encoded = integer_encoded.astype(np.int64)
        return fid, t1_data, label, data_range, age_sex, integer_encoded

    def get_img(self, fid):
        sub_path = self.path2 + fid
        img = self.nii_loader(sub_path)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return img

    def nii_loader(self, path):
        img = nib.load(str(path))
        data = img.get_fdata()
        return data

class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.grouped_indices = defaultdict(list)

        # for idx in range(len(dataset)):
        #     item = dataset[idx]
        #     key = (int(item[-2][0]*100), item[-2][1])
        #     key = (item[-2][1])
        for idx, fid in enumerate(dataset.files):
            sub = fid.split('.')[0]
            key = (int(dataset.age_dict[sub]) // 10, int(dataset.sex_dict[sub]))
            self.grouped_indices[key].append(idx)

        self.batches = []
        for key, indices in self.grouped_indices.items():
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i:i + self.batch_size])

    def __iter__(self):
        self.batches = []
        for key, indices in self.grouped_indices.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i:i + self.batch_size])
        random.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
