#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_function import Maskcompute, GeneMLPEncoder
from t1_gene_clip_adniall_interp_dataset import MRIandGenedataset  # ,GroupedBatchSampler
from vit import MRIMambaMAE
from options.train_options import TrainOptions
from mamba_model import Mamba_dflow
import itertools
import copy
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from tensorboardX import SummaryWriter

cuda = torch.cuda.is_available()

class CLIP(nn.Module):
    def __init__(self, temperature=0.07, feature_num=512):
        super(CLIP, self).__init__()
        self.image_proj = nn.Linear(feature_num, 512)
        self.snp_proj = nn.Linear(feature_num, 512)
        self.image_pooling = nn.AdaptiveMaxPool2d((1, feature_num))  # nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.snp_pooling = nn.AdaptiveMaxPool2d((1, feature_num))  # nn.AdaptiveAvgPool2d((1,config.hidden_size))#
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.nolin = nn.ELU()
        self.norm = nn.GroupNorm(4, feature_num)

    def forward(self, image_features, snp_features, mask):
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_proj(image_features)

        snp_features = snp_features.view(snp_features.size(0), -1)
        snp_features = self.snp_proj(snp_features)

        clip_scores = F.cosine_similarity(image_features, snp_features)
        # normalized features
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        snp_features_norm = snp_features / snp_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features_norm @ snp_features_norm.t()

        loss_img = torch.sum(-1.0 * F.log_softmax(logits, dim=0) * mask / torch.sum(mask, dim=0, keepdim=True)) / logits.shape[0]
        loss_snp = torch.sum(-1.0 * F.log_softmax(logits.t(), dim=0) * mask / torch.sum(mask, dim=0, keepdim=True)) / logits.shape[0]
        loss = loss_img + loss_snp

        return loss / 2, clip_scores, image_features, snp_features

    def forward2(self, mri_features, snp_features, mri_features2, snp_features2, mask):
        # style = self.nolin(self.male_fc(age_sex))
        mri_features = mri_features.view(mri_features.size(0), -1)
        mri_features = self.image_proj(mri_features)
        snp_features = snp_features.view(snp_features.size(0), -1)
        snp_features = self.snp_proj(snp_features)

        mri_features2 = mri_features2.view(mri_features2.size(0), -1)
        mri_features2 = self.image_proj(mri_features2)
        snp_features2 = snp_features2.view(snp_features2.size(0), -1)
        snp_features2 = self.snp_proj(snp_features2)

        # normalized features
        mri_features_norm = mri_features / mri_features.norm(dim=1, keepdim=True)
        snp_features_norm = snp_features / snp_features.norm(dim=1, keepdim=True)
        mri_features2_norm = mri_features2 / mri_features2.norm(dim=1, keepdim=True)
        snp_features2_norm = snp_features2 / snp_features2.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * mri_features_norm @ snp_features_norm.t()
        logits2 = logit_scale * mri_features2_norm @ snp_features2_norm.t()

        mask = torch.cat([mask, torch.zeros_like(mask)], dim=1)
        loss_img = torch.sum(-1.0 * F.log_softmax(torch.cat([logits, logits2], dim=1), dim=1) * mask / torch.sum(mask, dim=1,keepdim=True)) / logits.shape[0]
        loss_snp = torch.sum(-1.0 * F.log_softmax(torch.cat([logits.t(), logits2.t()], dim=1), dim=1) * mask / torch.sum(mask, dim=1,keepdim=True)) / logits.shape[0]
        loss = loss_img + loss_snp

        return loss / 2

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 2)
        self.nolin = nn.ReLU()
        self.droptout = nn.Dropout(0.5)

    def forward(self, image_features):
        out = self.fc1(image_features)
        out = self.droptout(out)
        out = self.fc2(self.nolin(out))
        return out

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def topk_keep(x: torch.Tensor, k: int):
    topk_vals, _ = torch.topk(x, k, dim=-1, largest=True)
    thresh = topk_vals[..., -1, None]
    mask = (x >= thresh).to(x.dtype)  # (B, L) 中 0 / 1
    return mask + x - x.detach()  # (x * mask).detach() + x - x.detach()#

opt = TrainOptions().parse()
# initial for recurrence
seed = 8
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

criterion = nn.CrossEntropyLoss().cuda()

task = 'adcls'
pretrain_dir = f"./generation_models/ADNIALL_T1-GENE-CLIP_MAE_{task}_{opt.use_sparse + 1}_{opt.use_sparse2 + 1}_{opt.mri_th}_{opt.snp_th}"
ep=100
MODEL_PATH = f"./generation_models/ADNIALL_T1-GENE-CLIP_MAECLS_{task}_{opt.use_sparse + 1}_{opt.use_sparse2 + 1}_{opt.mri_th}_{opt.snp_th}"
LOG_PATH = f"./logs/log_adniall_t1-gene-clip_MAECLS_{task}_{opt.use_sparse + 1}_{opt.use_sparse2 + 1}_{opt.mri_th}_{opt.snp_th}"

os.system("mkdir -p {}".format(MODEL_PATH))
os.system("mkdir -p {}".format(LOG_PATH))

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
lr = 1e-4
EPOCH = 200
WORKERS = TRAIN_BATCH_SIZE
WIDTH = 32
NUM = 10 * 13 * 11  # 1200 #
NUM2 = 4213  # 12621
SNP_NUM = 67406
EPOCH_PRE = 100
MRI_TOPK = opt.mri_th
SNP_TOPK = opt.snp_th
sparse_flag = 1
sparse_flag2 = 1
writer = SummaryWriter(logdir=LOG_PATH, comment='Gene2MRI')

for fold in range(5):
    # if fold ==  0:
    #     continue
    dataset_train_0 = MRIandGenedataset(fold=fold, phase="train", label=0)
    data_loader_train_0 = torch.utils.data.DataLoader(dataset_train_0, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                      num_workers=WORKERS)
    dataset_train_1 = MRIandGenedataset(fold=fold, phase="train", label=1)
    data_loader_train_1 = torch.utils.data.DataLoader(dataset_train_1, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                      num_workers=WORKERS)
    if len(dataset_train_0) >= len(dataset_train_1):
        data_loader_train_1 = itertools.cycle(data_loader_train_1)
        dataset_size = len(data_loader_train_0)
    else:
        data_loader_train_0 = itertools.cycle(data_loader_train_0)
        dataset_size = len(data_loader_train_1)

    dataset_train = MRIandGenedataset(fold=fold, phase="train", label=[0, 1])
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_test = MRIandGenedataset(fold=fold, phase="test", label=[0, 1])
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                   num_workers=WORKERS)
    dataset_size_test = len(data_loader_test)

    opt.hidden_dropout_prob = 0.2
    opt.emb_dropout_prob = 0.2
    opt2 = copy.deepcopy(opt)
    opt2.max_position_embeddings = 30000  #

    E2 = Mamba_dflow(opt2).cuda()
    G2 = GeneMLPEncoder().cuda()
    E = Mamba_dflow(opt).cuda()
    G = MRIMambaMAE(opt).cuda()
    model = CLIP().cuda()
    M1 = Maskcompute(opt).cuda()
    M2 = Maskcompute(opt2).cuda()
    C = Classifier().cuda()

    G = nn.DataParallel(G)
    G2 = nn.DataParallel(G2)
    E = nn.DataParallel(E)
    E2 = nn.DataParallel(E2)
    M1 = nn.DataParallel(M1)
    M2 = nn.DataParallel(M2)
    C = nn.DataParallel(C)

    E.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_E_{ep}.pth"))  # 57 24
    E2.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_E2_{ep}.pth"))  # 57 24
    G.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_G_{ep}.pth"))  # 57 24
    G2.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_G2_{ep}.pth"))  # 57 24
    model.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_clip_{ep}.pth"))  # 57 24
    M1.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_M1_{ep}.pth"))
    M2.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_M2_{ep}.pth"))

    for p in E.parameters():
        p.requires_grad = False
    for p in E2.parameters():
        p.requires_grad = False
    for p in G.parameters():
        p.requires_grad = False
    for p in G2.parameters():
        p.requires_grad = False
    for p in M1.parameters():
        p.requires_grad = False
    for p in M2.parameters():
        p.requires_grad = False
    for p in C.parameters():
        p.requires_grad = True
    for p in model.parameters():
        p.requires_grad = False
    w_sl = torch.ones([NUM, 1])
    w_sl = w_sl.cuda()
    w_sl.requires_grad = False
    w_sl2 = torch.ones([NUM2, 1])
    w_sl2 = w_sl2.cuda()
    w_sl2.requires_grad = False

    optim = torch.optim.AdamW([{'params': C.parameters()}],lr=lr)

    for ep in range(EPOCH):
        print("learning_rate:", optim.param_groups[0]['lr'])
        print(f'epoch {ep + 1}')
        total_ce_loss = 0

        E.eval()
        E2.eval()
        G.eval()
        G2.eval()
        M1.eval()
        M2.eval()
        model.eval()
        C.train()
        for train_data_0, train_data_1 in zip(data_loader_train_0, data_loader_train_1):
            fid_0, input_0, label_0, _, age_sex_0, integer_encoded_0 = train_data_0
            fid_1, input_1, label_1, _, age_sex_1, integer_encoded_1 = train_data_1
            label = torch.cat([label_0, label_1], dim=0)
            input = torch.cat([input_0, input_1], dim=0)
            age_sex = torch.cat([age_sex_0, age_sex_1], dim=0)
            integer_encoded = torch.cat([integer_encoded_0, integer_encoded_1], dim=0)

            label = label.cuda()
            B, L = integer_encoded.shape
            input = input.cuda()
            age_sex = age_sex.cuda()
            snp = integer_encoded.cuda()

            mask = torch.zeros((B, B)).cuda()
            for i in range(B):
                for j in range(B):
                    if label[i] == label[j]:
                        mask[i, j] = 1
            w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B, 1, 1)
            w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B, 1, 1)
            gate = F.gumbel_softmax(torch.log(torch.cat([w_sl_use, 1.0 - w_sl_use], dim=2) + 1e-10) * 10, tau=1,
                                    dim=2, hard=True)[:, :, 0]
            gate2 = F.gumbel_softmax(torch.log(torch.cat([w_sl_use2, 1.0 - w_sl_use2], dim=2) + 1e-10) * 10, tau=1,
                                     dim=2, hard=True)[:, :, 0]

            feature = G(x=input)
            feature = feature.view(B, opt.hidden_size, -1).permute([0, 2, 1])

            snp_onehot = F.one_hot(snp.clamp(0, 2), num_classes=3)
            snp_onehot[snp == 3] = 0
            feature2, _ = G2(snp_onehot.permute([0, 2, 1]).float())

            _, _, mri_feature, feature_emb = E(gate=gate, inputs_embeds=feature, batch=B, train=False,
                                               output_embedding=True)
            _, _, snp_feature, feature2_emb = E2(gate=gate2, inputs_embeds=feature2, batch=B, train=False,
                                                 output_embedding=True, age_sex=age_sex)
            _, _, mri_feature_s, snp_feature_s = model(mri_feature, snp_feature, mask)

            if sparse_flag:
                mask1 = M1(input_features=feature_emb.detach(), input_features2=mri_feature_s)
                w_sl_use_c = mask1.unsqueeze(2)
            else:
                w_sl_use_c = torch.ones([NUM, 1]).cuda().unsqueeze(0).repeat(B, 1, 1)
                mask1 = w_sl_use_c[:, :, 0]
            if sparse_flag2:
                mask2 = M2(input_features=snp_onehot, input_features2=snp_feature_s)
                w_sl_use2_c = mask2.unsqueeze(2)
            else:
                w_sl_use2_c = torch.ones([SNP_NUM, 1]).cuda().unsqueeze(0).repeat(B, 1, 1)
                mask2 = w_sl_use2_c[:, :, 0]

            if sparse_flag:
                gate_c = topk_keep(mask1, MRI_TOPK)
            else:
                gate_c = mask1
            if sparse_flag2:
                gate2_c = topk_keep(mask2, SNP_TOPK)
            else:
                gate2_c = mask2

            _, _, mri_feature_c = E(gate=gate * gate_c, inputs_embeds=feature, batch=B, train=False)
            snp_onehot_c = snp_onehot * gate2_c.unsqueeze(2)
            feature2_c, _ = G2(snp_onehot_c.permute([0, 2, 1]).float())
            _, _, snp_feature_c = E2(gate=gate2, inputs_embeds=feature2_c, batch=B, train=False, age_sex=age_sex)

            y = C(snp_feature_c)
            loss_ce = criterion(y, label)

            loss = loss_ce

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_ce_loss += loss_ce.item()

        print("ce loss:", total_ce_loss / dataset_size)
        writer.add_scalars('ce loss', {'ce loss' + str(fold): total_ce_loss / dataset_size, }, ep + 1)

        E.eval()
        E2.eval()
        G.eval()
        G2.eval()
        M1.eval()
        M2.eval()
        model.eval()
        C.eval()
        T_ = 0
        F_ = 0
        prob_all = []
        label_all = []
        predict_class_all = []
        with torch.no_grad():
            for train_data in data_loader_train:
                fid, input, label, _, age_sex, integer_encoded = train_data

                label = label.cuda()
                B, L = integer_encoded.shape
                input = input.cuda()
                age_sex = age_sex.cuda()
                snp = integer_encoded.cuda()

                mask = torch.zeros((B, B)).cuda()
                for i in range(B):
                    for j in range(B):
                        if label[i] == label[j]:
                            mask[i, j] = 1
                w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B, 1, 1)
                w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B, 1, 1)
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl_use, 1.0 - w_sl_use], dim=2) + 1e-10) * 10, tau=1,
                                        dim=2, hard=True)[:, :, 0]
                gate2 = F.gumbel_softmax(torch.log(torch.cat([w_sl_use2, 1.0 - w_sl_use2], dim=2) + 1e-10) * 10, tau=1,
                                         dim=2, hard=True)[:, :, 0]

                feature = G(x=input)
                feature = feature.view(B, opt.hidden_size, -1).permute([0, 2, 1])

                snp_onehot = F.one_hot(snp.clamp(0, 2), num_classes=3)
                snp_onehot[snp == 3] = 0
                feature2, _ = G2(snp_onehot.permute([0, 2, 1]).float())

                _, _, mri_feature, feature_emb = E(gate=gate, inputs_embeds=feature, batch=B, train=False,
                                                   output_embedding=True)
                _, _, snp_feature, feature2_emb = E2(gate=gate2, inputs_embeds=feature2, batch=B, train=False,
                                                     output_embedding=True, age_sex=age_sex)
                _, _, mri_feature_s, snp_feature_s = model(mri_feature, snp_feature, mask)

                if sparse_flag:
                    mask1 = M1(input_features=feature_emb.detach(), input_features2=mri_feature_s)
                    w_sl_use_c = mask1.unsqueeze(2)
                else:
                    w_sl_use_c = torch.ones([NUM, 1]).cuda().unsqueeze(0).repeat(B, 1, 1)
                    mask1 = w_sl_use_c[:, :, 0]
                if sparse_flag2:
                    mask2 = M2(input_features=snp_onehot, input_features2=snp_feature_s)
                    w_sl_use2_c = mask2.unsqueeze(2)
                else:
                    w_sl_use2_c = torch.ones([SNP_NUM, 1]).cuda().unsqueeze(0).repeat(B, 1, 1)
                    mask2 = w_sl_use2_c[:, :, 0]

                if sparse_flag:
                    gate_c = topk_keep(mask1, MRI_TOPK)
                else:
                    gate_c = mask1
                if sparse_flag2:
                    gate2_c = topk_keep(mask2, SNP_TOPK)
                else:
                    gate2_c = mask2

                _, _, mri_feature_c = E(gate=gate * gate_c, inputs_embeds=feature, batch=B, train=False)
                snp_onehot_c = snp_onehot * gate2_c.unsqueeze(2)
                feature2_c, _ = G2(snp_onehot_c.permute([0, 2, 1]).float())
                _, _, snp_feature_c = E2(gate=gate2, inputs_embeds=feature2_c, batch=B, train=False, age_sex=age_sex)

                y = C(snp_feature_c)

                out_c = F.softmax(y, dim=1)
                _, predicted = torch.max(out_c.data, 1)
                PREDICTED_ = predicted.data.cpu().numpy()
                REAL_ = label.data.cpu().numpy()
                # print(PREDICTED_)
                # print(REAL_)
                for k in range(PREDICTED_.shape[0]):
                    if PREDICTED_[k] == REAL_[k]:
                        T_ += 1
                    else:
                        F_ += 1
                prob_all.extend(out_c[:, 1].cpu().numpy().tolist())
                label_all.extend(label.cpu().numpy().tolist())
                predict_class_all.extend(PREDICTED_.tolist())

        train_acc = T_ / (T_ + F_)
        train_auc = roc_auc_score(label_all, prob_all)

        print("train_acc:", train_acc)
        print("train_auc:", train_auc)
        writer.add_scalars('train_acc', {'train_acc' + str(fold): train_acc, }, ep + 1)
        writer.add_scalars('train_auc', {'train_auc' + str(fold): train_auc, }, ep + 1)

        ###############test######################
        E.eval()
        E2.eval()
        G.eval()
        G2.eval()
        M1.eval()
        M2.eval()
        model.eval()
        C.eval()
        T_ = 0
        F_ = 0
        prob_all = []
        label_all = []
        predict_class_all = []
        with torch.no_grad():
            for test_data in data_loader_test:
                fid, input, label, _, age_sex, integer_encoded = test_data

                label = label.cuda()
                B, L = integer_encoded.shape
                input = input.cuda()
                age_sex = age_sex.cuda()
                snp = integer_encoded.cuda()

                mask = torch.zeros((B, B)).cuda()
                for i in range(B):
                    for j in range(B):
                        if label[i] == label[j]:
                            mask[i, j] = 1
                w_sl_use = torch.ones_like(w_sl).cuda().unsqueeze(0).repeat(B, 1, 1)
                w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B, 1, 1)
                gate = F.gumbel_softmax(torch.log(torch.cat([w_sl_use, 1.0 - w_sl_use], dim=2) + 1e-10) * 10, tau=1,
                                        dim=2, hard=True)[:, :, 0]
                gate2 = F.gumbel_softmax(torch.log(torch.cat([w_sl_use2, 1.0 - w_sl_use2], dim=2) + 1e-10) * 10, tau=1,
                                         dim=2, hard=True)[:, :, 0]

                feature = G(x=input)
                feature = feature.view(B, opt.hidden_size, -1).permute([0, 2, 1])

                snp_onehot = F.one_hot(snp.clamp(0, 2), num_classes=3)
                snp_onehot[snp == 3] = 0
                feature2, _ = G2(snp_onehot.permute([0, 2, 1]).float())

                _, _, mri_feature, feature_emb = E(gate=gate, inputs_embeds=feature, batch=B, train=False,
                                                   output_embedding=True)
                _, _, snp_feature, feature2_emb = E2(gate=gate2, inputs_embeds=feature2, batch=B, train=False,
                                                     output_embedding=True, age_sex=age_sex)
                _, _, mri_feature_s, snp_feature_s = model(mri_feature, snp_feature, mask)

                if sparse_flag:
                    mask1 = M1(input_features=feature_emb.detach(), input_features2=mri_feature_s)
                    w_sl_use_c = mask1.unsqueeze(2)
                else:
                    w_sl_use_c = torch.ones([NUM, 1]).cuda().unsqueeze(0).repeat(B, 1, 1)
                    mask1 = w_sl_use_c[:, :, 0]
                if sparse_flag2:
                    mask2 = M2(input_features=snp_onehot, input_features2=snp_feature_s)
                    w_sl_use2_c = mask2.unsqueeze(2)
                else:
                    w_sl_use2_c = torch.ones([SNP_NUM, 1]).cuda().unsqueeze(0).repeat(B, 1, 1)
                    mask2 = w_sl_use2_c[:, :, 0]

                if sparse_flag:
                    gate_c = topk_keep(mask1, MRI_TOPK)
                else:
                    gate_c = mask1
                if sparse_flag2:
                    gate2_c = topk_keep(mask2, SNP_TOPK)
                else:
                    gate2_c = mask2

                _, _, mri_feature_c = E(gate=gate * gate_c, inputs_embeds=feature, batch=B, train=False)
                snp_onehot_c = snp_onehot * gate2_c.unsqueeze(2)
                feature2_c, _ = G2(snp_onehot_c.permute([0, 2, 1]).float())
                _, _, snp_feature_c = E2(gate=gate2, inputs_embeds=feature2_c, batch=B, train=False, age_sex=age_sex)

                y = C(snp_feature_c)

                out_c = F.softmax(y, dim=1)
                _, predicted = torch.max(out_c.data, 1)
                PREDICTED_ = predicted.data.cpu().numpy()
                REAL_ = label.data.cpu().numpy()

                for k in range(PREDICTED_.shape[0]):
                    if PREDICTED_[k] == REAL_[k]:
                        T_ += 1
                    else:
                        F_ += 1
                prob_all.extend(out_c[:, 1].cpu().numpy().tolist())
                label_all.extend(label.cpu().numpy().tolist())
                predict_class_all.extend(PREDICTED_.tolist())

        test_acc = T_ / (T_ + F_)
        test_auc = roc_auc_score(label_all, prob_all)
        test_cm = confusion_matrix(label_all, predict_class_all)
        test_sen = test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1])
        test_spe = test_cm[1, 1] / (test_cm[1, 0] + test_cm[1, 1])
        print("test_acc:", test_acc)
        print("test_auc:", test_auc)
        print("test_sen:", test_sen)
        print("test_spe:", test_spe)

        writer.add_scalars('test_acc', {'test_acc' + str(fold): test_acc, }, ep + 1)
        writer.add_scalars('test_auc', {'test_auc' + str(fold): test_auc, }, ep + 1)
        writer.add_scalars('test_sen', {'test_sen' + str(fold): test_sen, }, ep + 1)
        writer.add_scalars('test_spe', {'test_spe' + str(fold): test_spe, }, ep + 1)

        if ((ep + 1) % 10) == 0:
            torch.save(C.state_dict(), MODEL_PATH + f"/fold_{fold}_C_{ep + 1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_C_{ep + 1}.pth")