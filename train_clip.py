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
from tensorboardX import SummaryWriter

cuda = torch.cuda.is_available()


def diagonal_topNum_probability(M=None, labels=None, k=5):
    """
    判断矩阵 M 中每行的对角线元素是否位于该行前 5 大元素之列，并返回其所占比例。

    参数：
    M: 形状为 (N, N) 的 numpy 数组

    返回：
    prob: 对角线元素位于各自行前 k 大值的概率
    """
    N = M.shape[0]
    count_topk = 0

    for i in range(N):
        row = M[i]
        index_k = np.argsort(row)[-k:]
        for index in index_k:
            if labels[index] == labels[i]:
                count_topk += 1
                # break
    prob = count_topk / N / k
    return prob

def topk_keep(x: torch.Tensor, k: int):
    topk_vals, _ = torch.topk(x, k, dim=-1, largest=True)
    thresh = topk_vals[..., -1, None]
    mask = (x >= thresh).to(x.dtype)  # (B, L) 中 0 / 1
    return mask + x - x.detach()  # (x * mask).detach() + x - x.detach()#

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

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

opt = TrainOptions().parse()
# initial for recurrence
seed = 8
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

task = 'adcls'
ep = 100
pretrain_dir = f"./generation_models/ADNI_T1-GENE-CLIP_MAECLS_Mask_0.8_0.8_{task}"
MODEL_PATH = f"./generation_models/ADNIALL_UKB_T1-GENE-CLIP_MAE_{task}_{opt.use_sparse + 1}_{opt.use_sparse2 + 1}_{opt.mri_th}_{opt.snp_th}"
LOG_PATH = f"./logs/log_adniall_ukb_t1-gene-clip_MAE_{task}_{opt.use_sparse + 1}_{opt.use_sparse2 + 1}_{opt.mri_th}_{opt.snp_th}"

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

writer = SummaryWriter(logdir=LOG_PATH, comment='Gene2MRI')

for fold in range(5):
    # if fold != 0:
    #     continue
    dataset_train_0 = MRIandGenedataset(fold=fold, phase="train", label=0)
    data_loader_train_0 = torch.utils.data.DataLoader(dataset_train_0, batch_size=TRAIN_BATCH_SIZE // 2, shuffle=True,
                                                      num_workers=WORKERS)
    dataset_train_1 = MRIandGenedataset(fold=fold, phase="train", label=1)
    data_loader_train_1 = torch.utils.data.DataLoader(dataset_train_1, batch_size=TRAIN_BATCH_SIZE // 2, shuffle=True,
                                                      num_workers=WORKERS)

    dataset_test_0 = MRIandGenedataset(fold=fold, phase="test", label=0)
    data_loader_test_0 = torch.utils.data.DataLoader(dataset_test_0, batch_size=TEST_BATCH_SIZE // 2, shuffle=False,
                                                     num_workers=WORKERS)
    dataset_test_1 = MRIandGenedataset(fold=fold, phase="test", label=1)
    data_loader_test_1 = torch.utils.data.DataLoader(dataset_test_1, batch_size=TEST_BATCH_SIZE // 2, shuffle=False,
                                                     num_workers=WORKERS)
    dataset_train = MRIandGenedataset(fold=fold, phase="train", label=[0, 1])
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                    num_workers=WORKERS)
    dataset_test = MRIandGenedataset(fold=fold, phase="test", label=[0, 1])
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                   num_workers=WORKERS)
    if len(dataset_train_0) >= len(dataset_train_1):
        data_loader_train_1 = itertools.cycle(data_loader_train_1)
        data_loader_test_1 = itertools.cycle(data_loader_test_1)
        dataset_size = len(data_loader_train_0)
        dataset_size_test = len(data_loader_test_0)
    else:
        data_loader_train_0 = itertools.cycle(data_loader_train_0)
        data_loader_test_0 = itertools.cycle(data_loader_test_0)
        dataset_size = len(data_loader_train_1)
        dataset_size_test = len(data_loader_test_1)
    print("dataset_size: ", dataset_size)
    print("data_loader_test: ", dataset_size_test)

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
    G = nn.DataParallel(G)
    G2 = nn.DataParallel(G2)
    E = nn.DataParallel(E)
    E2 = nn.DataParallel(E2)
    M1 = nn.DataParallel(M1)
    M2 = nn.DataParallel(M2)

    for p in G.parameters():
        p.requires_grad = False
    for p in G2.parameters():
        p.requires_grad = True
    for p in E.parameters():
        p.requires_grad = True
    for p in E2.parameters():
        p.requires_grad = True
    for p in M1.parameters():
        p.requires_grad = True
    for p in M2.parameters():
        p.requires_grad = True
    for p in model.parameters():
        p.requires_grad = True
    w_sl = torch.ones([NUM, 1])
    w_sl = w_sl.cuda()
    w_sl.requires_grad = False
    w_sl2 = torch.ones([NUM2, 1])
    w_sl2 = w_sl2.cuda()
    w_sl2.requires_grad = False

    E.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_E_{ep}.pth"))  # 57 24
    E2.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_E2_{ep}.pth"))  # 57 24
    G.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_G_{ep}.pth"))  # 57 24
    G2.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_G2_{ep}.pth"))  # 57 24
    model.load_state_dict(torch.load(pretrain_dir + f"/fold_{fold}_clip_{ep}.pth"))  # 57 24

    optim = torch.optim.AdamW([{'params': model.parameters()}], lr=lr)
    optim_E = torch.optim.AdamW([{'params': G2.parameters()}, {'params': E.parameters()}, {'params': E2.parameters()}], lr=lr)
    optim_M = torch.optim.AdamW([{'params': M1.parameters()}, {'params': M2.parameters()}], lr=lr)

    sparse_flag = 0
    sparse_flag2 = 0
    for ep in range(EPOCH):
        start_time = time.time()
        print(f'epoch {ep + 1}')
        total_clip_loss = 0
        total_clip_loss2 = 0
        total_clip_loss3 = 0
        total_clip_loss4 = 0
        total_clip_loss5 = 0
        E.train()
        E2.train()
        G.eval()
        G2.train()
        M1.train()
        M2.train()
        model.train()
        for train_data_0, train_data_1 in zip(data_loader_train_0, data_loader_train_1):
            fid_0, input_0, label_0, _, age_sex_0, integer_encoded_0 = train_data_0
            fid_1, input_1, label_1, _, age_sex_1, integer_encoded_1 = train_data_1
            # 80 100 83
            label = torch.cat([label_0, label_1], dim=0)
            input = torch.cat([input_0, input_1], dim=0)
            age_sex = torch.cat([age_sex_0, age_sex_1], dim=0)
            integer_encoded = torch.cat([integer_encoded_0, integer_encoded_1], dim=0)
            B, L = integer_encoded.shape
            label = label.cuda()
            input = input.cuda()
            age_sex = age_sex.cuda()
            snp = integer_encoded.cuda().long()

            mask = torch.zeros((B, B)).cuda()
            for i in range(B):
                for j in range(B):
                    if label[i] == label[j]:
                        mask[i, j] = 1

            w_sl_use = torch.ones_like(w_sl).cuda() * torch.bernoulli(torch.full_like(w_sl, opt.mask_dropout)).unsqueeze(0).repeat(B, 1, 1)
            w_sl_use2 = torch.ones_like(w_sl2).cuda().unsqueeze(0).repeat(B, 1, 1)
            gate = F.gumbel_softmax(torch.log(torch.cat([w_sl_use, 1.0 - w_sl_use], dim=2) + 1e-10) * 10, tau=1, dim=2, hard=False)[:, :, 0]
            gate2 = F.gumbel_softmax(torch.log(torch.cat([w_sl_use2, 1.0 - w_sl_use2], dim=2) + 1e-10) * 10, tau=1, dim=2, hard=False)[:, :, 0]

            feature =  G(x=input)
            feature =  feature.view(B,opt.hidden_size,-1).permute([0,2,1])

            snp_onehot = F.one_hot(snp.clamp(0, 2), num_classes=3)
            snp_onehot[snp == 3] = 0
            feature2, mask_snp = G2(snp_onehot.permute([0, 2, 1]).float(), M_Ratio=0.2)  # ,M_Ratio=0.2

            _, _, mri_feature, feature_emb = E(gate=gate, inputs_embeds=feature, batch=B, output_embedding=True)
            _, _, snp_feature, feature2_emb = E2(gate=gate2, inputs_embeds=feature2, batch=B, output_embedding=True, age_sex=age_sex)
            loss_clip, _, mri_feature_s, snp_feature_s = model(mri_feature, snp_feature, mask)

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
                gate_c_hard = topk_keep(F.softmax(mask1, dim=-1) * gate, MRI_TOPK)
                gate_c_hard = mask1 * gate
            if sparse_flag2:
                gate2_c_hard = topk_keep(F.softmax(mask2, dim=-1) * (1.0 - mask_snp), SNP_TOPK)
            else:
                gate2_c_hard = mask2 * (1.0 - mask_snp)

            _, _, mri_feature_c = E(gate=gate * gate_c_hard, inputs_embeds=feature, batch=B)
            snp_onehot_c = snp_onehot * gate2_c_hard.unsqueeze(2)
            feature2_c, _ = G2(snp_onehot_c.permute([0, 2, 1]).float(), mask=mask_snp)
            _, _, snp_feature_c = E2(gate=gate2, inputs_embeds=feature2_c, batch=B, age_sex=age_sex)

            loss_clip2, _, _, _ = model(mri_feature_c, snp_feature, mask)
            loss_clip3, _, _, _ = model(mri_feature, snp_feature_c, mask)
            feature_bl = gate_c_hard.unsqueeze(2) * feature + (1.0 - gate_c_hard.unsqueeze(2)) * torch.flip(feature, dims=[0])
            _, _, mri_feature_bl = E(gate=gate, inputs_embeds=feature_bl, batch=B)
            snp_onehot_bl = gate2_c_hard.unsqueeze(2) * snp_onehot + (1.0 - gate2_c_hard.unsqueeze(2)) * torch.flip(snp_onehot, dims=[0])
            feature2_bl, _ = G2(snp_onehot_bl.permute([0, 2, 1]).float(), mask=mask_snp)
            _, _, snp_feature_bl = E2(gate=gate2, inputs_embeds=feature2_bl, batch=B, age_sex=age_sex)
            loss_clip4, _, _, _ = model(mri_feature_bl, snp_feature, mask)
            loss_clip5, _, _, _ = model(mri_feature, snp_feature_bl, mask)

            loss = loss_clip + 0.1 * (loss_clip2 + loss_clip3) + 0.1 * (loss_clip4 + loss_clip5)

            optim.zero_grad()
            optim_E.zero_grad()
            optim_M.zero_grad()
            loss.backward()
            optim.step()
            optim_E.step()
            optim_M.step()
            total_clip_loss += loss_clip.item()
            total_clip_loss2 += loss_clip2.item()
            total_clip_loss3 += loss_clip3.item()
            total_clip_loss4 += loss_clip4.item()
            total_clip_loss5 += loss_clip5.item()

        print("clip_loss:", total_clip_loss / dataset_size)
        print("clip_loss2:", total_clip_loss2 / dataset_size)
        print("clip_loss3:", total_clip_loss3 / dataset_size)
        print("clip_loss4:", total_clip_loss4 / dataset_size)
        print("clip_loss5:", total_clip_loss5 / dataset_size)
        writer.add_scalars('clip_loss', {'clip_loss' + str(fold): total_clip_loss / dataset_size, }, ep + 1)
        writer.add_scalars('clip_loss2', {'clip_loss2' + str(fold): total_clip_loss2 / dataset_size, }, ep + 1)
        writer.add_scalars('clip_loss3', {'clip_loss3' + str(fold): total_clip_loss3 / dataset_size, }, ep + 1)
        writer.add_scalars('clip_loss4', {'clip_loss4' + str(fold): total_clip_loss4 / dataset_size, }, ep + 1)
        writer.add_scalars('clip_loss5', {'clip_loss5' + str(fold): total_clip_loss5 / dataset_size, }, ep + 1)

        E.eval()
        E2.eval()
        G.eval()
        G2.eval()
        M1.eval()
        M2.eval()
        model.eval()
        with torch.no_grad():
            total_train_clip_score = 0
            total_train_clip_loss = 0
            mri_feature_list = []
            snp_feature_list = []
            age_sex_list = []
            label_list = []
            total_mask1 = torch.zeros([NUM]).cuda()
            total_mask2 = torch.zeros([SNP_NUM]).cuda()
            for train_data_0, train_data_1 in zip(data_loader_train_0, data_loader_train_1):
                fid_0, input_0, label_0, _, age_sex_0, integer_encoded_0 = train_data_0
                fid_1, input_1, label_1, _, age_sex_1, integer_encoded_1 = train_data_1
                # 80 100 83
                label = torch.cat([label_0, label_1], dim=0)
                input = torch.cat([input_0, input_1], dim=0)
                age_sex = torch.cat([age_sex_0, age_sex_1], dim=0)
                integer_encoded = torch.cat([integer_encoded_0, integer_encoded_1], dim=0)
                B, L = integer_encoded.shape
                label = label.cuda()
                feature = feature.cuda()
                age_sex = age_sex.cuda()
                snp = integer_encoded.cuda().long()

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
                loss_clip_train, _, mri_feature_c, snp_feature_c = model(mri_feature_c, snp_feature_c, mask)

                mri_feature_list.append(mri_feature_c)
                snp_feature_list.append(snp_feature_c)
                age_sex_list.append(age_sex)
                label_list.append(label)
                total_mask1 += torch.mean(mask1, dim=0)
                total_mask2 += torch.mean(mask2, dim=0)
                total_train_clip_loss += loss_clip_train.item()

            print("indices:", torch.sort(total_mask1 / dataset_size, descending=True)[1].cpu().numpy().tolist()[0:20])
            print("values:", [round(value, 2) for value in
                              torch.sort(total_mask1 / dataset_size, descending=True)[0].cpu().numpy().tolist()[0:20]])
            print("indices2:", torch.sort(total_mask2 / dataset_size, descending=True)[1].cpu().numpy().tolist()[0:20])
            print("values2:", [round(value, 2) for value in
                               torch.sort(total_mask2 / dataset_size, descending=True)[0].cpu().numpy().tolist()[0:20]])

            mri_features = torch.cat(mri_feature_list, dim=0)
            snp_features = torch.cat(snp_feature_list, dim=0)
            age_sexs = torch.cat(age_sex_list, dim=0)
            labels = torch.cat(label_list, dim=0)
            sub_num = mri_features.shape[0]

            ages = age_sexs[:, 0].unsqueeze(1)
            sexs = age_sexs[:, 1].unsqueeze(1)
            # mask = ((sexs == sexs.T) & (torch.abs(ages - ages.T) <= 0)).float()

            mri_features = mri_features / mri_features.norm(dim=1, keepdim=True)
            snp_features = snp_features / snp_features.norm(dim=1, keepdim=True)

            clip_scores = (snp_features @ mri_features.t()).detach().cpu().numpy()
            top10_SR = diagonal_topNum_probability(M=clip_scores, labels=labels, k=10)
            top5_SR = diagonal_topNum_probability(M=clip_scores, labels=labels, k=5)
            top1_SR = diagonal_topNum_probability(M=clip_scores, labels=labels, k=1)

            clip_scores2 = (mri_features @ snp_features.t()).detach().cpu().numpy()
            top10_MR = diagonal_topNum_probability(M=clip_scores2, labels=labels, k=10)
            top5_MR = diagonal_topNum_probability(M=clip_scores2, labels=labels, k=5)
            top1_MR = diagonal_topNum_probability(M=clip_scores2, labels=labels, k=1)

            print("train_clip_loss:", total_train_clip_loss / dataset_size)
            writer.add_scalars('train_clip_loss',
                               {'train_clip_loss' + str(fold): total_train_clip_loss / dataset_size, }, ep + 1)
            print("train_top10_SR:", top10_SR)
            writer.add_scalars('train_top10_SR', {'train_top10_SR_' + str(fold): top10_SR, }, ep + 1)
            print("train_top5_SR:", top5_SR)
            writer.add_scalars('train_top5_SR', {'train_top5_SR_' + str(fold): top5_SR, }, ep + 1)
            print("train_top1_SR:", top1_SR)
            writer.add_scalars('train_top1_SR', {'train_top1_SR_' + str(fold): top1_SR, }, ep + 1)
            print("train_top10_MR:", top10_MR)
            writer.add_scalars('train_top10_MR', {'train_top10_MR_' + str(fold): top10_MR, }, ep + 1)
            print("train_top5_MR:", top5_MR)
            writer.add_scalars('train_top5_MR', {'train_top5_MR_' + str(fold): top5_MR, }, ep + 1)
            print("train_top1_MR:", top1_MR)
            writer.add_scalars('train_top1_MR', {'train_top1_MR_' + str(fold): top1_MR, }, ep + 1)

        np.save(MODEL_PATH + f"/fold_{fold}_mask1_{ep + 1}.npy", (total_mask1 / dataset_size).cpu().numpy())
        np.save(MODEL_PATH + f"/fold_{fold}_mask2_{ep + 1}.npy", (total_mask2 / dataset_size).cpu().numpy())

        ###########test phase############################
        E.eval()
        E2.eval()
        G.eval()
        G2.eval()
        M1.eval()
        M2.eval()
        model.eval()
        with torch.no_grad():
            total_test_clip_score = 0
            total_test_clip_loss = 0
            mri_feature_list = []
            snp_feature_list = []
            age_sex_list = []
            label_list = []
            for test_data_0, test_data_1 in zip(data_loader_test_0, data_loader_test_1):
                fid_0, input_0, label_0, _, age_sex_0, integer_encoded_0 = test_data_0
                fid_1, input_1, label_1, _, age_sex_1, integer_encoded_1 = test_data_1
                # 80 100 83
                label = torch.cat([label_0, label_1], dim=0)
                input = torch.cat([input_0, input_1], dim=0)
                age_sex = torch.cat([age_sex_0, age_sex_1], dim=0)
                integer_encoded = torch.cat([integer_encoded_0, integer_encoded_1], dim=0)
                B, L = integer_encoded.shape
                label = label.cuda()
                feature = feature.cuda()
                age_sex = age_sex.cuda()
                snp = integer_encoded.cuda().long()

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
                loss_clip_test, _, mri_feature_c, snp_feature_c = model(mri_feature_c, snp_feature_c, mask)

                mri_feature_list.append(mri_feature_c)
                snp_feature_list.append(snp_feature_c)
                age_sex_list.append(age_sex)
                label_list.append(label)
                total_test_clip_loss += loss_clip_test.item()

            mri_features = torch.cat(mri_feature_list, dim=0)
            snp_features = torch.cat(snp_feature_list, dim=0)
            age_sexs = torch.cat(age_sex_list, dim=0)
            labels = torch.cat(label_list, dim=0)
            sub_num = mri_features.shape[0]

            ages = age_sexs[:, 0].unsqueeze(1)
            sexs = age_sexs[:, 1].unsqueeze(1)

            mri_features = mri_features / mri_features.norm(dim=1, keepdim=True)
            snp_features = snp_features / snp_features.norm(dim=1, keepdim=True)

            clip_scores = (snp_features @ mri_features.t()).detach().cpu().numpy()
            top10_SR = diagonal_topNum_probability(M=clip_scores, labels=labels, k=10)
            top5_SR = diagonal_topNum_probability(M=clip_scores, labels=labels, k=5)
            top1_SR = diagonal_topNum_probability(M=clip_scores, labels=labels, k=1)  # !/usr/bin/env python

            clip_scores2 = (mri_features @ snp_features.t()).detach().cpu().numpy()
            top10_MR = diagonal_topNum_probability(M=clip_scores2, labels=labels, k=10)
            top5_MR = diagonal_topNum_probability(M=clip_scores2, labels=labels, k=5)
            top1_MR = diagonal_topNum_probability(M=clip_scores2, labels=labels, k=1)

            print("test_clip_loss:", total_test_clip_loss / dataset_size_test)
            writer.add_scalars('test_clip_loss',
                               {'test_clip_loss' + str(fold): total_test_clip_loss / dataset_size_test, }, ep + 1)
            print("test_top10_SR:", top10_SR)
            writer.add_scalars('test_top10_SR', {'test_top10_SR_' + str(fold): top10_SR, }, ep + 1)
            print("test_top5_SR:", top5_SR)
            writer.add_scalars('test_top5_SR', {'test_top5_SR_' + str(fold): top5_SR, }, ep + 1)
            print("test_top1_SR:", top1_SR)
            writer.add_scalars('test_top1_SR', {'test_top1_SR_' + str(fold): top1_SR, }, ep + 1)
            print("test_top10_MR:", top10_MR)
            writer.add_scalars('test_top10_MR', {'test_top10_MR_' + str(fold): top10_MR, }, ep + 1)
            print("test_top5_MR:", top5_MR)
            writer.add_scalars('test_top5_MR', {'test_top5_MR_' + str(fold): top5_MR, }, ep + 1)
            print("test_top1_MR:", top1_MR)
            writer.add_scalars('test_top1_MR', {'test_top1_MR_' + str(fold): top1_MR, }, ep + 1)

        end_time = time.time()
        print(f"time: {end_time - start_time}s")

        if (ep + 1) >= EPOCH_PRE and ((ep + 1) % 5) == 0:
            torch.save(model.state_dict(), MODEL_PATH + f"/fold_{fold}_clip_{ep + 1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_clip_{ep + 1}.pth")
            torch.save(E.state_dict(), MODEL_PATH + f"/fold_{fold}_E_{ep + 1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_E_{ep + 1}.pth")
            torch.save(E2.state_dict(), MODEL_PATH + f"/fold_{fold}_E2_{ep + 1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_E2_{ep + 1}.pth")
            torch.save(G.state_dict(), MODEL_PATH + f"/fold_{fold}_G_{ep + 1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_G_{ep + 1}.pth")
            torch.save(G2.state_dict(), MODEL_PATH + f"/fold_{fold}_G2_{ep + 1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_G2_{ep + 1}.pth")
            torch.save(M1.state_dict(), MODEL_PATH + f"/fold_{fold}_M1_{ep + 1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_M1_{ep + 1}.pth")
            torch.save(M2.state_dict(), MODEL_PATH + f"/fold_{fold}_M2_{ep + 1}.pth")
            print('saved model at ' + MODEL_PATH + f"/fold_{fold}_M2_{ep + 1}.pth")

        if opt.use_sparse and (ep + 1) >= EPOCH_PRE:
            sparse_flag = 1
            for p in M1.parameters():
                p.requires_grad = True
        if opt.use_sparse2 and (ep + 1) >= EPOCH_PRE:
            sparse_flag2 = 1
            for p in M2.parameters():
                p.requires_grad = True
