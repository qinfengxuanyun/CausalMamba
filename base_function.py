from torch import nn
from torch.nn import init
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import copy

def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False

def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def weights_init(m):
    """根据模块类型选择合适的初始化"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # ReLU 系列激活
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.GRU, nn.LSTM)):
        # RNN：输入->隐藏 用 Kaiming，隐藏->隐藏 用 Orthogonal
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # LSTM 可以把 forget 门偏置 +1
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=15, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_metric):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=0.25, gamma=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds,
                                  dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1,
                                                            1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode in ['wgangp', 'wgandiv']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real, is_disc=False):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            labels = (self.real_label if target_is_real else self.fake_label).expand_as(prediction).type_as(prediction)
            loss = self.loss(prediction, labels)
        elif self.gan_mode in ['hinge', 'wgangp', 'wgandiv']:
            if is_disc:
                if target_is_real:
                    prediction = -prediction
                if self.gan_mode == 'hinge':
                    loss = self.loss(1 + prediction).mean()
                elif self.gan_mode in ['wgangp', 'wgandiv']:
                    loss = prediction.mean()
            else:
                loss = -prediction.mean()

        return loss

class Maskcompute(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(3, config.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                 nn.ELU(),
                                 nn.Linear(config.hidden_size, config.hidden_size))

    def apply_rope(self, x):
        bsz, seqlen, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

        # 生成频率张量
        half_dim = head_dim // 2
        theta = 12000 ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(x.device)  # 12000
        seq_idx = torch.arange(seqlen, dtype=torch.float32).to(x.device)
        freqs = torch.einsum("i,j->ij", seq_idx, theta)  # [seq_len, half_dim]

        # 转换为 cos/sin 编码
        sin = freqs.sin()[None, :, :]  # shape: [1, seq_len, half_dim]
        cos = freqs.cos()[None, :, :]

        # 拆分并旋转
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return x_rotated

    def forward(self, input_features=None, input_features2=None):
        B, L, C = input_features.shape
        if C == 3:
            features = self.fc(input_features.float())
            features = self.apply_rope(features)
        else:
            features = input_features
        features = self.mlp(features)

        features2 = input_features2.unsqueeze(1)
        features2 = features2.expand_as(features)
        mask = F.cosine_similarity(features, features2, dim=-1)


        return mask

class Maskcompute2(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(3, config.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                 nn.ELU(),
                                 nn.Linear(config.hidden_size, config.hidden_size))

    def apply_rope(self, x):
        bsz, seqlen, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

        # 生成频率张量
        half_dim = head_dim // 2
        theta = 12000 ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(x.device)  # 12000
        seq_idx = torch.arange(seqlen, dtype=torch.float32).to(x.device)
        freqs = torch.einsum("i,j->ij", seq_idx, theta)  # [seq_len, half_dim]

        # 转换为 cos/sin 编码
        sin = freqs.sin()[None, :, :]  # shape: [1, seq_len, half_dim]
        cos = freqs.cos()[None, :, :]

        # 拆分并旋转
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return x_rotated

    def forward(self, input_features=None):
        B, L, C = input_features.shape
        if C == 3:
            features = self.fc(input_features.float())
            features = self.apply_rope(features)
        else:
            features = input_features
        mask = self.mlp(features).squeeze(-1)

        return mask

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings2 = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.emb_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None and inputs_embeds is not None:
            input_shape = (input_ids.size()[0], input_ids.size()[1] + inputs_embeds.size()[1])
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = torch.cat([self.word_embeddings(input_ids), inputs_embeds], dim=1)
        elif input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size ** 0.5)

        embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = position_embeddings * embeddings + position_embeddings2

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings, inputs_embeds

class GeneMLPEncoder(nn.Module):
    def __init__(self, config, input_channels=16 * 3, latent_feature=32):
        super(GeneMLPEncoder, self).__init__()
        self.config = copy.deepcopy(config)
        self.config.hidden_size = input_channels
        self.embeddings = Embeddings(self.config)
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, latent_feature),
            nn.ReLU(),
            nn.Linear(latent_feature, 512),
        )

    def patchify(self, x, p=16):
        pad_len = math.ceil(x.shape[2] / p) * p - x.shape[2]
        x = F.pad(x, (0, pad_len), 'constant', 0)
        l = x.shape[2] // p
        x = x.reshape(shape=(x.shape[0], 3, l, p))
        x = torch.einsum('nclp->nlpc', x)
        x = x.reshape(shape=(x.shape[0], l, p * 3))
        return x

    def forward(self, snp=None, M_Ratio=0, mask=None, p=16, use_embedding=True, gate=None, snp2=None):
        # 编码
        snp = snp.long()
        if snp2 is not None:
            snp2 = snp2.long()
            snp = gate.long() * snp + (1 - gate.long()) * snp2  # torch.flip(snp,dims=[0])
        x = F.one_hot(snp.clamp(0, 2), num_classes=3)
        x[snp == 3] = 0
        if snp2 is None and (gate is not None):
            x = x * gate.unsqueeze(2)
        x = x.permute([0, 2, 1]).float()

        if mask != None:
            x = x * (1.0 - mask.unsqueeze(1))
        elif M_Ratio != 0:
            x, mask = self.random_masking(x, M_Ratio)
        x = self.patchify(x, p=p)

        if use_embedding:
            x, _ = self.embeddings(inputs_embeds=x)
        x = self.encoder(x)
        return x, mask

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, D, L], sequence
        """
        N, D, L = x.shape  # batch, dim, length
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(1, L, device=x.device).repeat(N, 1)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x * (1.0 - mask.unsqueeze(1))

        return x_masked, mask

def patchify(imgs, p=8):
    """
    imgs: (N, 1, H, W,L)
    x: (N, L, patch_size**3 *1)
    """
    w, h, l = imgs.shape[2] // p, imgs.shape[3] // p, imgs.shape[4] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, w, p, h, p, l, p))
    x = torch.einsum('ncwohplq->nwhlopqc', x)
    x = x.reshape(shape=(imgs.shape[0], w * h * l, p ** 3 * 1))
    return x

def unpatchify(x, feature_shape, p=8):
    """
    x: (N, L, patch_size**3*1)
    imgs: (N, 1, H, W,L)
    """
    _, _, w, h, l = feature_shape

    x = x.reshape(shape=(x.shape[0], w, h, l, p, p, p, 1))
    x = torch.einsum('nwhlopqc->ncwohplq', x)
    imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
    return imgs

