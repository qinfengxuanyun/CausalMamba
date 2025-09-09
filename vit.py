import torch
import torch.nn as nn
import math
from torch.nn import functional as F

from mamba_model import BiMamba_block

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, max_position_embeddings=1201, hidden_size=512):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.position_embeddings2 = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        #self.dropout_embedding = nn.Dropout2d(0.2)

    def forward(self,inputs_embeds=None,past_key_values_length=0 ):
        seq_length = inputs_embeds.shape[1]
        position_ids = self.position_ids[:, past_key_values_length:past_key_values_length+seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings2 = self.position_embeddings2(position_ids)
        embeddings = inputs_embeds
        embeddings = position_embeddings * embeddings + position_embeddings2
        embeddings = self.LayerNorm(embeddings)
        return embeddings

class SE_block(nn.Module):
    def __init__(self, inchannels, reduction = 16 ):
        super(SE_block,self).__init__()
        self.GAP = nn.AdaptiveAvgPool3d((1,1,1))
        self.FC1 = nn.Linear(inchannels,inchannels//reduction)
        self.FC2 = nn.Linear(inchannels//reduction,inchannels)

    def forward(self,x):
        model_input = x
        x = self.GAP(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.FC1(x)
        x = nn.ReLU()(x)
        x = self.FC2(x)
        x = nn.Sigmoid()(x)
        x = x.view(x.size(0),x.size(1),1,1,1)
        return model_input * x

class AC_layer(nn.Module):
    def __init__(self,inchannels, outchannels):
        super(AC_layer,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.InstanceNorm3d(outchannels))
        self.conv2 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,3),stride=1,padding=(0,0,1),bias=False),
            nn.InstanceNorm3d(outchannels))
        self.conv3 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,1,1),stride=1,padding=(1,0,0),bias=False),
            nn.InstanceNorm3d(outchannels))
        self.conv4 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,3,1),stride=1,padding=(0,1,0),bias=False),
            nn.InstanceNorm3d(outchannels))
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4

class dense_layer(nn.Module):
    def __init__(self,inchannels,outchannels):

        super(dense_layer,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            AC_layer(inchannels,outchannels),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.Conv3d(outchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            AC_layer(outchannels,outchannels),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            SE_block(outchannels),
            nn.MaxPool3d(2,2),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,1),stride=1,padding=0,bias=False),
            nn.InstanceNorm3d(outchannels),
            nn.ELU(),
            nn.MaxPool3d(2,2),
        )
        #self.drop = nn.Dropout3d(0.1)

    def forward(self,x):
        #x = self.drop(x)
        new_features = self.block(x)
        x = F.max_pool3d(x,2)
        x = torch.cat([new_features,x], 1)
        #x = self.block(new_features) + self.block2(x)
        return x

class DenseNet3DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(DenseNet3DEncoder, self).__init__()
        self.block, last_channels = self._make_block(base_channels, 3)
        self.out = nn.Sequential(
            nn.Conv3d(last_channels, 512, (1, 1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm3d(512),
            nn.ELU(),
        )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = 1
        for i in range(nb_block):
            outchannels = nb_filter * pow(2, i + 1)
            blocks.append(dense_layer(inchannels, outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x):
        out = self.block(x)
        out = self.out(out)
        return out

class MRIMambaMAE(nn.Module):
    def __init__(self, config, in_channels=1, img_size=(80, 100, 83), patch_size=(8, 8, 8), emb_size=512, dropout=0.1,
                 nb_filter=32):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_num = config.max_position_embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.encoder = DenseNet3DEncoder()  # ResNet3DEncoder()#

        self.decoder_pred = nn.Linear(emb_size, patch_size[0] * patch_size[1] * patch_size[2] * in_channels, bias=True)

        self.up1 = nn.Sequential(
            nn.Conv3d(512, nb_filter * 4, (3, 3, 3), stride=1, padding=1, bias=False),
            # nn.ConvTranspose3d(512, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(nb_filter * 4),
            nn.ELU(),
            nn.Upsample((20, 25, 20)),
            nn.Conv3d(nb_filter * 4, nb_filter * 4, (3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(nb_filter * 4),
            nn.ELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(nb_filter * 4, nb_filter * 2, (3, 3, 3), stride=1, padding=1, bias=False),
            # nn.ConvTranspose3d(nb_filter*2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(nb_filter * 2),
            nn.ELU(),
            nn.Upsample((40, 50, 41)),
            nn.Conv3d(nb_filter * 2, nb_filter * 2, (3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(nb_filter * 2),
            nn.ELU(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(nb_filter * 2, nb_filter, (3, 3, 3), stride=1, padding=1, bias=False),
            # nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
            nn.Upsample((80, 100, 83)),
            nn.Conv3d(nb_filter, nb_filter, (3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(nb_filter),
            nn.ELU(),
        )

        self.out1 = nn.Sequential(nn.ReflectionPad3d(int(3 / 2)),
                                  nn.Conv3d(nb_filter * 4, 1, (3, 3, 3), stride=1, padding=0, bias=False))
        self.out2 = nn.Sequential(nn.ReflectionPad3d(int(3 / 2)),
                                  nn.Conv3d(nb_filter * 2, 1, (3, 3, 3), stride=1, padding=0, bias=False))
        self.out3 = nn.Sequential(nn.ReflectionPad3d(int(3 / 2)),
                                  nn.Conv3d(nb_filter, 1, (3, 3, 3), stride=1, padding=0, bias=False))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.mamba = nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
        ])
        self.mamba2 = nn.ModuleList([
            BiMamba_block(config) for i in range(config.num_hidden_layers)
        ])
        self.pos_embed = Embeddings(self.patch_num + 1, emb_size)
        self.pos_embed2 = Embeddings(self.patch_num + 1, emb_size)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking2(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(1, L, device=x.device).repeat(N, 1)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = x * (1.0 - mask.unsqueeze(2))

        return x_masked, mask

    def patchify(self, imgs, p=8):
        """
        imgs: (N, 1, H, W,L)
        x: (N, L, patch_size**3 *1)
        """
        imgs = F.pad(imgs, (3, 2, 2, 2, 0, 0), 'replicate')
        w, h, l = imgs.shape[2] // p, imgs.shape[3] // p, imgs.shape[4] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, w, p, h, p, l, p))
        x = torch.einsum('ncwohplq->nwhlopqc', x)
        x = x.reshape(shape=(imgs.shape[0], w, h, l, p ** 3 * 1))
        return x

    def unpatchify(self, x, feature_shape):
        """
        x: (N, L, patch_size**3*1)
        imgs: (N, 1, H, W,L)
        """
        p = self.patch_size[0]
        w, h, l = feature_shape

        x = x.reshape(shape=(x.shape[0], w, h, l, p, p, p, 1))
        x = torch.einsum('nwhlopqc->ncwohplq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, w * p, h * p, l * p))
        imgs = imgs[:, :, :, 2:-2, 3:-2]
        return imgs

    def forward(self, x=None, out_rec=False, M_Ratio=0.75):
        if out_rec:
            latent, mask, ids_restore = self.forward_encoder(x, M_Ratio)
            x_ = self.forward_decoder(latent, ids_restore)
            # x_ = self.unpatchify(x_,(B,C,L))[:,:,0:-pad_len]
            return latent, mask, x_
        else:
            feature = self.forward_(x)
            return feature

    def forward_encoder(self, x=None, M_Ratio=0.75):
        # Patch embedding
        # (80,100,83) (40,50,41)  (20,25,20) (10,12,10)
        feature = self.patchify(x)
        B, W, H, L, C = feature.shape
        feature = feature.view(B, -1, C)
        feature = self.encoder(
            feature.view(B * feature.shape[1], 1, self.patch_size[0], self.patch_size[1], self.patch_size[2])).view(B,
                                                                                                                    feature.shape[
                                                                                                                        1],
                                                                                                                    -1)
        feature_m, mask, ids_restore = self.random_masking(self.pos_embed(feature, 1), M_Ratio)
        # append cls token
        cls_tokens = self.cls_token.expand(feature_m.shape[0], -1, -1)
        feature_m = torch.cat((cls_tokens, feature_m), dim=1)

        for encoder in self.mamba:
            feature_m = encoder(feature_m)
        latent = feature_m  # .permute([0,2,1])

        return latent, mask, ids_restore

    def forward_decoder(self, latent=None, ids_restore=None):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(latent.shape[0], ids_restore.shape[1] + 1 - latent.shape[1], 1)
        feature_ = torch.cat([latent[:, 1:, :], mask_tokens], dim=1)  # no cls token
        feature_ = torch.gather(feature_, dim=1,
                                index=ids_restore.unsqueeze(-1).repeat(1, 1, feature_.shape[2]))  # unshuffle
        feature_ = torch.cat([latent[:, :1, :], feature_], dim=1)  # append cls token

        # add pos embed
        feature_ = self.pos_embed2(feature_, 0)
        for encoder in self.mamba2:
            feature_ = encoder(feature_)
        feature_ = feature_[:, 1:, :]
        w, h, l = math.ceil(1.0 * self.img_size[0] / self.patch_size[0]), math.ceil(
            1.0 * self.img_size[1] / self.patch_size[1]), math.ceil(1.0 * self.img_size[2] / self.patch_size[2])
        # feature_ = feature_.permute([0,2,1]).reshape(feature_.shape[0],feature_.shape[2],w,h,l)
        # x_1 =  self.up1(feature_)
        # x_2 =  self.up2(x_1)
        # x_3 =  self.up3(x_2)
        # x_ = self.out3(x_3)
        x_ = self.decoder_pred(feature_)
        x_ = self.unpatchify(x_, (w, h, l))
        return x_

    def forward_(self, x=None, M_Ratio=0):
        # Patch embedding
        # (80,100,83) (40,50,41)  (20,25,20) (10,12,10)
        feature = self.patchify(x)
        B, W, H, L, C = feature.shape
        feature = feature.view(B, -1, C)
        feature = self.encoder(
            feature.view(B * feature.shape[1], 1, self.patch_size[0], self.patch_size[1], self.patch_size[2])).view(B,
                                                                                                                    feature.shape[
                                                                                                                        1],
                                                                                                                    -1)
        if M_Ratio != 0:
            feature, mask = self.random_masking2(feature, M_Ratio)
        return feature
