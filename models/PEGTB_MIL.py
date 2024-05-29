import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import TransLayer
from utils.util import coords_norm

class PositionEncoder(nn.Module):
    def __init__(self, pos_dim=128):
        super(PositionEncoder, self).__init__()
        self.fc = torch.nn.Linear(in_features=2, out_features=pos_dim)
        self.norm = nn.LayerNorm(pos_dim)
        self.MHSA = TransLayer(dim=pos_dim, return_attn=False)

    def forward(self, coordinates):
        x = self.fc(coordinates)
        _, x = self.MHSA(x)
        x = self.norm(x)
        return x


class PositionDecoder(nn.Module):
    def __init__(self, later_dim=512):
        super(PositionDecoder, self).__init__()
        self.fc = torch.nn.Linear(in_features=later_dim, out_features=2)
        self.norm = nn.LayerNorm(later_dim)
        self.MHSA = TransLayer(dim=later_dim, return_attn=False)
        self.activation = nn.Sigmoid()

    def forward(self, pred_coordinates):
        _, x = self.MHSA(pred_coordinates)
        x = self.norm(x)
        x = self.fc(x)
        x = self.activation(x)
        return x


class PEGTBMIL(nn.Module):
    def __init__(self, n_classes=3, feat_dim=1024, pos_dim=128, later_dim=512, mask_ratio=0.25,
                 return_atte=False, return_pred_coordinates=False, return_pos_error=False):
        super(PEGTBMIL, self).__init__()

        self.mask_ratio = mask_ratio
        self.return_atte = return_atte
        self.return_pred_coordinates = return_pred_coordinates
        self.return_pos_error = return_pos_error

        self.PE = PositionEncoder(pos_dim=pos_dim)
        self.PD = PositionDecoder(later_dim=later_dim)

        self.fc = nn.Linear(feat_dim + pos_dim, later_dim)
        self.norm = nn.LayerNorm(later_dim)

        self.MHSA = TransLayer(dim=later_dim, return_attn=return_atte)

        self.classifier = nn.Linear(later_dim, n_classes)

        self.mask_token = nn.Parameter(torch.zeros((1, later_dim)))
        torch.nn.init.normal_(self.mask_token)

    def forward(self, in_features, in_coordinates):

        pos_emb = self.PE(in_coordinates)
        fused_features = torch.cat([in_features, pos_emb], dim=2)
        fused_features = self.fc(fused_features)
        fused_features = self.norm(fused_features)
        attn, tokens = self.MHSA(fused_features)
        slide_token = torch.mean(tokens, dim=1, keepdim=True)
        logits = self.classifier(slide_token)
        prob = F.softmax(logits, dim=-1)
        result_dict = {'logits': logits, 'Y_prob': prob}

        tmp_tokens = torch.randn(tokens.shape).to(tokens.device)
        tmp_tokens.copy_(tokens)
        pos_loss, pred_pos = self.reconstruction_loss(tmp_tokens, in_coordinates, self.mask_ratio)

        if self.return_pred_coordinates:
            result_dict['pred_pos'] = pred_pos

        if self.training or self.return_pos_error:
            result_dict['pos_loss'] = pos_loss

        if self.return_atte:
            result_dict['attn'] = attn

        return result_dict

    def reconstruction_loss(self, tokens, true_pos, mask_ratio, r=100):
        B, N, _ = tokens.shape

        len_mask = int(N * mask_ratio)
        shuffle_index = torch.LongTensor(random.sample(range(N), len_mask)).to(tokens.device)
        tokens[:, shuffle_index] = self.mask_token.repeat(len_mask, 1)

        pre_pos = self.PD(tokens)
        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = loss_fn(true_pos.float(), pre_pos.float())

        return r * loss, pre_pos.float()


if __name__ == '__main__':
    features = torch.randn((1, 500, 1024)).cuda()  # batch_size, num_patches, feature_dim
    coordinates = torch.randn((1, 500, 2)).cuda()  # batch_size, num_patches, coordinate_dim
    # coordinates = coords_norm(coordinates) # normalization
    net = PEGTBMIL(n_classes=3,
                   feat_dim=1024,
                   pos_dim=128,
                   later_dim=512,
                   mask_ratio=0.25,
                   return_atte=False,
                   return_pred_coordinates=False,
                   return_pos_error=False).cuda()
    net.train()
    output = net(features, coordinates)
    print(output)
