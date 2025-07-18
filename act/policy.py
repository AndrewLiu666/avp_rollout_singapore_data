import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import v2
import torch

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.qpos_noise_std = args_override['qpos_noise_std']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        patch_h = 16
        patch_w = 22
        if actions is not None: # training time
            # transform = v2.Compose([
            #     v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            #     v2.RandomPerspective(distortion_scale=0.5),
            #     v2.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
            #     v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
            #     v2.Normalize(
            #         mean=[0.485, 0.456, 0.406],
            #         std=[0.229, 0.224, 0.225])
            # ])
            transform = v2.Compose([
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                v2.RandomPerspective(distortion_scale=0.5),
                v2.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
                v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
                v2.Resize((patch_h * 14, patch_w * 14)),
                # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            qpos += (self.qpos_noise_std**0.5)*torch.randn_like(qpos)
        else: # inference time
            transform = v2.Compose([
                v2.Resize((patch_h * 14, patch_w * 14)),
                # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
        image = transform(image)

        # 14 -> 28
        # if qpos.shape[-1] == 14 and self.model.state_dim == 28:
        #     padded = torch.zeros(*qpos.shape[:-1], 28, device=qpos.device, dtype=qpos.dtype)
        #     padded[..., :14] = qpos
        #     qpos = padded

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
