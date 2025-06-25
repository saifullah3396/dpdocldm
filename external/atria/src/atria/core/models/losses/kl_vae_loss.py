import torch
import torch.nn as nn
import torch.nn.functional as F


class KLVAELoss(nn.Module):
    def __init__(
        self,
        kl_weight=1.0,
        pixelloss_weight=1.0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
    ):
        rec_loss = F.mse_loss(reconstructions, inputs)
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = self.pixel_weight * rec_loss + self.kl_weight * kl_loss

        log = {
            "ae_loss": loss.clone().detach().mean(),
            "kl_loss": kl_loss.detach().mean(),
            "rec_loss": rec_loss.detach().mean(),
        }
        return loss, log
