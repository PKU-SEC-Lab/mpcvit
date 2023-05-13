import torch
import torch.nn.functional as F
import torch.nn as nn


class KLLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, target, T=1.0):
        output = output[0] if isinstance(output, tuple) else output
        target = target[0] if isinstance(target, tuple) else target
        output, target = output / T, target / T
        target_prob = F.softmax(target, dim=1)
        output_log_prob = F.log_softmax(output, dim=1)
        loss = - torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class KLTokenMSELoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        kd_type: str = "last",
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kl_loss = KLLossSoft(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.kd_type = kd_type

    def _kl_loss(self, output, target):
        return self.kl_loss(output, target)

    def _mse_loss(self, output, target):
        mse_loss = 0
        if self.kd_type == "last":
            if isinstance(output, torch.Tensor):
                _, N, _ = target.size()
                mse_loss = self.mse_loss(output[:, -N:], target)
            else:
                _, N, _ = target[-1].size()
                mse_loss = self.mse_loss(output[-1][:, -N:], target[-1])
        elif self.kd_type == "all":
            if isinstance(output, torch.Tensor):
                _, N, _ = target.size()
                mse_loss = self.mse_loss(output[:, -N:], target)
            else:
                assert len(output) == len(target)
                for i in range(len(output)):
                    _, N, _ = target[i].size()
                    mse_loss += self.mse_loss(output[i][:, -N:], target[i])
                mse_loss = mse_loss / len(output)
        else:
            raise NotImplementedError
        return mse_loss

    def forward(self, output, target):
        assert len(output) == len(target)
        kl_loss = self.kl_loss(output[0], target[0])
        mse_loss = self._mse_loss(output[1], target[1])
        loss = kl_loss + self.alpha * mse_loss
        # print(f"KL loss {kl_loss}, MSE loss {mse_loss}, total loss {loss}")

        return loss