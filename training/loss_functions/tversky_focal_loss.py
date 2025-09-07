from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss
import torch


# TODO докстринг
class TverskyFocalLoss(torch.nn.Module):
    def __init__(
        self,
        mode: str = "binary",
        weight_focal: float = 0.5,
        weight_tversky: float = 0.5,
        smoothing: float = 0.1,
        alpha_tversky: float = 0.5,
        beta_tversky: float = 0.5,
        alpha_focal: float | None = None,
        gamma_focal: float = 2.0,
        ignore_index: int | None = 255,
    ):
        """Compound loss of sum of Tversky and Focal losses with moving average smoothing

        Args:
            mode (str, optional): Training mode. Defaults to "binary".
            weight_focal (float, optional): Weight of focal loss. Defaults to 0.5.
            weight_tversky (float, optional): Weight of tversky loss. Defaults to 0.5.
            smoothing (float, optional): Smoothing coefficient. Defaults to 0.1.
            alpha_tversky (float, optional): Alpha parameter for tversky. Defaults to 0.5.
            beta_tversky (float, optional): Beta parameter for tversky. Defaults to 0.5.
            alpha_focal (float | None, optional): Alpha parameter for focal. Defaults to None.
            gamma_focal (float, optional): Gamma parameter for focal. Defaults to 2.0.
            ignore_index (int | None, optional): Index not used in computing loss. Defaults to 255.
        """
        super().__init__()

        self.alpha_tversky = alpha_tversky
        self.beta_tversky = beta_tversky
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal

        self.tversky = TverskyLoss(
            mode=mode,
            from_logits=True,
            alpha=alpha_tversky,
            beta=beta_tversky,
            ignore_index=ignore_index,
        )
        self.focal = FocalLoss(
            mode=mode, alpha=alpha_focal, gamma=gamma_focal, ignore_index=ignore_index
        )

        # weights
        self.weight_focal = weight_focal
        self.weight_tversky = weight_tversky

        # moving average
        self.smooth = smoothing
        self.focal_avg = 1.0
        self.tversky_avg = 1.0

    def forward(self, y_pred, y_true):
        focal_loss = self.focal(y_pred, y_true)
        tversky_loss = self.tversky(y_pred, y_true)

        self.focal_avg = (
            1 - self.smooth
        ) * self.focal_avg + self.smooth * focal_loss.detach()
        self.tversky_avg = (
            1 - self.smooth
        ) * self.tversky_avg + self.smooth * tversky_loss.detach()

        focal_norm = focal_loss / (self.focal_avg + 1e-8)
        tversky_norm = tversky_loss / (self.tversky_avg + 1e-8)

        loss = self.weight_focal * focal_norm + self.weight_tversky * tversky_norm
        return loss

    def __str__(self):
        res = f"{self.weight_tversky} * Tversky(alpha={self.alpha_tversky}, beta={self.beta_tversky}) + \
            + {self.weight_Focal} * Tversky(alpha={self.alpha_focal}, gamma={self.gamma_focal})"
        return res
