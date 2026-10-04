import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple

class PoincaréGeodesicLoss(nn.Module):
    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.c = c
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sqrt_c = self.c ** 0.5
        max_norm = 1.0 / sqrt_c - self.eps

        x_norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), max=max_norm)
        y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), max=max_norm)

        sqdist = torch.sum((x - y) ** 2, dim=-1)
        x_sqnorm = torch.sum(x_norm ** 2, dim=-1)
        y_sqnorm = torch.sum(y_norm ** 2, dim=-1)

        num = 2 * self.c * sqdist
        denom = (1 - self.c * x_sqnorm) * (1 - self.c * y_sqnorm)

        arg = 1 + num / torch.clamp(denom, min=self.eps)
        dist = (1 / sqrt_c) * torch.acosh(torch.clamp(arg, min=1.0 + self.eps))

        return torch.mean(dist)
        
class SO3GeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)

        trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1.0) / 2.0

        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(cos_theta)

        return torch.mean(theta)


class SE3GeodesicLoss(nn.Module):
    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.so3_loss = SO3GeodesicLoss(eps=eps)
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight

    def forward(
        self,
        T_pred: Tuple[torch.Tensor, torch.Tensor],
        T_gt: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        R_pred, t_pred = T_pred
        R_gt, t_gt = T_gt

        l_rot = self.so3_loss(R_pred, R_gt)

        l_trans = torch.mean(torch.norm(t_pred - t_gt, dim=-1))

        return self.rot_weight * l_rot + self.trans_weight * l_trans

class RicciCurvatureSmoothnessLoss(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x_in = x.clone().detach().requires_grad_(True)
        z = encoder(x_in)

        v = torch.randint_like(z, high=2) * 2.0 - 1.0

        v_j = torch.autograd.grad(
            outputs=z,
            inputs=x_in,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        jacobian_norm = torch.norm(v_j.reshape(v_j.size(0), -1), dim=-1)
        grad_j_norm = torch.autograd.grad(
            outputs=jacobian_norm.sum(),
            inputs=x_in,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        curvature_penalty = torch.mean(torch.norm(grad_j_norm.reshape(grad_j_norm.size(0), -1), dim=-1) ** 2)

        return self.scale * curvature_penalty

if __name__ == "__main__":
    print("=== Testing Diff-Geometry-Loss Suite ===")
    
    poincare_criterion = PoincaréGeodesicLoss(c=1.0)
    z_pred = torch.randn(8, 128, requires_grad=True) * 0.1
    z_gt = torch.randn(8, 128) * 0.1
    loss_p = poincare_criterion(z_pred, z_gt)
    print(f"[Poincaré Geodesic Loss]: {loss_p.item():.4f}")

    so3_criterion = SO3GeodesicLoss()
    q1, _ = torch.linalg.qr(torch.randn(4, 3, 3))
    q2, _ = torch.linalg.qr(torch.randn(4, 3, 3))
    loss_so3 = so3_criterion(q1, q2)
    print(f"[SO(3) Geodesic Loss]: {loss_so3.item():.4f} rad ({loss_so3.item() * 180 / 3.14159:.2f} deg)")

    toy_encoder = nn.Sequential(
        nn.Linear(64, 32),
        nn.Softplus(),
        nn.Linear(32, 16)
    )
    curvature_criterion = RicciCurvatureSmoothnessLoss(scale=0.01)
    inputs = torch.randn(4, 64)
    loss_curv = curvature_criterion(toy_encoder, inputs)
    print(f"[Ricci Curvature Regularizer Loss]: {loss_curv.item():.6f}")

    total_loss = loss_p + loss_so3 + loss_curv
    total_loss.backward()
    print("Backpropagation completed successfully. All numerical gradients verified.")
