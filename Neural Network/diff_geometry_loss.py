import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Dict

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

class CliffordMultivector3D:
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 8, "Multivector in 3D requires 8 basis components."
        self.data = data

    @classmethod
    def from_vector(cls, v: torch.Tensor) -> "CliffordMultivector3D":
        zeros = torch.zeros_like(v[..., :1])
        mv = torch.cat([zeros, v, torch.zeros_like(v), zeros], dim=-1)
        return cls(mv)

    def geometric_product(self, other: "CliffordMultivector3D") -> "CliffordMultivector3D":
        a, b = self.data, other.data
        a0, a1, a2, a3, a12, a23, a31, a123 = a.unbind(-1)
        b0, b1, b2, b3, b12, b23, b31, b123 = b.unbind(-1)

        res = [
            a0*b0 + a1*b1 + a2*b2 + a3*b3 - a12*b12 - a23*b23 - a31*b31 - a123*b123, # scalar
            a0*b1 + a1*b0 - a2*b12 + a3*b31 + a12*b2 - a23*b123 - a31*b3 + a123*b23, # e1
            a0*b2 + a1*b12 + a2*b0 - a3*b23 - a12*b1 + a23*b3 - a31*b123 + a123*b31, # e2
            a0*b3 - a1*b31 + a2*b23 + a3*b0 - a12*b123 - a23*b2 + a31*b1 + a123*b12, # e3
            a0*b12 + a1*b2 - a2*b1 + a3*b123 + a12*b0 - a23*b31 + a31*b23 + a123*b3, # e12
            a0*b23 + a1*b123 + a2*b3 - a3*b2 + a12*b31 + a23*b0 - a31*b12 + a123*b1, # e23
            a0*b31 - a1*b3 + a2*b123 + a3*b1 - a12*b23 + a23*b12 + a31*b0 + a123*b2, # e31
            a0*b123 + a1*b23 + a2*b31 + a3*b12 + a12*b3 + a23*b1 + a31*b2 + a123*b0  # e123
        ]
        return CliffordMultivector3D(torch.stack(res, dim=-1))

    def reverse(self) -> "CliffordMultivector3D":
        signs = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], device=self.data.device, dtype=self.data.dtype)
        return CliffordMultivector3D(self.data * signs)


class RotorLayer3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bivector = nn.Parameter(torch.randn(channels, 3) * 0.1)
        self.angle = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        B, C, _ = v.shape
        b_norm = F.normalize(self.bivector, dim=-1, eps=1e-6)
        half_theta = self.angle / 2.0
        
        r_scalar = torch.cos(half_theta)
        r_bivector = -b_norm * torch.sin(half_theta)
        
        zeros = torch.zeros(C, 3, device=v.device)
        r_data = torch.cat([r_scalar, zeros, r_bivector, zeros[..., :1]], dim=-1)
        rotor = CliffordMultivector3D(r_data)
        rotor_rev = rotor.reverse()

        # Transform vectors
        v_mv = CliffordMultivector3D.from_vector(v)
        rotated_mv = rotor.geometric_product(v_mv).geometric_product(rotor_rev)
        
        return rotated_mv.data[..., 1:4]

class Wasserstein1DLoss(nn.Module):
    def __init__(self, num_projections: int = 64):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dim = x.size(-1)
        projections = torch.randn(dim, self.num_projections, device=x.device)
        projections = F.normalize(projections, dim=0)

        x_proj = torch.matmul(x, projections)
        y_proj = torch.matmul(y, projections)

        x_sorted, _ = torch.sort(x_proj, dim=0)
        y_sorted, _ = torch.sort(y_proj, dim=0)

        return torch.mean(torch.abs(x_sorted - y_sorted))


class DifferentiableVR0PersistenceLoss(nn.Module):
    def __init__(self, target_clusters: int = 1):
        super().__init__()
        self.target_clusters = target_clusters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        dist_matrix = torch.cdist(x, x, p=2)

        upper_tri_indices = torch.triu_indices(N, N, offset=1)
        edge_lengths = dist_matrix[upper_tri_indices[0], upper_tri_indices[1]]
        
        sorted_edges, _ = torch.sort(edge_lengths)
        
        persistence_penalty = torch.sum(sorted_edges[:N - self.target_clusters])
        return persistence_penalty

class LieEulerIntegratorSE3(nn.Module):
    def __init__(self, dt: float = 0.01):
        super().__init__()
        self.dt = dt

    @staticmethod
    def exp_so3(w: torch.Tensor) -> torch.Tensor:
        theta = torch.norm(w, dim=-1, keepdim=True) + 1e-8
        k = w / theta
        
        K = torch.zeros(w.size(0), 3, 3, device=w.device)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] = k[:, 0]

        I = torch.eye(3, device=w.device).unsqueeze(0)
        R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)
        return R

    def forward(
        self,
        T_curr: Tuple[torch.Tensor, torch.Tensor],
        velocity_field: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        steps: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R, t = T_curr
        for _ in range(steps):
            w, v = velocity_field(R, t)
            
            dR = self.exp_so3(w * self.dt)
            R = torch.bmm(R, dR)
            
            t = t + torch.bmm(R, (v * self.dt).unsqueeze(-1)).squeeze(-1)

        return R, t

if __name__ == "__main__":
    print("=== Testing Diff-Geometry-Loss Complete Suite ===")

    poincare_criterion = PoincaréGeodesicLoss(c=1.0)
    z_pred = torch.randn(8, 128, requires_grad=True) * 0.1
    z_gt = torch.randn(8, 128) * 0.1
    loss_p = poincare_criterion(z_pred, z_gt)
    print(f"[Poincaré Geodesic Loss]: {loss_p.item():.4f}")

    so3_criterion = SO3GeodesicLoss()
    q1, _ = torch.linalg.qr(torch.randn(4, 3, 3))
    q2, _ = torch.linalg.qr(torch.randn(4, 3, 3))
    loss_so3 = so3_criterion(q1, q2)
    print(f"[SO(3) Geodesic Loss]: {loss_so3.item():.4f} rad")

    toy_encoder = nn.Sequential(nn.Linear(64, 32), nn.Softplus(), nn.Linear(32, 16))
    curvature_criterion = RicciCurvatureSmoothnessLoss(scale=0.01)
    inputs = torch.randn(4, 64)
    loss_curv = curvature_criterion(toy_encoder, inputs)
    print(f"[Ricci Curvature Regularizer Loss]: {loss_curv.item():.6f}")

    rotor_layer = RotorLayer3D(channels=16)
    pts_3d = torch.randn(8, 16, 3, requires_grad=True)
    rotated_pts = rotor_layer(pts_3d)
    print(f"[Clifford Rotor Layer Output Shape]: {rotated_pts.shape}")

    # 5. Direction 2: 1D Sliced Wasserstein Loss
    ot_criterion = Wasserstein1DLoss(num_projections=32)
    loss_ot = ot_criterion(z_pred, z_gt)
    print(f"[1D Sliced Wasserstein Loss]: {loss_ot.item():.4f}")

    # 6. Direction 3: TDA Persistence Loss
    tda_criterion = DifferentiableVR0PersistenceLoss(target_clusters=2)
    cloud = torch.randn(20, 3, requires_grad=True)
    loss_tda = tda_criterion(cloud)
    print(f"[TDA 0D Persistence Loss]: {loss_tda.item():.4f}")

    integrator = LieEulerIntegratorSE3(dt=0.05)
    dummy_vel = lambda R, t: (torch.ones(4, 3), torch.ones(4, 3))
    R_init, _ = torch.linalg.qr(torch.randn(4, 3, 3))
    t_init = torch.zeros(4, 3)
    R_next, t_next = integrator((R_init, t_init), dummy_vel, steps=5)
    print(f"[SE(3) Lie Integrator Final Trajectory norm]: {torch.norm(t_next).item():.4f}")

    total_loss = loss_p + loss_so3 + loss_curv + loss_ot + loss_tda + torch.sum(rotated_pts)
    total_loss.backward()
    print("Backpropagation across all mathematical operations executed successfully.")
