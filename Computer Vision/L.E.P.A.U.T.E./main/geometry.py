import os
import ctypes
import platform
import numpy as np
import torch
from globals import mps_safe, logger

def _load_c_library():
    system = platform.system()
    if system == "Darwin":
        ext = ".dylib"
    elif system == "Windows":
        ext = ".dll"
    else:
        ext = ".so"

    this_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(this_dir, "..", "core", "build", f"liblepaute_core{ext}"),
        os.path.join(this_dir, "..", "core", f"liblepaute_core{ext}"),
        os.path.join(this_dir, f"liblepaute_core{ext}"),
        f"liblepaute_core{ext}",
    ]

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                lib = ctypes.CDLL(abs_path)
                logger.info(f"[Geometry C-Core] Loaded: {abs_path}")
                return lib
            except Exception as e:
                logger.warning(f"[Geometry C-Core] Failed to load {abs_path}: {e}")

    logger.warning("[Geometry C-Core] Native library not found – using pure PyTorch fallback")
    return None

_c_lib = _load_c_library()

class LePauteSe3(ctypes.Structure):
    _fields_ = [("data", ctypes.c_double * 16)]

class LePauteTangent(ctypes.Structure):
    _fields_ = [("data", ctypes.c_double * 6)]

class LePauteIntrinsics(ctypes.Structure):
    _fields_ = [
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
    ]

class LePauteGnConfig(ctypes.Structure):
    _fields_ = [
        ("num_levels", ctypes.c_int),
        ("max_iters_per_level", ctypes.c_int),
        ("huber_delta", ctypes.c_double),
        ("use_robust_loss", ctypes.c_int),
        ("initial_lm_lambda", ctypes.c_double),
        ("min_grad_thresh", ctypes.c_double),
    ]

if _c_lib is not None:
    try:
        _c_lib.lepaute_skew_symmetric.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _c_lib.lepaute_skew_symmetric.restype = None

        _c_lib.lepaute_se3_exp.argtypes = [
            ctypes.POINTER(LePauteTangent),
            ctypes.POINTER(LePauteSe3),
        ]
        _c_lib.lepaute_se3_exp.restype = None

        _c_lib.lepaute_se3_log.argtypes = [
            ctypes.POINTER(LePauteSe3),
            ctypes.POINTER(LePauteTangent),
        ]
        _c_lib.lepaute_se3_log.restype = None

        _c_lib.lepaute_se3_adjoint.argtypes = [
            ctypes.POINTER(LePauteSe3),
            ctypes.POINTER(ctypes.c_double),
        ]
        _c_lib.lepaute_se3_adjoint.restype = None

        _c_lib.lepaute_so3_left_jacobian.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _c_lib.lepaute_so3_left_jacobian.restype = None

        _c_lib.lepaute_so3_left_jacobian_inv.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _c_lib.lepaute_so3_left_jacobian_inv.restype = None

        _c_lib.lepaute_lm_refine_pose_pyramid.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(LePauteIntrinsics),
            ctypes.POINTER(LePauteGnConfig),
            ctypes.POINTER(LePauteSe3),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _c_lib.lepaute_lm_refine_pose_pyramid.restype = ctypes.c_double

        _HAS_OPTIMIZATION = True
        logger.info("[Geometry C-Core] Photometric LM optimization symbols loaded")
    except AttributeError as e:
        logger.warning(f"[Geometry C-Core] Missing symbols, partial fallback: {e}")
        _HAS_OPTIMIZATION = False
else:
    _HAS_OPTIMIZATION = False
    
def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    if _c_lib is not None and not v.requires_grad:
        B = v.shape[0]
        v_np = v.detach().cpu().numpy().astype(np.float64)
        out_np = np.zeros((B, 3, 3), dtype=np.float64)
        for i in range(B):
            _c_lib.lepaute_skew_symmetric(
                v_np[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_np[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        return torch.from_numpy(out_np).to(device=v.device, dtype=v.dtype)

    with mps_safe(v.device):
        B = v.shape[0]
        zero = torch.zeros(B, device=v.device, dtype=v.dtype)
        return torch.stack([
            zero, -v[:, 2],  v[:, 1],
            v[:, 2],  zero, -v[:, 0],
           -v[:, 1],  v[:, 0],  zero
        ], dim=1).view(B, 3, 3)


def se3_exp_map(xi: torch.Tensor) -> torch.Tensor:
    if _c_lib is not None and not xi.requires_grad:
        B = xi.shape[0]
        xi_np = xi.detach().cpu().numpy().astype(np.float64)
        T_np = np.zeros((B, 4, 4), dtype=np.float64)
        for i in range(B):
            tangent = LePauteTangent()
            for j in range(6):
                tangent.data[j] = xi_np[i, j]
            se3 = LePauteSe3()
            _c_lib.lepaute_se3_exp(ctypes.byref(tangent), ctypes.byref(se3))
            T_np[i] = np.ctypeslib.as_array(se3.data).reshape(4, 4)
        return torch.from_numpy(T_np).to(device=xi.device, dtype=xi.dtype)

    with mps_safe(xi.device):
        B = xi.shape[0]
        rho, phi = xi[:, :3], xi[:, 3:]
        theta_sq = torch.sum(phi ** 2, dim=1, keepdim=True)
        theta = torch.sqrt(torch.clamp(theta_sq, min=1e-20))

        T = torch.eye(4, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B, 1, 1)
        zero = torch.zeros(B, device=xi.device, dtype=xi.dtype)
        K = torch.stack([
            zero, -phi[:, 2],  phi[:, 1],
            phi[:, 2],  zero, -phi[:, 0],
           -phi[:, 1],  phi[:, 0],  zero
        ], dim=1).view(B, 3, 3)
        K2 = torch.bmm(K, K)

        mask_large = (theta.squeeze(1) > 1e-4)
        mask_small = ~mask_large

        if mask_large.any():
            th = theta[mask_large]
            th2 = th ** 2
            A = torch.sin(th) / th
            Bcoef = (1.0 - torch.cos(th)) / th2
            C = (1.0 - A) / th2
            I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(mask_large.sum(), -1, -1)
            Kl, K2l = K[mask_large], K2[mask_large]
            T[mask_large, :3, :3] = I + A.unsqueeze(-1) * Kl + Bcoef.unsqueeze(-1) * K2l
            V = I + Bcoef.unsqueeze(-1) * Kl + C.unsqueeze(-1) * K2l
            T[mask_large, :3, 3] = torch.bmm(V, rho[mask_large].unsqueeze(-1)).squeeze(-1)

        if mask_small.any():
            th2 = theta_sq[mask_small]
            A = 1.0 - th2 / 6.0 + (th2 * th2) / 120.0
            Bcoef = 0.5 - th2 / 24.0 + (th2 * th2) / 720.0
            C = 1.0 / 6.0 - th2 / 120.0 + (th2 * th2) / 5040.0
            I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(mask_small.sum(), -1, -1)
            Ks, K2s = K[mask_small], K2[mask_small]
            T[mask_small, :3, :3] = I + A.unsqueeze(-1) * Ks + Bcoef.unsqueeze(-1) * K2s
            V = I + Bcoef.unsqueeze(-1) * Ks + C.unsqueeze(-1) * K2s
            T[mask_small, :3, 3] = torch.bmm(V, rho[mask_small].unsqueeze(-1)).squeeze(-1)

        return T


def se3_log_map(T: torch.Tensor) -> torch.Tensor:
    if _c_lib is not None and not T.requires_grad:
        B = T.shape[0]
        T_np = T.detach().cpu().numpy().astype(np.float64)
        xi_np = np.zeros((B, 6), dtype=np.float64)
        for i in range(B):
            se3 = LePauteSe3()
            for j in range(16):
                se3.data[j] = T_np[i].flat[j]
            tangent = LePauteTangent()
            _c_lib.lepaute_se3_log(ctypes.byref(se3), ctypes.byref(tangent))
            xi_np[i] = np.ctypeslib.as_array(tangent.data)
        return torch.from_numpy(xi_np).to(device=T.device, dtype=T.dtype)

    with mps_safe(T.device):
        B = T.shape[0]
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        trace_R = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        cos_theta = torch.clamp((trace_R - 1.0) * 0.5, -1.0, 1.0)
        theta = torch.acos(cos_theta)

        xi = torch.zeros(B, 6, device=T.device, dtype=T.dtype)
        phi_raw = torch.stack([
            R[:, 2, 1] - R[:, 1, 2],
            R[:, 0, 2] - R[:, 2, 0],
            R[:, 1, 0] - R[:, 0, 1]
        ], dim=1)

        mask_small = (theta < 1e-4)
        mask_pi    = (theta > (torch.pi - 1e-3))
        mask_large = ~(mask_small | mask_pi)

        if mask_large.any():
            th = theta[mask_large].unsqueeze(-1)
            sin_th = torch.sin(th)
            phi_l = (th / (2.0 * sin_th)) * phi_raw[mask_large]
            xi[mask_large, 3:] = phi_l
            K = skew_symmetric(phi_l)
            K2 = torch.bmm(K, K)
            I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_large.sum(), -1, -1)
            half_th = th * 0.5
            coef = (1.0 - (th * torch.cos(half_th)) / (2.0 * torch.sin(half_th))) / (th ** 2)
            V_inv = I - 0.5 * K + coef * K2
            xi[mask_large, :3] = torch.bmm(V_inv, t[mask_large].unsqueeze(-1)).squeeze(-1)

        if mask_small.any():
            phi_s = 0.5 * phi_raw[mask_small]
            xi[mask_small, 3:] = phi_s
            K = skew_symmetric(phi_s)
            K2 = torch.bmm(K, K)
            I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_small.sum(), -1, -1)
            V_inv = I - 0.5 * K + (1.0 / 12.0) * K2
            xi[mask_small, :3] = torch.bmm(V_inv, t[mask_small].unsqueeze(-1)).squeeze(-1)

        if mask_pi.any():
            th = theta[mask_pi].unsqueeze(-1)
            R_pi = R[mask_pi]
            t_pi = t[mask_pi]
            A = (R_pi + torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0)) * 0.5
            diag_A = torch.diagonal(A, dim1=-2, dim2=-1)
            max_idx = torch.argmax(diag_A, dim=-1)
            batch_idx = torch.arange(R_pi.shape[0], device=T.device)
            v = A[batch_idx, :, max_idx]
            v = v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-10)
            sign = torch.sign(torch.sum(v * phi_raw[mask_pi], dim=-1, keepdim=True))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            v = v * sign
            phi_pi = th * v
            xi[mask_pi, 3:] = phi_pi
            K = skew_symmetric(phi_pi)
            K2 = torch.bmm(K, K)
            I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_pi.sum(), -1, -1)
            half_th = th * 0.5
            coef = (1.0 - (th * torch.cos(half_th)) / (2.0 * torch.sin(half_th))) / (th ** 2)
            V_inv = I - 0.5 * K + coef * K2
            xi[mask_pi, :3] = torch.bmm(V_inv, t_pi.unsqueeze(-1)).squeeze(-1)

        return xi


def se3_adjoint(T: torch.Tensor) -> torch.Tensor:
    if _c_lib is not None and not T.requires_grad:
        B = T.shape[0]
        T_np = T.detach().cpu().numpy().astype(np.float64)
        Ad_np = np.zeros((B, 6, 6), dtype=np.float64)
        for i in range(B):
            se3 = LePauteSe3()
            for j in range(16):
                se3.data[j] = T_np[i].flat[j]
            Ad = (ctypes.c_double * 36)()
            _c_lib.lepaute_se3_adjoint(ctypes.byref(se3), Ad)
            Ad_np[i] = np.ctypeslib.as_array(Ad).reshape(6, 6)
        return torch.from_numpy(Ad_np).to(device=T.device, dtype=T.dtype)

    with mps_safe(T.device):
        B = T.shape[0]
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        t_hat = skew_symmetric(t)
        t_hat_R = torch.bmm(t_hat, R)
        zeros = torch.zeros(B, 3, 3, device=T.device, dtype=T.dtype)
        top = torch.cat([R, t_hat_R], dim=2)
        bottom = torch.cat([zeros, R], dim=2)
        return torch.cat([top, bottom], dim=1)


def so3_left_jacobian(phi: torch.Tensor) -> torch.Tensor:
    if _c_lib is not None and not phi.requires_grad:
        B = phi.shape[0]
        phi_np = phi.detach().cpu().numpy().astype(np.float64)
        Jl_np = np.zeros((B, 3, 3), dtype=np.float64)
        for i in range(B):
            _c_lib.lepaute_so3_left_jacobian(
                phi_np[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                Jl_np[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        return torch.from_numpy(Jl_np).to(device=phi.device, dtype=phi.dtype)

    with mps_safe(phi.device):
        B = phi.shape[0]
        theta_sq = torch.sum(phi ** 2, dim=1, keepdim=True)
        theta = torch.sqrt(torch.clamp(theta_sq, min=1e-20))
        K = skew_symmetric(phi)
        K2 = torch.bmm(K, K)
        I = torch.eye(3, device=phi.device, dtype=phi.dtype).unsqueeze(0).expand(B, -1, -1)
        mask_large = (theta.squeeze(1) > 1e-4)
        Bcoef = torch.where(mask_large.unsqueeze(-1),
                            (1.0 - torch.cos(theta)) / theta_sq,
                            0.5 - theta_sq / 24.0)
        A = torch.sin(theta) / theta
        Ccoef = torch.where(mask_large.unsqueeze(-1),
                            (1.0 - A) / theta_sq,
                            1.0 / 6.0 - theta_sq / 120.0)
        return I + Bcoef.unsqueeze(-1) * K + Ccoef.unsqueeze(-1) * K2


def so3_left_jacobian_inv(phi: torch.Tensor) -> torch.Tensor:
    if _c_lib is not None and not phi.requires_grad:
        B = phi.shape[0]
        phi_np = phi.detach().cpu().numpy().astype(np.float64)
        Jl_inv_np = np.zeros((B, 3, 3), dtype=np.float64)
        for i in range(B):
            _c_lib.lepaute_so3_left_jacobian_inv(
                phi_np[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                Jl_inv_np[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        return torch.from_numpy(Jl_inv_np).to(device=phi.device, dtype=phi.dtype)

    with mps_safe(phi.device):
        B = phi.shape[0]
        theta_sq = torch.sum(phi ** 2, dim=1, keepdim=True)
        theta = torch.sqrt(torch.clamp(theta_sq, min=1e-20))
        K = skew_symmetric(phi)
        K2 = torch.bmm(K, K)
        I = torch.eye(3, device=phi.device, dtype=phi.dtype).unsqueeze(0).expand(B, -1, -1)
        mask_large = (theta.squeeze(1) > 1e-4)
        half_th = theta * 0.5
        coef = torch.where(
            mask_large.unsqueeze(-1),
            (1.0 - (theta * torch.cos(half_th)) / (2.0 * torch.sin(half_th))) / theta_sq,
            torch.full_like(theta_sq, 1.0 / 12.0)
        )
        return I - 0.5 * K + coef.unsqueeze(-1) * K2

def compose_poses(T_global: torch.Tensor, T_rel: torch.Tensor) -> torch.Tensor:
    with mps_safe(T_global.device):
        return torch.bmm(T_global, T_rel)

def lm_refine_pose_pyramid(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    *,
    num_levels: int = 3,
    max_iters: int = 10,
    huber_delta: float = 10.0,
    use_robust: bool = True,
    initial_lambda: float = 1e-3,
    min_grad_thresh: float = 25.0,
    T_init: np.ndarray = None,
    scale_prior: float = 1.0,
) -> tuple[np.ndarray, float]:
    if not _HAS_OPTIMIZATION:
        raise RuntimeError("C optimization symbols not available")

    assert ref_gray.dtype == np.uint8 and cur_gray.dtype == np.uint8
    assert ref_gray.ndim == 2 and cur_gray.ndim == 2
    h, w = ref_gray.shape

    ref_c = np.ascontiguousarray(ref_gray, dtype=np.uint8)
    cur_c = np.ascontiguousarray(cur_gray, dtype=np.uint8)

    inv_d_val = 1.0 / max(float(scale_prior), 0.1)
    inv_depth = np.full((h, w), inv_d_val, dtype=np.float32)
    inv_depth_c = np.ascontiguousarray(inv_depth)

    K = LePauteIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))

    cfg = LePauteGnConfig(
        num_levels=int(num_levels),
        max_iters_per_level=int(max_iters),
        huber_delta=float(huber_delta),
        use_robust_loss=1 if use_robust else 0,
        initial_lm_lambda=float(initial_lambda),
        min_grad_thresh=float(min_grad_thresh),
    )

    T = LePauteSe3()
    if T_init is not None:
        T_flat = np.asarray(T_init, dtype=np.float64).ravel()
        for i in range(16):
            T.data[i] = T_flat[i]
    else:
        for i in range(16):
            T.data[i] = 0.0
        T.data[0] = T.data[5] = T.data[10] = T.data[15] = 1.0

    photo_a = ctypes.c_double(1.0)
    photo_b = ctypes.c_double(0.0)

    cost = _c_lib.lepaute_lm_refine_pose_pyramid(
        ref_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        cur_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        inv_depth_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(w),
        ctypes.c_int(h),
        ctypes.byref(K),
        ctypes.byref(cfg),
        ctypes.byref(T),
        ctypes.byref(photo_a),
        ctypes.byref(photo_b),
    )

    tangent = LePauteTangent()
    _c_lib.lepaute_se3_log(ctypes.byref(T), ctypes.byref(tangent))
    xi = np.ctypeslib.as_array(tangent.data).copy()

    score = float(1.0 / (1.0 + np.sqrt(max(cost, 0.0))))
    score = float(np.clip(score, 0.0, 0.95))

    return xi.astype(np.float32), score


def has_c_optimization() -> bool:
    return _HAS_OPTIMIZATION