#include "optimization.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef LEPAUTE_STATIC_MEM

static uint8_t s_pyramid_pool[2][LEPAUTE_MAX_PYRAMID_LEVELS][LEPAUTE_MAX_IMAGE_PIXELS];
static int     s_pool_used[2] = {0, 0};

static uint8_t* static_alloc(int which, int pixels)
{
    if (which < 0 || which > 1) return NULL;
    if (s_pool_used[which]) return NULL;
    if (pixels > LEPAUTE_MAX_IMAGE_PIXELS) return NULL;
    s_pool_used[which] = 1;
    return s_pyramid_pool[which][0];
}

static void static_free(int which)
{
    if (which >= 0 && which <= 1)
        s_pool_used[which] = 0;
}

#endif

static void gaussian_downsample_2x(const uint8_t* src, int w, int h, uint8_t* dst)
{
    int nw = w / 2, nh = h / 2;
    for (int y = 0; y < nh; ++y) {
        for (int x = 0; x < nw; ++x) {
            int cx = x * 2, cy = y * 2;
            int sum = 0, weight_sum = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int px = cx + dx, py = cy + dy;
                    if (px < 0) px = 0; if (px >= w) px = w - 1;
                    if (py < 0) py = 0; if (py >= h) py = h - 1;
                    int w_k = (dx == 0 ? 2 : 1) * (dy == 0 ? 2 : 1);
                    sum += src[py * w + px] * w_k;
                    weight_sum += w_k;
                }
            }
            dst[y * nw + x] = (uint8_t)(sum / weight_sum);
        }
    }
}

int lepaute_create_pyramid(const uint8_t* src, int w, int h,
                           int levels, lepaute_image_t* pyramid)
{
    if (!src || !pyramid || w <= 0 || h <= 0 || levels <= 0)
        return -1;
    if (levels > LEPAUTE_MAX_PYRAMID_LEVELS)
        levels = LEPAUTE_MAX_PYRAMID_LEVELS;

#ifdef LEPAUTE_STATIC_MEM
    static int next_slot = 0;
    int slot = next_slot;
    next_slot = 1 - next_slot;

    if (s_pool_used[slot]) {
        return -2;
    }
    s_pool_used[slot] = 1;

    int offset = 0;
    for (int l = 0; l < levels; ++l) {
        int cur_w = (l == 0) ? w : (pyramid[l-1].width / 2);
        int cur_h = (l == 0) ? h : (pyramid[l-1].height / 2);
        if (cur_w < 4 || cur_h < 4) {
            levels = l;
            break;
        }
        if (offset + cur_w * cur_h > LEPAUTE_MAX_IMAGE_PIXELS) {
            s_pool_used[slot] = 0;
            return -3;
        }
        pyramid[l].width  = cur_w;
        pyramid[l].height = cur_h;
        pyramid[l].data   = &s_pyramid_pool[slot][0][offset];
        offset += cur_w * cur_h;
    }

    memcpy(pyramid[0].data, src, (size_t)w * h);

    for (int l = 1; l < levels; ++l) {
        gaussian_downsample_2x(pyramid[l-1].data,
                               pyramid[l-1].width, pyramid[l-1].height,
                               pyramid[l].data);
    }

    for (int l = levels; l < LEPAUTE_MAX_PYRAMID_LEVELS; ++l) {
        pyramid[l].data = NULL;
        pyramid[l].width = pyramid[l].height = 0;
    }

    pyramid[0].height = (pyramid[0].height & 0x7FFF) | (slot << 15);

#else
    pyramid[0].width  = w;
    pyramid[0].height = h;
    pyramid[0].data   = (uint8_t*)malloc((size_t)w * h);
    if (!pyramid[0].data) return -1;
    memcpy(pyramid[0].data, src, (size_t)w * h);

    for (int l = 1; l < levels; ++l) {
        int prev_w = pyramid[l-1].width;
        int prev_h = pyramid[l-1].height;
        int cur_w  = prev_w / 2;
        int cur_h  = prev_h / 2;
        if (cur_w < 4 || cur_h < 4) {
            levels = l;
            break;
        }
        pyramid[l].width  = cur_w;
        pyramid[l].height = cur_h;
        pyramid[l].data   = (uint8_t*)malloc((size_t)cur_w * cur_h);
        if (!pyramid[l].data) {
            for (int k = 0; k < l; ++k) free(pyramid[k].data);
            return -1;
        }
        gaussian_downsample_2x(pyramid[l-1].data, prev_w, prev_h, pyramid[l].data);
    }

    for (int l = levels; l < LEPAUTE_MAX_PYRAMID_LEVELS; ++l) {
        pyramid[l].data = NULL;
        pyramid[l].width = pyramid[l].height = 0;
    }
#endif

    return 0;
}

void lepaute_free_pyramid(lepaute_image_t* pyramid, int levels)
{
    if (!pyramid) return;

#ifdef LEPAUTE_STATIC_MEM
    int slot = (pyramid[0].height >> 15) & 1;
    pyramid[0].height &= 0x7FFF;
    static_free(slot);

    for (int l = 0; l < LEPAUTE_MAX_PYRAMID_LEVELS; ++l) {
        pyramid[l].data = NULL;
        pyramid[l].width = pyramid[l].height = 0;
    }
#else
    for (int l = 0; l < levels && l < LEPAUTE_MAX_PYRAMID_LEVELS; ++l) {
        if (pyramid[l].data) {
            free(pyramid[l].data);
            pyramid[l].data = NULL;
        }
    }
#endif
}

static inline double compute_huber_weight(double residual, double delta)
{
    double abs_r = fabs(residual);
    return (abs_r <= delta) ? 1.0 : (delta / abs_r);
}

static inline double bilinear_interpolate_with_grad(
    const uint8_t* img, int w, int h,
    double x, double y, double* gx, double* gy)
{
    if (x < 1.0 || x >= (double)(w - 2) || y < 1.0 || y >= (double)(h - 2))
        return -1.0;

    int ix = (int)x, iy = (int)y;
    double fx = x - (double)ix, fy = y - (double)iy;

    double p00 = (double)img[iy * w + ix];
    double p10 = (double)img[iy * w + (ix + 1)];
    double p01 = (double)img[(iy + 1) * w + ix];
    double p11 = (double)img[(iy + 1) * w + (ix + 1)];

    if (gx && gy) {
        *gx = (p10 - p00) * (1.0 - fy) + (p11 - p01) * fy;
        *gy = (p01 - p00) * (1.0 - fx) + (p11 - p10) * fx;
    }
    return (1.0 - fx) * (1.0 - fy) * p00 + fx * (1.0 - fy) * p10
         + (1.0 - fx) * fy * p01 + fx * fy * p11;
}

static bool solve_linear_NxN(int N, const double* A, const double* b, double* x)
{
    double M[8][9];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) M[i][j] = A[i * N + j];
        M[i][N] = b[i];
    }
    for (int i = 0; i < N; ++i) {
        int max_r = i;
        double max_v = fabs(M[i][i]);
        for (int k = i + 1; k < N; ++k) {
            if (fabs(M[k][i]) > max_v) {
                max_v = fabs(M[k][i]);
                max_r = k;
            }
        }
        if (max_v < 1e-12) return false;
        if (max_r != i) {
            for (int j = i; j <= N; ++j) {
                double tmp = M[i][j];
                M[i][j] = M[max_r][j];
                M[max_r][j] = tmp;
            }
        }
        for (int k = i + 1; k < N; ++k) {
            double factor = M[k][i] / M[i][i];
            for (int j = i; j <= N; ++j)
                M[k][j] -= factor * M[i][j];
        }
    }
    for (int i = N - 1; i >= 0; --i) {
        double sum = M[i][N];
        for (int j = i + 1; j < N; ++j)
            sum -= M[i][j] * x[j];
        x[i] = sum / M[i][i];
    }
    return true;
}

static double compute_cost(
    const lepaute_image_t* ref_img, const lepaute_image_t* cur_img,
    const float* ref_inv_depth, int width, int height,
    const lepaute_intrinsics_t* K, const lepaute_se3_t* T_cw,
    double photo_a, double photo_b, double scale, double huber_delta,
    int use_robust, int step, double min_grad_thresh)
{
    int lw = ref_img->width, lh = ref_img->height;
    double cost_sum = 0.0;
    int valid_cnt = 0;

    for (int v = 2; v < lh - 2; v += step) {
        for (int u = 2; u < lw - 2; u += step) {
            double I_ref = (double)ref_img->data[v * lw + u];

            int orig_u = (int)(u / scale);
            int orig_v = (int)(v / scale);
            double inv_d = (ref_inv_depth) ?
                (double)ref_inv_depth[orig_v * width + orig_u] : 1.0;
            if (inv_d <= 0.0) inv_d = 1.0;
            double Z_ref = 1.0 / inv_d;

            double X_ref = (u - K->cx) * Z_ref / K->fx;
            double Y_ref = (v - K->cy) * Z_ref / K->fy;

            double X_cur = T_cw->data[0] * X_ref + T_cw->data[1] * Y_ref
                         + T_cw->data[2] * Z_ref + T_cw->data[3];
            double Y_cur = T_cw->data[4] * X_ref + T_cw->data[5] * Y_ref
                         + T_cw->data[6] * Z_ref + T_cw->data[7];
            double Z_cur = T_cw->data[8] * X_ref + T_cw->data[9] * Y_ref
                         + T_cw->data[10] * Z_ref + T_cw->data[11];

            if (Z_cur <= 0.1) continue;

            double u_cur = K->fx * (X_cur / Z_cur) + K->cx;
            double v_cur = K->fy * (Y_cur / Z_cur) + K->cy;

            double gx = 0.0, gy = 0.0;
            double I_cur = bilinear_interpolate_with_grad(
                cur_img->data, lw, lh, u_cur, v_cur, &gx, &gy);
            if (I_cur < 0.0) continue;

            if (min_grad_thresh > 0.0 && (gx*gx + gy*gy) < min_grad_thresh)
                continue;

            double res = I_cur - (photo_a * I_ref + photo_b);
            double w_huber = use_robust ? compute_huber_weight(res, huber_delta) : 1.0;

            cost_sum += w_huber * res * res;
            valid_cnt++;
        }
    }
    return valid_cnt > 0 ? (cost_sum / valid_cnt) : 1e9;
}

double lepaute_lm_refine_pose_pyramid(
    const uint8_t* ref_img, const uint8_t* cur_img, const float* ref_inv_depth,
    int width, int height, const lepaute_intrinsics_t* intrinsics,
    const lepaute_gn_config_t* config, lepaute_se3_t* T_cw,
    double* photo_a, double* photo_b)
{
    if (!ref_img || !cur_img || !intrinsics || !config || !T_cw)
        return 1e9;

    int levels = (config->num_levels > 0) ? config->num_levels : 3;
    if (levels > LEPAUTE_MAX_PYRAMID_LEVELS)
        levels = LEPAUTE_MAX_PYRAMID_LEVELS;

    lepaute_image_t ref_pyr[LEPAUTE_MAX_PYRAMID_LEVELS];
    lepaute_image_t cur_pyr[LEPAUTE_MAX_PYRAMID_LEVELS];

    if (lepaute_create_pyramid(ref_img, width, height, levels, ref_pyr) != 0)
        return 1e9;
    if (lepaute_create_pyramid(cur_img, width, height, levels, cur_pyr) != 0) {
        lepaute_free_pyramid(ref_pyr, levels);
        return 1e9;
    }

    double local_a = 1.0, local_b = 0.0;
    double* pa = photo_a ? photo_a : &local_a;
    double* pb = photo_b ? photo_b : &local_b;

    double current_cost = 0.0;
    double grad_thresh = (config->min_grad_thresh > 0.0) ?
                         config->min_grad_thresh : 25.0;

    for (int level = levels - 1; level >= 0; --level) {
        int lw = ref_pyr[level].width;
        int lh = ref_pyr[level].height;
        if (lw < 8 || lh < 8) continue;

        double scale = (double)lw / (double)width;

        lepaute_intrinsics_t K_level = {
            intrinsics->fx * (float)scale,
            intrinsics->fy * (float)scale,
            intrinsics->cx * (float)scale,
            intrinsics->cy * (float)scale
        };

        double lambda = (config->initial_lm_lambda > 0.0) ?
                        config->initial_lm_lambda : 1e-3;
        int step = (level == 0) ? 2 : 1;

        current_cost = compute_cost(
            &ref_pyr[level], &cur_pyr[level], ref_inv_depth,
            width, height, &K_level, T_cw,
            *pa, *pb, scale,
            config->huber_delta, config->use_robust_loss,
            step, grad_thresh);

        for (int iter = 0; iter < config->max_iters_per_level; ++iter) {
            double H[64] = {0.0};
            double b[8]  = {0.0};
            int valid_points = 0;

            for (int v = 2; v < lh - 2; v += step) {
                for (int u = 2; u < lw - 2; u += step) {
                    double I_ref = (double)ref_pyr[level].data[v * lw + u];

                    int orig_u = (int)(u / scale);
                    int orig_v = (int)(v / scale);
                    double inv_d = (ref_inv_depth) ?
                        (double)ref_inv_depth[orig_v * width + orig_u] : 1.0;
                    if (inv_d <= 0.0) inv_d = 1.0;
                    double Z_ref = 1.0 / inv_d;

                    double X_ref = (u - K_level.cx) * Z_ref / K_level.fx;
                    double Y_ref = (v - K_level.cy) * Z_ref / K_level.fy;

                    double X_cur = T_cw->data[0] * X_ref + T_cw->data[1] * Y_ref
                                 + T_cw->data[2] * Z_ref + T_cw->data[3];
                    double Y_cur = T_cw->data[4] * X_ref + T_cw->data[5] * Y_ref
                                 + T_cw->data[6] * Z_ref + T_cw->data[7];
                    double Z_cur = T_cw->data[8] * X_ref + T_cw->data[9] * Y_ref
                                 + T_cw->data[10] * Z_ref + T_cw->data[11];

                    if (Z_cur <= 0.1) continue;

                    double u_cur = K_level.fx * (X_cur / Z_cur) + K_level.cx;
                    double v_cur = K_level.fy * (Y_cur / Z_cur) + K_level.cy;

                    double gx = 0.0, gy = 0.0;
                    double I_cur = bilinear_interpolate_with_grad(
                        cur_pyr[level].data, lw, lh, u_cur, v_cur, &gx, &gy);
                    if (I_cur < 0.0) continue;

                    if (grad_thresh > 0.0 && (gx*gx + gy*gy) < grad_thresh)
                        continue;

                    double res = I_cur - ((*pa) * I_ref + (*pb));
                    double w_huber = config->use_robust_loss ?
                        compute_huber_weight(res, config->huber_delta) : 1.0;

                    valid_points++;

                    double inv_Z    = 1.0 / Z_cur;
                    double inv_Z_sq = inv_Z * inv_Z;

                    double du_dX = K_level.fx * inv_Z;
                    double du_dZ = -K_level.fx * X_cur * inv_Z_sq;
                    double dv_dY = K_level.fy * inv_Z;
                    double dv_dZ = -K_level.fy * Y_cur * inv_Z_sq;

                    double dI_dX = gx * du_dX;
                    double dI_dY = gy * dv_dY;
                    double dI_dZ = gx * du_dZ + gy * dv_dZ;

                    double J[8];
                    J[0] = dI_dX;
                    J[1] = dI_dY;
                    J[2] = dI_dZ;
                    J[3] = -dI_dY * Z_cur + dI_dZ * Y_cur;
                    J[4] =  dI_dX * Z_cur - dI_dZ * X_cur;
                    J[5] = -dI_dX * Y_cur + dI_dY * X_cur;
                    J[6] = -I_ref;
                    J[7] = -1.0;

                    for (int i = 0; i < 8; ++i) {
                        for (int j = 0; j < 8; ++j)
                            H[i * 8 + j] += w_huber * J[i] * J[j];
                        b[i] += -w_huber * J[i] * res;
                    }
                }
            }

            if (valid_points < 10) break;

            double H_lm[64];
            memcpy(H_lm, H, sizeof(double) * 64);
            for (int i = 0; i < 8; ++i) {
                double diag = H[i * 8 + i];
                if (diag < 1e-6) diag = 1e-6;
                H_lm[i * 8 + i] += lambda * diag;
            }

            double scale_factor[8];
            for (int i = 0; i < 8; ++i)
                scale_factor[i] = 1.0 / sqrt(H_lm[i * 8 + i] + 1e-12);

            double H_scaled[64], b_scaled[8];
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j)
                    H_scaled[i * 8 + j] = H_lm[i * 8 + j] * scale_factor[i] * scale_factor[j];
                b_scaled[i] = b[i] * scale_factor[i];
            }

            double delta_scaled[8];
            if (!solve_linear_NxN(8, H_scaled, b_scaled, delta_scaled)) {
                lambda *= 10.0;
                continue;
            }

            double delta[8];
            for (int i = 0; i < 8; ++i)
                delta[i] = delta_scaled[i] * scale_factor[i];

            lepaute_tangent_t dxi_tangent;
            memcpy(dxi_tangent.data, delta, sizeof(double) * 6);

            lepaute_se3_t dT, T_updated;
            lepaute_se3_exp(&dxi_tangent, &dT);
            lepaute_se3_mul(&dT, T_cw, &T_updated);

            double new_a = fmax(0.01, *pa + delta[6]);
            double new_b = *pb + delta[7];

            double new_cost = compute_cost(
                &ref_pyr[level], &cur_pyr[level], ref_inv_depth,
                width, height, &K_level, &T_updated,
                new_a, new_b, scale,
                config->huber_delta, config->use_robust_loss,
                step, grad_thresh);

            if (new_cost < current_cost) {
                *T_cw = T_updated;
                *pa = new_a;
                *pb = new_b;
                current_cost = new_cost;
                lambda = (lambda > 1e-6) ? (lambda / 10.0) : 1e-6;

                double norm_sq = 0.0;
                for (int i = 0; i < 6; ++i)
                    norm_sq += delta[i] * delta[i];
                if (sqrt(norm_sq) < 1e-5) break;
            } else {
                lambda *= 10.0;
            }
        }
    }

    lepaute_free_pyramid(ref_pyr, levels);
    lepaute_free_pyramid(cur_pyr, levels);

    return current_cost;
}