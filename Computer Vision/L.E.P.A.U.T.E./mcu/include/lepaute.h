#ifndef LEPAUTE_H
#define LEPAUTE_H

#include "se3.h"
#include "optimization.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    lepaute_intrinsics_t K;
    lepaute_gn_config_t  gn;
    float                scale_prior;
} lepaute_cfg_t;

static inline void lepaute_default_cfg(lepaute_cfg_t* cfg)
{
    if (!cfg) return;

    cfg->K.fx = 250.0f;
    cfg->K.fy = 250.0f;
    cfg->K.cx = 160.0f;
    cfg->K.cy = 120.0f;

    cfg->gn.num_levels          = 3;
    cfg->gn.max_iters_per_level = 6;
    cfg->gn.huber_delta         = 10.0;
    cfg->gn.use_robust_loss     = 1;
    cfg->gn.initial_lm_lambda   = 1e-3;
    cfg->gn.min_grad_thresh     = 5.0;

    cfg->scale_prior = 1.0f;
}

double lepaute_refine(
    const uint8_t*       ref_gray,
    const uint8_t*       cur_gray,
    int                  width,
    int                  height,
    const lepaute_cfg_t* cfg,
    lepaute_se3_t*       T_inout
);

void lepaute_se3_from_xi(const double xi[6], lepaute_se3_t* T);
void lepaute_xi_from_se3(const lepaute_se3_t* T, double xi[6]);

#ifdef __cplusplus
}
#endif

#endif