#include "lepaute.h"
#include <string.h>

double lepaute_refine(
    const uint8_t*       ref_gray,
    const uint8_t*       cur_gray,
    int                  width,
    int                  height,
    const lepaute_cfg_t* cfg,
    lepaute_se3_t*       T_inout)
{
    if (!ref_gray || !cur_gray || !cfg || !T_inout || width <= 0 || height <= 0)
        return 1e9;

    double photo_a = 1.0;
    double photo_b = 0.0;

    double cost = lepaute_lm_refine_pose_pyramid(
        ref_gray,
        cur_gray,
        NULL,
        width,
        height,
        &cfg->K,
        &cfg->gn,
        T_inout,
        &photo_a,
        &photo_b
    );

    return cost;
}

void lepaute_se3_from_xi(const double xi[6], lepaute_se3_t* T)
{
    if (!xi || !T) return;
    lepaute_tangent_t tang;
    memcpy(tang.data, xi, sizeof(double) * 6);
    lepaute_se3_exp(&tang, T);
}

void lepaute_xi_from_se3(const lepaute_se3_t* T, double xi[6])
{
    if (!T || !xi) return;
    lepaute_tangent_t tang;
    lepaute_se3_log(T, &tang);
    memcpy(xi, tang.data, sizeof(double) * 6);
}