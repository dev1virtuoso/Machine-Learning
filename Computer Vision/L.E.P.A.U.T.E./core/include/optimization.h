#ifndef LEPAUTE_OPTIMIZATION_H
#define LEPAUTE_OPTIMIZATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "se3.h"
#include <stdint.h>
#include <stdbool.h>

#ifndef LEPAUTE_MAX_PYRAMID_LEVELS
#define LEPAUTE_MAX_PYRAMID_LEVELS  8
#endif

#ifndef LEPAUTE_MAX_IMAGE_PIXELS
#define LEPAUTE_MAX_IMAGE_PIXELS    (320 * 240)
#endif

typedef struct {
    int num_levels;
    int max_iters_per_level;
    double huber_delta;
    int use_robust_loss; 
    double initial_lm_lambda;
    double min_grad_thresh;
} lepaute_gn_config_t;

typedef struct {
    uint8_t* data;
    int width;
    int height;
} lepaute_image_t;

int lepaute_create_pyramid(const uint8_t* src, int w, int h,
                           int levels, lepaute_image_t* pyramid);

void lepaute_free_pyramid(lepaute_image_t* pyramid, int levels);

double lepaute_lm_refine_pose_pyramid(
    const uint8_t* ref_img,
    const uint8_t* cur_img,
    const float* ref_inv_depth,
    int width,
    int height,
    const lepaute_intrinsics_t* intrinsics,
    const lepaute_gn_config_t* config,
    lepaute_se3_t* T_inout,
    double* photo_a,
    double* photo_b
);

static inline double lepaute_gn_refine_pose_pyramid(
    const uint8_t* ref_img,
    const uint8_t* cur_img,
    int width,
    int height,
    const lepaute_intrinsics_t* intrinsics,
    const lepaute_gn_config_t* config,
    lepaute_se3_t* T_inout
) {
    double photo_a = 1.0;
    double photo_b = 0.0;
    return lepaute_lm_refine_pose_pyramid(
        ref_img, cur_img, NULL, width, height,
        intrinsics, config, T_inout, &photo_a, &photo_b
    );
}

#ifdef __cplusplus
}
#endif

#endif