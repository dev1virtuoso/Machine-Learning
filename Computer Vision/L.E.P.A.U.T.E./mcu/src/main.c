#include "lepaute.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#define IMG_W 160
#define IMG_H 120

static void make_test_image(uint8_t* img, int w, int h, int shift_x)
{
    memset(img, 40, (size_t)w * h);

    int cx = w / 2 + shift_x;
    int cy = h / 2;
    int size = 16;

    for (int y = cy - size; y < cy + size; ++y) {
        for (int x = cx - size; x < cx + size; ++x) {
            if (x >= 0 && x < w && y >= 0 && y < h)
                img[y * w + x] = 200;
        }
    }

    for (int i = 0; i < w; i += 12)
        for (int y = 0; y < h; ++y)
            img[y * w + i] = 90;
}

int main(void)
{
    static uint8_t ref_img[IMG_W * IMG_H];
    static uint8_t cur_img[IMG_W * IMG_H];

    make_test_image(ref_img, IMG_W, IMG_H, 0);
    make_test_image(cur_img, IMG_W, IMG_H, 5);

    lepaute_cfg_t cfg;
    lepaute_default_cfg(&cfg);

    cfg.K.fx = 100.0f;
    cfg.K.fy = 100.0f;
    cfg.K.cx = IMG_W / 2.0f;
    cfg.K.cy = IMG_H / 2.0f;
    cfg.scale_prior = 1.2f;

    cfg.gn.num_levels          = 2;
    cfg.gn.max_iters_per_level = 5;
    cfg.gn.min_grad_thresh     = 4.0;

    lepaute_se3_t T;
    lepaute_se3_identity(&T);

    printf("LEPAUTE MCU Demo\n");
    printf("Resolution : %d x %d\n", IMG_W, IMG_H);
    printf("Running photometric LM...\n");

    double cost = lepaute_refine(ref_img, cur_img, IMG_W, IMG_H, &cfg, &T);

    double xi[6];
    lepaute_xi_from_se3(&T, xi);

    printf("Final cost : %.4f\n", cost);
    printf("xi = [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
           xi[0], xi[1], xi[2], xi[3], xi[4], xi[5]);

    if (cost < 80.0)
        printf("Quality: GOOD\n");
    else if (cost < 250.0)
        printf("Quality: ACCEPTABLE\n");
    else
        printf("Quality: POOR\n");

    return 0;
}