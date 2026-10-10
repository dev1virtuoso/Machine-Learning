#include <gtest/gtest.h>
#include "se3.h"
#include "optimization.h"
#include <cmath>
#include <vector>

TEST(SE3Test, IdentityCheck) {
    lepaute_se3_t T;
    lepaute_se3_identity(&T);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(T.data[i * 4 + j], expected);
        }
    }
}

TEST(SE3Test, MulAndInverseRoundtrip) {
    lepaute_tangent_t xi;
    xi.data[0] = 0.3; xi.data[1] = -0.1; xi.data[2] = 0.4;
    xi.data[3] = 0.1; xi.data[4] = 0.2;  xi.data[5] = -0.3;

    lepaute_se3_t T, T_inv, T_res;
    lepaute_se3_exp(&xi, &T);
    lepaute_se3_inv(&T, &T_inv);
    lepaute_se3_mul(&T, &T_inv, &T_res);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(T_res.data[i * 4 + j], expected, 1e-8);
        }
    }
}

TEST(SE3Test, ExpLogRoundtrip) {
    lepaute_tangent_t xi_in;
    xi_in.data[0] = 0.1;
    xi_in.data[1] = -0.2;
    xi_in.data[2] = 0.5;
    xi_in.data[3] = 0.05;
    xi_in.data[4] = -0.03;
    xi_in.data[5] = 0.02;

    lepaute_se3_t T;
    lepaute_se3_exp(&xi_in, &T);

    lepaute_tangent_t xi_out;
    lepaute_se3_log(&T, &xi_out);

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(xi_in.data[i], xi_out.data[i], 1e-6);
    }
}

TEST(SE3Test, LogSingularityNearPi) {
    lepaute_tangent_t xi_in;
    xi_in.data[0] = 0.0; xi_in.data[1] = 0.0; xi_in.data[2] = 0.0;
    xi_in.data[3] = 0.0; xi_in.data[4] = 0.0; xi_in.data[5] = M_PI - 1e-4;

    lepaute_se3_t T;
    lepaute_se3_exp(&xi_in, &T);

    lepaute_tangent_t xi_out;
    lepaute_se3_log(&T, &xi_out);

    EXPECT_NEAR(xi_in.data[5], xi_out.data[5], 1e-3);
}

TEST(SE3Test, LeftJacobianInverseTest) {
    double phi[3] = {0.1, -0.2, 0.15};
    double Jl[9], Jl_inv[9];

    lepaute_so3_left_jacobian(phi, Jl);
    lepaute_so3_left_jacobian_inv(phi, Jl_inv);

    double res[9] = {0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                res[i * 3 + j] += Jl[i * 3 + k] * Jl_inv[k * 3 + j];
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(res[i * 3 + j], expected, 1e-6);
        }
    }
}

TEST(OptimizationTest, DirectLMAlignmentPureGeometry) {
    const int width = 128;
    const int height = 128;
    std::vector<uint8_t> ref_img(width * height, 100);
    std::vector<uint8_t> cur_img(width * height, 100);

    for (int y = 10; y < 118; ++y) {
    for (int x = 10; x < 118; ++x) {
        double dx = x - 64.0;
        double dy = y - 64.0;
        double val = 100.0 + 130.0 * exp(-(dx*dx + dy*dy) / (2.0 * 15.0 * 15.0));
        ref_img[y * width + x] = (uint8_t)val;
    }
}

    for (int y = 30; y < 90; ++y) {
        for (int x = 34; x < 94; ++x) {
            cur_img[y * width + x] = 230;
        }
    }

    lepaute_intrinsics_t K = { 100.0f, 100.0f, 64.0f, 64.0f };
    lepaute_se3_t T_est;
    lepaute_se3_identity(&T_est);

    lepaute_gn_config_t config = {
        .num_levels = 3,
        .max_iters_per_level = 20,
        .huber_delta = 5.0,
        .use_robust_loss = true,
        .initial_lm_lambda = 1e-3,
        .min_grad_thresh = 10.0
    };

    double residual = lepaute_gn_refine_pose_pyramid(
        ref_img.data(),
        cur_img.data(),
        width,
        height,
        &K,
        &config,
        &T_est
    );

    EXPECT_LT(residual, 1.0);
}

TEST(OptimizationTest, DirectLMAlignmentWithAffineLighting) {
    const int width  = 128;
    const int height = 128;
    const uint8_t bg = 80;
    std::vector<uint8_t> ref_img(width * height, bg);
    std::vector<uint8_t> cur_img(width * height, bg);

    const double cx = 64.0, cy = 64.0, sigma = 10.0;
    for (int y = 25; y < 103; ++y) {
        for (int x = 25; x < 103; ++x) {
            double dx  = x - cx;
            double dy  = y - cy;
            double val = bg + 100.0 * exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));
            ref_img[y * width + x] = static_cast<uint8_t>(val);
        }
    }

    const double a_gt = 1.1, b_gt = 10.0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int src_x = x - 3;
            double I_ref;
            if (src_x >= 0 && src_x < width)
                I_ref = static_cast<double>(ref_img[y * width + src_x]);
            else
                I_ref = bg;
            double I_cur = a_gt * I_ref + b_gt;
            if (I_cur > 255.0) I_cur = 255.0;
            cur_img[y * width + x] = static_cast<uint8_t>(I_cur);
        }
    }

    lepaute_intrinsics_t K = {100.0f, 100.0f, 64.0f, 64.0f};
    lepaute_se3_t T_est;
    lepaute_se3_identity(&T_est);

    lepaute_gn_config_t config = {
        .num_levels          = 3,
        .max_iters_per_level = 25,
        .huber_delta         = 8.0,
        .use_robust_loss     = true,
        .initial_lm_lambda   = 1e-3,
        .min_grad_thresh     = 4.0
    };

    double photo_a = 1.0, photo_b = 0.0;
    double residual = lepaute_lm_refine_pose_pyramid(
        ref_img.data(), cur_img.data(), /*ref_inv_depth=*/nullptr,
        width, height, &K, &config, &T_est,
        &photo_a, &photo_b);

    EXPECT_NEAR(photo_a, a_gt, 0.05);
    EXPECT_NEAR(photo_b, b_gt, 1.0);
    EXPECT_LT(residual, 1.0);
}