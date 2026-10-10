#ifndef LEPAUTE_SE3_H
#define LEPAUTE_SE3_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

typedef struct { double data[16]; } lepaute_se3_t;
typedef struct { double data[6];  } lepaute_tangent_t;
typedef struct { float fx, fy, cx, cy; } lepaute_intrinsics_t;

void lepaute_se3_identity(lepaute_se3_t* T);
void lepaute_skew_symmetric(const double v[3], double K[9]);
void lepaute_se3_mul(const lepaute_se3_t* A, const lepaute_se3_t* B, lepaute_se3_t* Out);
void lepaute_se3_inv(const lepaute_se3_t* In, lepaute_se3_t* Out);

void lepaute_se3_exp(const lepaute_tangent_t* xi, lepaute_se3_t* T);
void lepaute_se3_log(const lepaute_se3_t* T, lepaute_tangent_t* xi);

void lepaute_se3_adjoint(const lepaute_se3_t* T, double Ad[36]);
void lepaute_so3_left_jacobian(const double phi[3], double Jl[9]);
void lepaute_so3_left_jacobian_inv(const double phi[3], double Jl_inv[9]);

#ifdef __cplusplus
}
#endif

#endif