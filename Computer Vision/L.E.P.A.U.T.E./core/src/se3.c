#include "se3.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void lepaute_se3_identity(lepaute_se3_t* T) {
    memset(T->data, 0, sizeof(double) * 16);
    T->data[0] = 1.0; T->data[5] = 1.0; T->data[10] = 1.0; T->data[15] = 1.0;
}

void lepaute_skew_symmetric(const double v[3], double K[9]) {
    K[0] =  0.0;  K[1] = -v[2]; K[2] =  v[1];
    K[3] =  v[2]; K[4] =  0.0;  K[5] = -v[0];
    K[6] = -v[1]; K[7] =  v[0]; K[8] =  0.0;
}

void lepaute_se3_mul(const lepaute_se3_t* A, const lepaute_se3_t* B, lepaute_se3_t* Out) {
    double res[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) sum += A->data[i * 4 + k] * B->data[k * 4 + j];
            res[i * 4 + j] = sum;
        }
    }
    memcpy(Out->data, res, sizeof(double) * 16);
}

void lepaute_se3_inv(const lepaute_se3_t* In, lepaute_se3_t* Out) {
    lepaute_se3_identity(Out);
    Out->data[0] = In->data[0]; Out->data[1] = In->data[4]; Out->data[2] = In->data[8];
    Out->data[4] = In->data[1]; Out->data[5] = In->data[5]; Out->data[6] = In->data[9];
    Out->data[8] = In->data[2]; Out->data[9] = In->data[6]; Out->data[10] = In->data[10];

    double tx = In->data[3], ty = In->data[7], tz = In->data[11];
    Out->data[3]  = -(Out->data[0] * tx + Out->data[1] * ty + Out->data[2] * tz);
    Out->data[7]  = -(Out->data[4] * tx + Out->data[5] * ty + Out->data[6] * tz);
    Out->data[11] = -(Out->data[8] * tx + Out->data[9] * ty + Out->data[10] * tz);
}

void lepaute_se3_exp(const lepaute_tangent_t* xi, lepaute_se3_t* T) {
    const double* rho = &xi->data[0];
    const double* phi = &xi->data[3];

    double theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    double theta = sqrt(theta_sq);

    double K[9];
    lepaute_skew_symmetric(phi, K);

    double K2[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) sum += K[i * 3 + k] * K[k * 3 + j];
            K2[i * 3 + j] = sum;
        }
    }

    double A_coef, B_coef, C_coef;
    if (theta < 1e-4) {
        A_coef = 1.0 - theta_sq / 6.0 + (theta_sq * theta_sq) / 120.0;
        B_coef = 0.5 - theta_sq / 24.0 + (theta_sq * theta_sq) / 720.0;
        C_coef = 1.0 / 6.0 - theta_sq / 120.0 + (theta_sq * theta_sq) / 5040.0;
    } else {
        A_coef = sin(theta) / theta;
        B_coef = (1.0 - cos(theta)) / theta_sq;
        C_coef = (1.0 - A_coef) / theta_sq;
    }

    lepaute_se3_identity(T);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double I_ij = (i == j) ? 1.0 : 0.0;
            T->data[i * 4 + j] = I_ij + A_coef * K[i * 3 + j] + B_coef * K2[i * 3 + j];
        }
    }

    double V[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double I_ij = (i == j) ? 1.0 : 0.0;
            V[i * 3 + j] = I_ij + B_coef * K[i * 3 + j] + C_coef * K2[i * 3 + j];
        }
    }

    T->data[3]  = V[0] * rho[0] + V[1] * rho[1] + V[2] * rho[2];
    T->data[7]  = V[3] * rho[0] + V[4] * rho[1] + V[5] * rho[2];
    T->data[11] = V[6] * rho[0] + V[7] * rho[1] + V[8] * rho[2];
}

void lepaute_se3_log(const lepaute_se3_t* T, lepaute_tangent_t* xi) {
    double R[9] = {
        T->data[0], T->data[1], T->data[2],
        T->data[4], T->data[5], T->data[6],
        T->data[8], T->data[9], T->data[10]
    };
    double t[3] = { T->data[3], T->data[7], T->data[11] };

    double trace_R = R[0] + R[4] + R[8];
    double cos_theta = (trace_R - 1.0) * 0.5;
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;

    double theta = acos(cos_theta);
    double phi[3];
    double V_inv[9];

    if (theta < 1e-4) {
        double phi_raw[3] = { R[7] - R[5], R[2] - R[6], R[3] - R[1] };
        phi[0] = 0.5 * phi_raw[0];
        phi[1] = 0.5 * phi_raw[1];
        phi[2] = 0.5 * phi_raw[2];

        double K[9];
        lepaute_skew_symmetric(phi, K);
        double K2[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) sum += K[i * 3 + k] * K[k * 3 + j];
                K2[i * 3 + j] = sum;
            }
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double I_ij = (i == j) ? 1.0 : 0.0;
                V_inv[i * 3 + j] = I_ij - 0.5 * K[i * 3 + j] + (1.0 / 12.0) * K2[i * 3 + j];
            }
        }
    } else if (theta >= M_PI - 1e-3) {
        double u[3];
        if (R[0] > R[4] && R[0] > R[8]) {
            u[0] = R[0] + 1.0; u[1] = R[3] + R[1]; u[2] = R[6] + R[2];
        } else if (R[4] > R[8]) {
            u[0] = R[1] + R[3]; u[1] = R[4] + 1.0; u[2] = R[7] + R[5];
        } else {
            u[0] = R[2] + R[6]; u[1] = R[5] + R[7]; u[2] = R[8] + 1.0;
        }
        double norm_u = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
        if (norm_u > 1e-6) {
            u[0] /= norm_u; u[1] /= norm_u; u[2] /= norm_u;
        }
        phi[0] = M_PI * u[0]; phi[1] = M_PI * u[1]; phi[2] = M_PI * u[2];

        double K[9];
        lepaute_skew_symmetric(phi, K);
        double K2[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) sum += K[i * 3 + k] * K[k * 3 + j];
                K2[i * 3 + j] = sum;
            }
        }
        
        double coef = 1.0 / (M_PI * M_PI);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double I_ij = (i == j) ? 1.0 : 0.0;
                V_inv[i * 3 + j] = I_ij - 0.5 * K[i * 3 + j] + coef * K2[i * 3 + j];
            }
        }
    } else {
        double phi_raw[3] = { R[7] - R[5], R[2] - R[6], R[3] - R[1] };
        double mult = theta / (2.0 * sin(theta));
        phi[0] = mult * phi_raw[0];
        phi[1] = mult * phi_raw[1];
        phi[2] = mult * phi_raw[2];

        double K[9];
        lepaute_skew_symmetric(phi, K);
        double K2[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) sum += K[i * 3 + k] * K[k * 3 + j];
                K2[i * 3 + j] = sum;
            }
        }

        double half_th = theta * 0.5;
        double coef = (1.0 - (theta * cos(half_th)) / (2.0 * sin(half_th))) / (theta * theta);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double I_ij = (i == j) ? 1.0 : 0.0;
                V_inv[i * 3 + j] = I_ij - 0.5 * K[i * 3 + j] + coef * K2[i * 3 + j];
            }
        }
    }

    xi->data[3] = phi[0]; xi->data[4] = phi[1]; xi->data[5] = phi[2];
    xi->data[0] = V_inv[0] * t[0] + V_inv[1] * t[1] + V_inv[2] * t[2];
    xi->data[1] = V_inv[3] * t[0] + V_inv[4] * t[1] + V_inv[5] * t[2];
    xi->data[2] = V_inv[6] * t[0] + V_inv[7] * t[1] + V_inv[8] * t[2];
}

void lepaute_se3_adjoint(const lepaute_se3_t* T, double Ad[36]) {
    memset(Ad, 0, sizeof(double) * 36);
    double R[9] = { T->data[0], T->data[1], T->data[2], T->data[4], T->data[5], T->data[6], T->data[8], T->data[9], T->data[10] };
    double t[3] = { T->data[3], T->data[7], T->data[11] };
    double t_hat[9];
    lepaute_skew_symmetric(t, t_hat);

    double t_hat_R[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) sum += t_hat[i * 3 + k] * R[k * 3 + j];
            t_hat_R[i * 3 + j] = sum;
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Ad[i * 6 + j]         = R[i * 3 + j];
            Ad[i * 6 + (j + 3)]   = t_hat_R[i * 3 + j];
            Ad[(i + 3) * 6 + j]   = 0.0;
            Ad[(i + 3) * 6 + j+3] = R[i * 3 + j];
        }
    }
}

void lepaute_so3_left_jacobian(const double phi[3], double Jl[9]) {
    double theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    double theta = sqrt(theta_sq);

    double K[9];
    lepaute_skew_symmetric(phi, K);

    double K2[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) sum += K[i * 3 + k] * K[k * 3 + j];
            K2[i * 3 + j] = sum;
        }
    }

    double B_coef, C_coef;
    if (theta < 1e-4) {
        B_coef = 0.5 - theta_sq / 24.0;
        C_coef = 1.0 / 6.0 - theta_sq / 120.0;
    } else {
        B_coef = (1.0 - cos(theta)) / theta_sq;
        C_coef = (theta - sin(theta)) / (theta_sq * theta);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double I_ij = (i == j) ? 1.0 : 0.0;
            Jl[i * 3 + j] = I_ij + B_coef * K[i * 3 + j] + C_coef * K2[i * 3 + j];
        }
    }
}

void lepaute_so3_left_jacobian_inv(const double phi[3], double Jl_inv[9]) {
    double theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    double theta = sqrt(theta_sq);

    double K[9];
    lepaute_skew_symmetric(phi, K);

    double K2[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) sum += K[i * 3 + k] * K[k * 3 + j];
            K2[i * 3 + j] = sum;
        }
    }

    if (theta < 1e-4) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double I_ij = (i == j) ? 1.0 : 0.0;
                Jl_inv[i * 3 + j] = I_ij - 0.5 * K[i * 3 + j] + (1.0 / 12.0) * K2[i * 3 + j];
            }
        }
    } else {
        double half_th = theta * 0.5;
        double coef = (1.0 - (theta * cos(half_th)) / (2.0 * sin(half_th))) / theta_sq;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double I_ij = (i == j) ? 1.0 : 0.0;
                Jl_inv[i * 3 + j] = I_ij - 0.5 * K[i * 3 + j] + coef * K2[i * 3 + j];
            }
        }
    }
}