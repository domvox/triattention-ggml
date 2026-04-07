/*
 * TriAttention scoring — standalone C reference (eq 6-13)
 *
 * Computes per-key importance scores identical to Python score_keys().
 * No ggml dependency — pure C with math.h for validation.
 * Will be ported to ggml compute graph for llama.cpp integration.
 *
 * Build: gcc -O2 -o triattention_score_ref triattention_score_ref.c -lm
 * Test:  reads binary test vectors from stdin, writes scores to stdout
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_OFFSETS 17  /* geometric: 1, 2, 4, ..., 2^16 */

/*
 * score_keys: compute TriAttention importance for each key
 *
 * k_real[seq_len][fc]  — pre-RoPE key, real part (first half of head_dim)
 * k_imag[seq_len][fc]  — pre-RoPE key, imag part (second half of head_dim)
 * q_mean_real[fc]      — calibrated Q center, real
 * q_mean_imag[fc]      — calibrated Q center, imag
 * q_abs_mean[fc]       — E[||q_f||] per frequency band
 * omega[fc]            — RoPE frequencies: theta^(-2i/head_dim)
 * key_pos[seq_len]     — absolute position of each key
 * cur_pos              — current query position (trigger point)
 * fc                   — frequency count (head_dim / 2)
 * seq_len              — number of keys to score
 * out_scores[seq_len]  — output scores
 */
void score_keys(
    const float *k_real,       /* [seq_len * fc] */
    const float *k_imag,       /* [seq_len * fc] */
    const float *q_mean_real,  /* [fc] */
    const float *q_mean_imag,  /* [fc] */
    const float *q_abs_mean,   /* [fc] */
    const float *omega,        /* [fc] */
    const int   *key_pos,      /* [seq_len] */
    int          cur_pos,
    int          fc,
    int          seq_len,
    float       *out_scores    /* [seq_len] */
) {
    /* Precompute per-band Q stats */
    float qma[fc];  /* |E[q_f]| */
    for (int f = 0; f < fc; f++) {
        qma[f] = sqrtf(q_mean_real[f]*q_mean_real[f] + q_mean_imag[f]*q_mean_imag[f]);
    }

    /* Precompute geometric offsets */
    float offsets[N_OFFSETS];
    for (int i = 0; i < N_OFFSETS; i++) {
        offsets[i] = (float)(1 << i);  /* 1, 2, 4, ..., 65536 */
    }

    for (int s = 0; s < seq_len; s++) {
        const float *kr = k_real + s * fc;
        const float *ki = k_imag + s * fc;

        /* Per-key: compute phi, amp, ka, extra across frequency bands */
        float extra = 0.0f;
        float phi[fc], amp[fc];

        for (int f = 0; f < fc; f++) {
            float ka_f = sqrtf(kr[f]*kr[f] + ki[f]*ki[f]);

            /* Phase: angle(q_mean * conj(k))
             * q_mean * conj(k) = (qr + j*qi)(kr - j*ki)
             *                  = (qr*kr + qi*ki) + j*(qi*kr - qr*ki) */
            float rel_real = q_mean_real[f]*kr[f] + q_mean_imag[f]*ki[f];
            float rel_imag = q_mean_imag[f]*kr[f] - q_mean_real[f]*ki[f];
            phi[f] = atan2f(rel_imag, rel_real);

            amp[f] = qma[f] * ka_f;

            /* Norm extra: max(0, E[||q||] - ||E[q]||) * ||k|| */
            float residual = q_abs_mean[f] - qma[f];
            if (residual < 0.0f) residual = 0.0f;
            extra += residual * ka_f;
        }

        /* Trig score averaged over geometric offsets */
        float base_delta = (float)(cur_pos - key_pos[s]);
        float trig_sum = 0.0f;

        for (int o = 0; o < N_OFFSETS; o++) {
            float delta = base_delta + offsets[o];
            float trig = 0.0f;
            for (int f = 0; f < fc; f++) {
                /* cos(delta * omega_f + phi_f) */
                trig += amp[f] * cosf(delta * omega[f] + phi[f]);
            }
            trig_sum += trig;
        }

        out_scores[s] = trig_sum / (float)N_OFFSETS + extra;
    }
}

/* ------------------------------------------------------------------ */
/* Test harness: read binary vectors, compute scores, compare         */
/* ------------------------------------------------------------------ */

/*
 * Binary test format (all little-endian float32/int32):
 *   fc        i32
 *   seq_len   i32
 *   cur_pos   i32
 *   q_mean_real[fc]
 *   q_mean_imag[fc]
 *   q_abs_mean[fc]
 *   omega[fc]
 *   key_pos[seq_len]   (i32)
 *   k_real[seq_len*fc]
 *   k_imag[seq_len*fc]
 *   expected_scores[seq_len]  (from Python)
 */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <test_vectors.bin>\n", argv[0]);
        fprintf(stderr, "Generate test vectors with triattention_export_test.py\n");
        return 1;
    }

    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("fopen"); return 1; }

    int32_t fc, seq_len, cur_pos;
    fread(&fc, 4, 1, fp);
    fread(&seq_len, 4, 1, fp);
    fread(&cur_pos, 4, 1, fp);

    printf("fc=%d seq_len=%d cur_pos=%d\n", fc, seq_len, cur_pos);

    float *q_mean_real = malloc(fc * sizeof(float));
    float *q_mean_imag = malloc(fc * sizeof(float));
    float *q_abs_mean  = malloc(fc * sizeof(float));
    float *omega_arr   = malloc(fc * sizeof(float));
    int   *key_pos     = malloc(seq_len * sizeof(int));
    float *k_real      = malloc(seq_len * fc * sizeof(float));
    float *k_imag      = malloc(seq_len * fc * sizeof(float));
    float *expected     = malloc(seq_len * sizeof(float));
    float *computed     = malloc(seq_len * sizeof(float));

    fread(q_mean_real, 4, fc, fp);
    fread(q_mean_imag, 4, fc, fp);
    fread(q_abs_mean,  4, fc, fp);
    fread(omega_arr,   4, fc, fp);
    fread(key_pos,     4, seq_len, fp);
    fread(k_real,      4, seq_len * fc, fp);
    fread(k_imag,      4, seq_len * fc, fp);
    fread(expected,    4, seq_len, fp);
    fclose(fp);

    score_keys(k_real, k_imag, q_mean_real, q_mean_imag, q_abs_mean,
               omega_arr, key_pos, cur_pos, fc, seq_len, computed);

    /* Compare */
    float max_err = 0.0f, sum_err = 0.0f;
    for (int s = 0; s < seq_len; s++) {
        float err = fabsf(computed[s] - expected[s]);
        float rel = (fabsf(expected[s]) > 1e-6f) ? err / fabsf(expected[s]) : err;
        if (rel > max_err) max_err = rel;
        sum_err += rel;
    }
    float avg_err = sum_err / seq_len;

    printf("Max relative error: %.6e\n", max_err);
    printf("Avg relative error: %.6e\n", avg_err);
    printf("%s\n", (max_err < 1e-3f) ? "PASS" : "FAIL");

    free(q_mean_real); free(q_mean_imag); free(q_abs_mean);
    free(omega_arr); free(key_pos); free(k_real); free(k_imag);
    free(expected); free(computed);

    return (max_err < 1e-3f) ? 0 : 1;
}
