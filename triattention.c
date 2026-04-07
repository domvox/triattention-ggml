/*
 * triattention.c — TriAttention scoring implementation
 *
 * Implements TRIA binary loader and per-KV-head scoring with GQA aggregation.
 * Standalone — compiles without ggml for testing, integrates via ggml_map_custom1.
 */

#include "triattention.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* TRIA binary loader                                                  */
/* ------------------------------------------------------------------ */

struct tria_stats * tria_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }

    uint32_t magic, version;
    fread(&magic, 4, 1, fp);
    fread(&version, 4, 1, fp);
    if (magic != TRIA_MAGIC || (version != 1 && version != 2)) {
        fprintf(stderr, "tria_load: bad magic/version: %x v%u\n", magic, version);
        fclose(fp); return NULL;
    }

    struct tria_stats *s = calloc(1, sizeof(*s));
    fread(&s->num_layers,   4, 1, fp);
    fread(&s->num_heads,    4, 1, fp);
    fread(&s->num_kv_heads, 4, 1, fp);
    fread(&s->head_dim,     4, 1, fp);
    fread(&s->freq_count,   4, 1, fp);
    fread(&s->rope_theta,   4, 1, fp);
    fread(&s->attn_scale,   4, 1, fp);

    /* Skip reserved bytes to reach end of 64-byte header */
    fseek(fp, TRIA_HEADER_SIZE, SEEK_SET);

    uint32_t nl = s->num_layers, nh = s->num_heads, fc = s->freq_count;

    /* Per-layer budget scales (v2) */
    s->layer_budget_scales = malloc(nl * sizeof(float));
    if (version >= 2) {
        fread(s->layer_budget_scales, 4, nl, fp);
    } else {
        for (uint32_t i = 0; i < nl; i++) s->layer_budget_scales[i] = 1.0f;
    }

    /* Precompute omega: theta^(-2i/head_dim) */
    s->omega = malloc(fc * sizeof(float));
    for (uint32_t i = 0; i < fc; i++) {
        s->omega[i] = powf(s->rope_theta, -2.0f * i / s->head_dim);
    }

    /* Per-head stats */
    uint32_t total = nl * nh;
    s->heads = calloc(total, sizeof(struct tria_head_stats));
    for (uint32_t h = 0; h < total; h++) {
        struct tria_head_stats *hs = &s->heads[h];
        hs->q_mean_real = malloc(fc * sizeof(float));
        hs->q_mean_imag = malloc(fc * sizeof(float));
        hs->q_abs_mean  = malloc(fc * sizeof(float));
        hs->qma         = malloc(fc * sizeof(float));

        fread(hs->q_mean_real, 4, fc, fp);
        fread(hs->q_mean_imag, 4, fc, fp);
        fread(hs->q_abs_mean,  4, fc, fp);
        fseek(fp, fc * 4, SEEK_CUR);  /* skip mrl */

        /* Precompute |E[q_f]| */
        for (uint32_t f = 0; f < fc; f++) {
            float r = hs->q_mean_real[f], i = hs->q_mean_imag[f];
            hs->qma[f] = sqrtf(r*r + i*i);
        }
    }

    fclose(fp);
    return s;
}

void tria_free(struct tria_stats *s) {
    if (!s) return;
    uint32_t total = s->num_layers * s->num_heads;
    for (uint32_t h = 0; h < total; h++) {
        free(s->heads[h].q_mean_real);
        free(s->heads[h].q_mean_imag);
        free(s->heads[h].q_abs_mean);
        free(s->heads[h].qma);
    }
    free(s->heads);
    free(s->layer_budget_scales);
    free(s->omega);
    free(s);
}

/* ------------------------------------------------------------------ */
/* Single-head scoring (eq 6-11)                                       */
/* ------------------------------------------------------------------ */

static void score_keys_single_head(
    const struct tria_head_stats *hs,
    const float *omega,
    const float *k_real,    /* [seq_len * fc] */
    const float *k_imag,    /* [seq_len * fc] */
    const int   *key_pos,
    int          cur_pos,
    int          fc,
    int          seq_len,
    float       *out        /* [seq_len] */
) {
    static const float offsets[TRIA_N_OFFSETS] = {
        1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536
    };

    /*
     * Optimized: avoid atan2/cos(phase) in inner loop.
     *
     * amp * cos(delta*omega + phi) = Re[ (q_mean * conj(k)) * e^(j*delta*omega) ]
     *                              = rel_real * cos(delta*omega) - rel_imag * sin(delta*omega)
     *
     * where rel = q_mean * conj(k):
     *   rel_real = qr*kr + qi*ki
     *   rel_imag = qi*kr - qr*ki
     *
     * cos/sin(delta*omega) depend only on position, not on head stats,
     * so we precompute them per offset.
     */

    for (int s = 0; s < seq_len; s++) {
        const float *kr = k_real + s * fc;
        const float *ki = k_imag + s * fc;
        float base_delta = (float)(cur_pos - key_pos[s]);

        /* Precompute rel and ka per freq band */
        float rel_r[fc], rel_i[fc], ka[fc];
        float extra = 0.0f;

        for (int f = 0; f < fc; f++) {
            ka[f] = sqrtf(kr[f]*kr[f] + ki[f]*ki[f]);
            /* rel = q_mean * conj(k) — includes both amplitude and phase */
            rel_r[f] = hs->q_mean_real[f]*kr[f] + hs->q_mean_imag[f]*ki[f];
            rel_i[f] = hs->q_mean_imag[f]*kr[f] - hs->q_mean_real[f]*ki[f];

            float residual = hs->q_abs_mean[f] - hs->qma[f];
            if (residual > 0.0f) extra += residual * ka[f];
        }

        /* Accumulate trig score over geometric offsets */
        float trig_sum = 0.0f;
        for (int o = 0; o < TRIA_N_OFFSETS; o++) {
            float delta = base_delta + offsets[o];
            float trig = 0.0f;
            for (int f = 0; f < fc; f++) {
                /* Re[ rel * e^(j*delta*omega) ] — no atan2 needed */
                float angle = delta * omega[f];
                float cos_d = cosf(angle);
                float sin_d = sinf(angle);
                trig += rel_r[f] * cos_d - rel_i[f] * sin_d;
            }
            trig_sum += trig;
        }

        out[s] = trig_sum / (float)TRIA_N_OFFSETS + extra;
    }
}

/* ------------------------------------------------------------------ */
/* GQA-aggregated scoring (eq 12-13)                                   */
/* ------------------------------------------------------------------ */

void tria_score_kv_head(
    const struct tria_stats *stats,
    const float *k_pre_real,
    const float *k_pre_imag,
    const int   *key_pos,
    int          cur_pos,
    int          seq_len,
    int          layer_idx,
    int          kv_head_idx,
    float       *out_scores
) {
    int nh = stats->num_heads;
    int nkv = stats->num_kv_heads;
    int fc = stats->freq_count;
    int gqa = nh / nkv;

    float *tmp = malloc(seq_len * sizeof(float));
    /* z-normalized scores per query head, then max-aggregate */
    bool first = true;

    for (int g = 0; g < gqa; g++) {
        int ah = kv_head_idx * gqa + g;
        const struct tria_head_stats *hs = &stats->heads[layer_idx * nh + ah];

        score_keys_single_head(hs, stats->omega, k_pre_real, k_pre_imag,
                               key_pos, cur_pos, fc, seq_len, tmp);

        /* Z-normalize */
        float mean = 0.0f;
        for (int s = 0; s < seq_len; s++) mean += tmp[s];
        mean /= seq_len;

        float var = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            float d = tmp[s] - mean;
            var += d * d;
        }
        float std = sqrtf(var / seq_len);
        if (std < 1e-6f) std = 1e-6f;

        for (int s = 0; s < seq_len; s++) {
            float z = (tmp[s] - mean) / std;
            if (first) {
                out_scores[s] = z;
            } else if (z > out_scores[s]) {
                out_scores[s] = z;
            }
        }
        first = false;
    }

    free(tmp);
}
