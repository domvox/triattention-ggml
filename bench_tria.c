/*
 * bench_tria.c — Benchmark TriAttention scoring throughput
 *
 * Build: gcc -O2 -o bench_tria bench_tria.c triattention.c -lm
 */

#include "triattention.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <stats.bin>\n", argv[0]);
        return 1;
    }

    struct tria_stats *s = tria_load(argv[1]);
    if (!s) return 1;

    int fc = s->freq_count, nkv = s->num_kv_heads, nl = s->num_layers;

    int test_lens[] = {256, 512, 1024, 2048, 4096};
    int n_tests = 5;

    printf("TriAttention scoring benchmark (CPU, single-thread)\n");
    printf("  %u layers × %u kv_heads, fc=%u\n\n", nl, nkv, fc);
    printf("%8s %10s %8s\n", "seq_len", "time_ms", "us/head");
    printf("--------------------------------\n");

    for (int t = 0; t < n_tests; t++) {
        int seq_len = test_lens[t];

        float *k_real = calloc(seq_len * fc, sizeof(float));
        float *k_imag = calloc(seq_len * fc, sizeof(float));
        int *key_pos = calloc(seq_len, sizeof(int));
        float *scores = calloc(seq_len, sizeof(float));

        srand(42);
        for (int i = 0; i < seq_len * fc; i++) {
            k_real[i] = (float)rand() / RAND_MAX - 0.5f;
            k_imag[i] = (float)rand() / RAND_MAX - 0.5f;
        }
        for (int i = 0; i < seq_len; i++) key_pos[i] = i;

        /* Warmup */
        tria_score_kv_head(s, k_real, k_imag, key_pos, seq_len, seq_len, 0, 0, scores);

        /* Benchmark: score ALL layer×kv_head pairs (full pruning pass) */
        double best = 1e9;
        for (int iter = 0; iter < 3; iter++) {
            double t0 = now_ms();
            for (int li = 0; li < nl; li++) {
                for (int kvi = 0; kvi < nkv; kvi++) {
                    tria_score_kv_head(s, k_real, k_imag, key_pos,
                                       seq_len, seq_len, li, kvi, scores);
                }
            }
            double elapsed = now_ms() - t0;
            if (elapsed < best) best = elapsed;
        }

        int total_heads = nl * nkv;
        printf("%8d %10.1f %8.1f\n", seq_len, best, best * 1000.0 / total_heads);

        free(k_real); free(k_imag); free(key_pos); free(scores);
    }

    tria_free(s);
    return 0;
}
