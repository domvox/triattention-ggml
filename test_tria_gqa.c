#include "triattention.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <stats.bin> <test.bin> <layer> <kv_head>\n", argv[0]);
        return 1;
    }
    struct tria_stats *s = tria_load(argv[1]);
    if (!s) return 1;

    int layer = atoi(argv[3]), kv_head = atoi(argv[4]);

    FILE *fp = fopen(argv[2], "rb");
    int32_t fc, seq_len, cur_pos, gqa;
    fread(&fc, 4, 1, fp); fread(&seq_len, 4, 1, fp);
    fread(&cur_pos, 4, 1, fp); fread(&gqa, 4, 1, fp);

    int *key_pos = malloc(seq_len * 4);
    float *k_real = malloc(seq_len * fc * 4);
    float *k_imag = malloc(seq_len * fc * 4);
    float *expected = malloc(seq_len * 4);
    float *computed = malloc(seq_len * 4);

    fread(key_pos, 4, seq_len, fp);
    fread(k_real, 4, seq_len * fc, fp);
    fread(k_imag, 4, seq_len * fc, fp);
    fread(expected, 4, seq_len, fp);
    fclose(fp);

    tria_score_kv_head(s, k_real, k_imag, key_pos, cur_pos, seq_len, layer, kv_head, computed);

    float max_err = 0;
    for (int i = 0; i < seq_len; i++) {
        float err = fabsf(computed[i] - expected[i]);
        float denom = fabsf(expected[i]) > 1e-6f ? fabsf(expected[i]) : 1.0f;
        float rel = err / denom;
        if (rel > max_err) max_err = rel;
    }
    printf("GQA L%d KV%d: max_rel_err=%.6e %s\n", layer, kv_head, max_err, max_err < 1e-2 ? "PASS" : "FAIL");

    tria_free(s); free(key_pos); free(k_real); free(k_imag); free(expected); free(computed);
    return max_err < 1e-2 ? 0 : 1;
}
