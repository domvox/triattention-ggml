#include "triattention.h"
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <stats.bin>\n", argv[0]); return 1; }
    struct tria_stats *s = tria_load(argv[1]);
    if (!s) return 1;
    printf("Loaded: %u layers, %u heads, %u kv_heads, hd=%u, fc=%u\n",
           s->num_layers, s->num_heads, s->num_kv_heads, s->head_dim, s->freq_count);
    printf("Budget scales: [%.3f .. %.3f]\n",
           s->layer_budget_scales[0], s->layer_budget_scales[s->num_layers-1]);
    printf("Omega[0]=%.6f Omega[63]=%.10f\n", s->omega[0], s->omega[s->freq_count-1]);
    tria_free(s);
    printf("OK\n");
    return 0;
}
