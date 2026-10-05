/* yolo26n_seg_mem.c -- Acesso as saidas. A arena fica em forward.c.
 *
 * O forward.c e a unica fonte de verdade da arena (g_arena) e expoe
 * yolo_arena() para quem quiser ler as ativacoes. Aqui so mapeamos
 * os tensores de saida em cima dessa arena.
 */
#include "yolo26n_seg_mem.h"
#include "yolo26n_seg.h"   /* for yolo_arena() e YOLO_TENSOR */

const float *yolo_output0(void) {
    return yolo_arena() + YOLO_TENSOR[T_output0].off;
}

const float *yolo_output1(void) {
    return yolo_arena() + YOLO_TENSOR[T_output1].off;
}
