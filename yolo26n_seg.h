/* yolo26n_seg.h -- tipos, prototipos e helpers. */
#ifndef YOLO26N_SEG_H
#define YOLO26N_SEG_H
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "yolo26n_seg_layout.h"
#include "yolo26n_seg_mem.h"

typedef struct { float *data; int c, h, w; } Tensor;

static inline int tsz(const Tensor *t){ return t->c * t->h * t->w; }
static inline int thw(const Tensor *t){ return t->h * t->w; }

static inline Tensor tensor_at(float *arena, TensorId id){
    Tensor t;
    t.data = arena + YOLO_TENSOR[id].off;
    t.c    = YOLO_TENSOR[id].c;
    t.h    = YOLO_TENSOR[id].h;
    t.w    = YOLO_TENSOR[id].w;
    return t;
}
#define TV(arena, id) ((Tensor){ \
    .data = (arena) + YOLO_TENSOR[id].off, \
    .c    = YOLO_TENSOR[id].c, \
    .h    = YOLO_TENSOR[id].h, \
    .w    = YOLO_TENSOR[id].w })

/* ---- primitivas (kernels.c) ---- */
void conv2d          (const Tensor *in, Tensor *out, ConvId id, const float *blob);
void convT2x2s2      (const Tensor *in, Tensor *out, ConvId id, const float *blob);
void maxpool5s1p2    (const Tensor *in, Tensor *out);
void resize_nearest  (const Tensor *in, Tensor *out, int scale);
void add_tensor      (const Tensor *a, const Tensor *b, Tensor *out);
void add_inplace     (Tensor *a, const Tensor *b);
void concat_ch       (const Tensor *const *ins, int n, Tensor *out);
void concat_w(const Tensor *const *ins, int n, Tensor *out);
void silu_tensor     (Tensor *t);
void sigmoid_tensor  (Tensor *t);
void mul_scalar      (Tensor *t, float s);
void softmax_last    (Tensor *t);           /* rows = c*h, cols = w */
void matmul_batched  (const float *A, const float *B, float *C, int batch, int M, int K, int N);
void matmul_nt       (const float *A, const float *B, float *C, int batch, int M, int K, int N);
void topk_desc       (const float *x, int n, int k, float *val, int *idx);

/* ---- blocos compostos (blocks.c) ---- */
typedef struct {
    ConvId cv1, bcv1, bcv2, cv2;
    int t_in, t_y, t_bcv1, t_bcv2, t_add, t_cat, t_out;
} C3k2A_Cfg;
void c3k2_a(float *A, const float *blob, const C3k2A_Cfg *g);

typedef struct {
    ConvId cv1, mcv1, mcv2, mcv3, cv2;
    ConvId bot[2][2];             /* [n_bot][{cv1,cv2}] */
    int    n_bot;
    int    t_in, t_y, t_u, t_v;
    int    t_m_bot1, t_cv2_bot1, t_bot1_out;   /* novo: t_cv2_bot1 */
    int    t_m_bot2, t_cv2_bot2, t_bot2_out;   /* novo: t_cv2_bot2 */
    int    t_cat2, t_m, t_cat3, t_out;
} C3k2B_Cfg;

void c3k2_b(float *A, const float *blob, const C3k2B_Cfg *g);

typedef struct {
    ConvId cv1, cv2;
    int t_in, t_cv1, t_p1, t_p2, t_p3, t_cat, t_cv2, t_out;
} SPPF_Cfg;
void sppf(float *A, const float *blob, const SPPF_Cfg *g);


/* ---- attention (blocks.c) ---- */
void psa_attention(float *A, const float *blob,
                   ConvId qkv, ConvId pe, ConvId proj,
                   int t_in, int t_qkv, int t_v_reshaped,
                   int t_pe, int t_o_sum, int t_proj_out);

/* ---- cabeça (head.c) ---- */
typedef struct {
    ConvId box_cv0, box_cv1, box_cv2;
    ConvId cls_cv0, cls_cv1, cls_cv2, cls_cv3, cls_cv4;
    ConvId coef_cv0, coef_cv1, coef_cv2;
    int t_in;
    int t_box0, t_box1, t_box_out;      /* [4, H, W] */
    int t_cls0, t_cls1, t_cls2, t_cls3, t_cls4;
    int t_coef0, t_coef1, t_coef_out;
    int t_box_flat, t_cls_flat, t_coef_flat;   /* [C, HW] */
    int H, W;
} HeadBranch_Cfg;

void head_branch(float *A, const float *blob, const HeadBranch_Cfg *g);
void head_decode_topk(float *A, int t_box_2100, int t_score_2100, int t_coef_2100,
                      int t_output0);
void proto26(float *A, const float *blob,
             int t_p3, int t_p4, int t_p5,
             int t_p4_ref, int t_p5_ref,
             int t_p4_up, int t_p5_up,
             int t_sum_p3r4, int t_fuse, int t_cv1, int t_convT,
             int t_cv2, int t_output1);

/* ---- top-level ---- */
float *yolo_arena(void);
const float *yolo_weights(void);
void yolo_forward(const float *input_nchw);
const float *yolo_output0(void);
const float *yolo_output1(void);

#endif
