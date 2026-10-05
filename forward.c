/* forward.c -- orquestracao do forward. */
#include "yolo26n_seg.h"
#include "yolo26n_seg_mem.h"

static float g_arena[YOLO_ARENA_FLOATS] __attribute__((aligned(64)));
static float g_weights[YOLO_WEIGHTS_FLOATS] __attribute__((aligned(64)));

float *yolo_arena(void){ return g_arena; }
const float *yolo_weights(void){ return g_weights; }

#define TVi(id)  TV(g_arena, (TensorId)(id))
#define W  g_weights
#define A  g_arena


/* -------- bloco PSABlock inline (para c3k2_c e c2psa) --------
 * x: tensor de entrada [128,10,10]
 * t_qkv: temp [256,10,10]     t_v_resh: temp [128,10,10]
 * t_pe:  temp [128,10,10]     t_o:      temp [128,10,10]
 * t_ffn0:temp [256,10,10]     t_attn:   temp [128,10,10] (proj out)
 * resultado final escrito em x (in-place).
 */
static void psablock_inplace(Tensor *x,
                             ConvId c_qkv, ConvId c_pe, ConvId c_proj,
                             ConvId c_ffn0, ConvId c_ffn1,
                             int t_qkv, int t_v_resh, int t_pe,
                             int t_o, int t_ffn0, int t_attn)
{
    /* qkv conv */
    conv2d(x, &TVi(t_qkv), c_qkv, W);


    /* montar Q,K,V por head (2 heads: 32q + 32k + 64v) */
    static float Q[2][32*100], K[2][32*100], V[2][64*100];
    const int HW = 100;
    Tensor Tqkv = TVi(t_qkv);
    for (int hd = 0; hd < 2; hd++){
        const float *hh = Tqkv.data + hd*128*HW;
        for (int t = 0; t < 100; t++){
            for (int i = 0; i < 32; i++) Q[hd][t*32 + i] = hh[(0  + i)*HW + t] * ATTN_SCALE;
            for (int i = 0; i < 32; i++) K[hd][i*100 + t] = hh[(32 + i)*HW + t];
            for (int i = 0; i < 64; i++) V[hd][i*100 + t] = hh[(64 + i)*HW + t];
        }
    }
    /* v reshaped para pe */
    Tensor Tv = TVi(t_v_resh);
    for (int hd = 0; hd < 2; hd++)
        for (int c = 0; c < 64; c++)
            for (int t = 0; t < 100; t++)
                Tv.data[(hd*64 + c)*HW + t] = V[hd][c*100 + t];

    /* pe (DW 3x3 linear) */
    conv2d(&Tv, &TVi(t_pe), c_pe, W);

    /* attention + proj -> TVi(t_attn) */
    {
        static float Attn[100*100], O[64*100];
        Tensor To = TVi(t_o);
        for (int hd = 0; hd < 2; hd++){
            matmul_batched(Q[hd], K[hd], Attn, 1, 100, 32, 100);
            for (int qi = 0; qi < 100; qi++){
                float *row = Attn + qi*100;
                float m = row[0];
                for (int j = 1; j < 100; j++) if (row[j] > m) m = row[j];
                float s = 0.f;
                for (int j = 0; j < 100; j++){ row[j] = expf(row[j]-m); s += row[j]; }
                float inv = 1.f/s;
                for (int j = 0; j < 100; j++) row[j] *= inv;
            }
            matmul_nt(V[hd], Attn, O, 1, 64, 100, 100);
            for (int c = 0; c < 64; c++)
                for (int t = 0; t < 100; t++)
                    To.data[(hd*64 + c)*HW + t] = O[c*100 + t];
        }
        /* o + pe */
        add_inplace(&To, &TVi(t_pe));

        conv2d(&To, &TVi(t_attn), c_proj, W);
    }

    /* x1 = x + proj */
    add_inplace(x, &TVi(t_attn));

    /* ffn: t_ffn0 = ffn0(x1); t_attn = ffn1(t_ffn0); x = x1 + t_attn */
    conv2d(x, &TVi(t_ffn0), c_ffn0, W);

    conv2d(&TVi(t_ffn0), &TVi(t_attn), c_ffn1, W);

    add_inplace(x, &TVi(t_attn));
}

/* -------- C2PSA (module 10) -------- */
static void c2psa_module10(void)
{
    /* cv1: 256->256 */
    conv2d(&TVi(T__model_9_Add_output_0), &TVi(T__model_10_cv1_act_Mul_output_0),
           CV_model_10_cv1, W);
    /* split */
    Tensor Ty = TVi(T__model_10_cv1_act_Mul_output_0);
    Tensor a  = { Ty.data,              128, 10, 10 };
    Tensor b  = { Ty.data + 128*100,    128, 10, 10 };

    /* PSABlock sobre b (in-place) */
    Tensor x = b;
    psablock_inplace(&x,
        CV_model_10_m_0_attn_qkv, CV_model_10_m_0_attn_pe, CV_model_10_m_0_attn_proj,
        CV_model_10_m_0_ffn_0, CV_model_10_m_0_ffn_1,
        T__model_10_m_m_0_attn_qkv_conv_Conv_output_0,
        T__model_10_m_m_0_attn_Reshape_2_output_0,
        T__model_10_m_m_0_attn_pe_conv_Conv_output_0,
        T__model_10_m_m_0_attn_Add_output_0,
        T__model_10_m_m_0_ffn_ffn_0_act_Mul_output_0,
        T__model_10_m_m_0_attn_proj_conv_Conv_output_0);

    /* concat [a, b] -> t82 */
    {
        const Tensor *parts[2] = { &a, &b };
        concat_ch(parts, 2, &TVi(T__model_10_Concat_output_0));
    }
    /* cv2 */
    conv2d(&TVi(T__model_10_Concat_output_0), &TVi(T__model_10_cv2_act_Mul_output_0),
           CV_model_10_cv2, W);
}

/* -------- C3k2 variante C (module 22) -------- */
static void c3k2_c_module22(void)
{

    /* cv1: 384->256 */
    conv2d(&TVi(T__model_21_Concat_output_0), &TVi(T__model_22_cv1_act_Mul_output_0),
           CV_model_22_cv1, W);
    Tensor Ty = TVi(T__model_22_cv1_act_Mul_output_0);
    Tensor a  = { Ty.data,              128, 10, 10 };
    Tensor b  = { Ty.data + 128*100,    128, 10, 10 };

    /* m = b + bcv2(bcv1(b)) */
    conv2d(&b, &TVi(T__model_22_m_0_m_0_0_cv1_act_Mul_output_0),
           CV_model_22_m_0_0_cv1, W);
    conv2d(&TVi(T__model_22_m_0_m_0_0_cv1_act_Mul_output_0),
           &TVi(T__model_22_m_0_m_0_0_cv2_act_Mul_output_0),
           CV_model_22_m_0_0_cv2, W);
    add_tensor(&b, &TVi(T__model_22_m_0_m_0_0_cv2_act_Mul_output_0),
               &TVi(T__model_22_m_0_m_0_0_Add_output_0));

    /* PSABlock sobre m (in-place) */
    Tensor m = TVi(T__model_22_m_0_m_0_0_Add_output_0);
    psablock_inplace(&m,
        CV_model_22_m_0_1_attn_qkv, CV_model_22_m_0_1_attn_pe, CV_model_22_m_0_1_attn_proj,
        CV_model_22_m_0_1_ffn_0, CV_model_22_m_0_1_ffn_1,
        T__model_22_m_0_m_0_1_attn_qkv_conv_Conv_output_0,
        T__model_22_m_0_m_0_1_attn_Reshape_2_output_0,
        T__model_22_m_0_m_0_1_attn_pe_conv_Conv_output_0,
        T__model_22_m_0_m_0_1_attn_Add_output_0,
        T__model_22_m_0_m_0_1_ffn_ffn_0_act_Mul_output_0,
        T__model_22_m_0_m_0_1_attn_proj_conv_Conv_output_0);

    /* concat [a, b, m] */
    {
        const Tensor *parts[3] = { &a, &b, &m };
        concat_ch(parts, 3, &TVi(T__model_22_Concat_output_0));
    }
    conv2d(&TVi(T__model_22_Concat_output_0), &TVi(T__model_22_cv2_act_Mul_output_0),
           CV_model_22_cv2, W);

}


/* -------- Forward completo -------- */
void yolo_forward(const float *input_nchw)
{

    memcpy(TVi(T_images).data, input_nchw, (size_t)IN_C*IN_H*IN_W*sizeof(float));

    /* m0, m1 */
    conv2d(&TVi(T_images), &TVi(T__model_0_act_Mul_output_0), CV_model_0, W);
    conv2d(&TVi(T__model_0_act_Mul_output_0), &TVi(T__model_1_act_Mul_output_0), CV_model_1, W);

    /* m2: C3k2 A (var A) */
    {
        C3k2A_Cfg g = {
            CV_model_2_cv1, CV_model_2_m_0_cv1, CV_model_2_m_0_cv2, CV_model_2_cv2,
            T__model_1_act_Mul_output_0,
            T__model_2_cv1_act_Mul_output_0,
            T__model_2_m_0_cv1_act_Mul_output_0,
            T__model_2_m_0_cv2_act_Mul_output_0,
            T__model_2_m_0_Add_output_0,
            T__model_2_Concat_output_0,
            T__model_2_cv2_act_Mul_output_0
        };
        c3k2_a(A, W, &g);
    }

    /* m3, m4 (A) */
    conv2d(&TVi(T__model_2_cv2_act_Mul_output_0), &TVi(T__model_3_act_Mul_output_0), CV_model_3, W);
    {
        C3k2A_Cfg g = {
            CV_model_4_cv1, CV_model_4_m_0_cv1, CV_model_4_m_0_cv2, CV_model_4_cv2,
            T__model_3_act_Mul_output_0,
            T__model_4_cv1_act_Mul_output_0,
            T__model_4_m_0_cv1_act_Mul_output_0,
            T__model_4_m_0_cv2_act_Mul_output_0,
            T__model_4_m_0_Add_output_0,
            T__model_4_Concat_output_0,
            T__model_4_cv2_act_Mul_output_0
        };
        c3k2_a(A, W, &g);
    }

    /* m5, m6 (C3k2 B) */
    conv2d(&TVi(T__model_4_cv2_act_Mul_output_0), &TVi(T__model_5_act_Mul_output_0), CV_model_5, W);
    {
        C3k2B_Cfg g = {
            CV_model_6_cv1,
            CV_model_6_m_0_cv1, CV_model_6_m_0_cv2, CV_model_6_m_0_cv3,
            CV_model_6_cv2,
            { {CV_model_6_m_0_m_0_cv1, CV_model_6_m_0_m_0_cv2},
              {CV_model_6_m_0_m_1_cv1, CV_model_6_m_0_m_1_cv2} },
            2,
            T__model_5_act_Mul_output_0,
            T__model_6_cv1_act_Mul_output_0,
            T__model_6_m_0_cv1_act_Mul_output_0,
            T__model_6_m_0_cv2_act_Mul_output_0,
            T__model_6_m_0_m_m_0_cv1_act_Mul_output_0,
            T__model_6_m_0_m_m_0_cv2_act_Mul_output_0,   /* ← NOVO */
            T__model_6_m_0_m_m_0_Add_output_0,
            T__model_6_m_0_m_m_1_cv1_act_Mul_output_0,
            T__model_6_m_0_m_m_1_cv2_act_Mul_output_0,   /* ← NOVO */
            T__model_6_m_0_m_m_1_Add_output_0,
            T__model_6_m_0_Concat_output_0,
            T__model_6_m_0_cv3_act_Mul_output_0,
            T__model_6_Concat_output_0,
            T__model_6_cv2_act_Mul_output_0
        };
        c3k2_b(A, W, &g);
    }

    /* m7, m8 (B) */
    conv2d(&TVi(T__model_6_cv2_act_Mul_output_0), &TVi(T__model_7_act_Mul_output_0), CV_model_7, W);
    {
        C3k2B_Cfg g = {
            CV_model_8_cv1,
            CV_model_8_m_0_cv1, CV_model_8_m_0_cv2, CV_model_8_m_0_cv3,
            CV_model_8_cv2,
            { {CV_model_8_m_0_m_0_cv1, CV_model_8_m_0_m_0_cv2},
              {CV_model_8_m_0_m_1_cv1, CV_model_8_m_0_m_1_cv2} },
            2,
            T__model_7_act_Mul_output_0,
            T__model_8_cv1_act_Mul_output_0,
            T__model_8_m_0_cv1_act_Mul_output_0,
            T__model_8_m_0_cv2_act_Mul_output_0,
            T__model_8_m_0_m_m_0_cv1_act_Mul_output_0,
            T__model_8_m_0_m_m_0_cv2_act_Mul_output_0,   /* ← NOVO */
            T__model_8_m_0_m_m_0_Add_output_0,
            T__model_8_m_0_m_m_1_cv1_act_Mul_output_0,
            T__model_8_m_0_m_m_1_cv2_act_Mul_output_0,   /* ← NOVO */
            T__model_8_m_0_m_m_1_Add_output_0,
            T__model_8_m_0_Concat_output_0,
            T__model_8_m_0_cv3_act_Mul_output_0,
            T__model_8_Concat_output_0,
            T__model_8_cv2_act_Mul_output_0
        };
        c3k2_b(A, W, &g);
    }

    /* m9 SPPF */
    {
        SPPF_Cfg g = {
            CV_model_9_cv1, CV_model_9_cv2,
            T__model_8_cv2_act_Mul_output_0,
            T__model_9_cv1_conv_Conv_output_0,
            T__model_9_m_MaxPool_output_0,
            T__model_9_m_1_MaxPool_output_0,
            T__model_9_m_2_MaxPool_output_0,
            T__model_9_Concat_output_0,
            T__model_9_cv2_act_Mul_output_0,
            T__model_9_Add_output_0
        };
        sppf(A, W, &g);
    }

    /* m10 C2PSA */
    c2psa_module10();

    /* m11 Resize x2 */
    resize_nearest(&TVi(T__model_10_cv2_act_Mul_output_0),
                   &TVi(T__model_11_Resize_output_0), 2);

    /* m12 Concat [m11 | m6] */
    {
        const Tensor *parts[2] = { &TVi(T__model_11_Resize_output_0),
                                   &TVi(T__model_6_cv2_act_Mul_output_0) };
        concat_ch(parts, 2, &TVi(T__model_12_Concat_output_0));
    }

    /* m13 C3k2 B */
    {
        C3k2B_Cfg g = {
            CV_model_13_cv1,
            CV_model_13_m_0_cv1, CV_model_13_m_0_cv2, CV_model_13_m_0_cv3,
            CV_model_13_cv2,
            { {CV_model_13_m_0_m_0_cv1, CV_model_13_m_0_m_0_cv2},
              {CV_model_13_m_0_m_1_cv1, CV_model_13_m_0_m_1_cv2} },
            2,
            T__model_12_Concat_output_0,
            T__model_13_cv1_act_Mul_output_0,
            T__model_13_m_0_cv1_act_Mul_output_0,
            T__model_13_m_0_cv2_act_Mul_output_0,
            T__model_13_m_0_m_m_0_cv1_act_Mul_output_0,
            T__model_13_m_0_m_m_0_cv2_act_Mul_output_0,   /* ← NOVO (t92) */
            T__model_13_m_0_m_m_0_Add_output_0,
            T__model_13_m_0_m_m_1_cv1_act_Mul_output_0,
            T__model_13_m_0_m_m_1_cv2_act_Mul_output_0,   /* ← NOVO (t95) */
            T__model_13_m_0_m_m_1_Add_output_0,
            T__model_13_m_0_Concat_output_0,
            T__model_13_m_0_cv3_act_Mul_output_0,
            T__model_13_Concat_output_0,
            T__model_13_cv2_act_Mul_output_0
        };
        c3k2_b(A, W, &g);
    }

    /* m14 Resize x2 */
    resize_nearest(&TVi(T__model_13_cv2_act_Mul_output_0),
                   &TVi(T__model_14_Resize_output_0), 2);

    /* m15 Concat [m14 | m4] */
    {
        const Tensor *parts[2] = { &TVi(T__model_14_Resize_output_0),
                                   &TVi(T__model_4_cv2_act_Mul_output_0) };
        concat_ch(parts, 2, &TVi(T__model_15_Concat_output_0));
    }

    /* m16 C3k2 B -> P3 */
    {
        C3k2B_Cfg g = {
            CV_model_16_cv1,
            CV_model_16_m_0_cv1, CV_model_16_m_0_cv2, CV_model_16_m_0_cv3,
            CV_model_16_cv2,
            { {CV_model_16_m_0_m_0_cv1, CV_model_16_m_0_m_0_cv2},
              {CV_model_16_m_0_m_1_cv1, CV_model_16_m_0_m_1_cv2} },
            2,
            T__model_15_Concat_output_0,
            T__model_16_cv1_act_Mul_output_0,
            T__model_16_m_0_cv1_act_Mul_output_0,
            T__model_16_m_0_cv2_act_Mul_output_0,
            T__model_16_m_0_m_m_0_cv1_act_Mul_output_0,
            T__model_16_m_0_m_m_0_cv2_act_Mul_output_0,   /* ← NOVO (t109) */
            T__model_16_m_0_m_m_0_Add_output_0,
            T__model_16_m_0_m_m_1_cv1_act_Mul_output_0,
            T__model_16_m_0_m_m_1_cv2_act_Mul_output_0,   /* ← NOVO (t112) */
            T__model_16_m_0_m_m_1_Add_output_0,
            T__model_16_m_0_Concat_output_0,
            T__model_16_m_0_cv3_act_Mul_output_0,
            T__model_16_Concat_output_0,
            T__model_16_cv2_act_Mul_output_0
        };
        c3k2_b(A, W, &g);
    }

    /* m17 Conv 3x3 s2 -> 64x20x20 */
    conv2d(&TVi(T__model_16_cv2_act_Mul_output_0), &TVi(T__model_17_act_Mul_output_0),
           CV_model_17, W);

    /* m18 Concat [m17 | m13] */
    {
        const Tensor *parts[2] = { &TVi(T__model_17_act_Mul_output_0),
                                   &TVi(T__model_13_cv2_act_Mul_output_0) };
        concat_ch(parts, 2, &TVi(T__model_18_Concat_output_0));
    }

    /* m19 C3k2 B -> P4 */
    {
        C3k2B_Cfg g = {
            CV_model_19_cv1,
            CV_model_19_m_0_cv1, CV_model_19_m_0_cv2, CV_model_19_m_0_cv3,
            CV_model_19_cv2,
            { {CV_model_19_m_0_m_0_cv1, CV_model_19_m_0_m_0_cv2},
              {CV_model_19_m_0_m_1_cv1, CV_model_19_m_0_m_1_cv2} },
            2,
            T__model_18_Concat_output_0,
            T__model_19_cv1_act_Mul_output_0,
            T__model_19_m_0_cv1_act_Mul_output_0,
            T__model_19_m_0_cv2_act_Mul_output_0,
            T__model_19_m_0_m_m_0_cv1_act_Mul_output_0,
            T__model_19_m_0_m_m_0_cv2_act_Mul_output_0,   /* ← NOVO (t140) */
            T__model_19_m_0_m_m_0_Add_output_0,
            T__model_19_m_0_m_m_1_cv1_act_Mul_output_0,
            T__model_19_m_0_m_m_1_cv2_act_Mul_output_0,   /* ← NOVO (t143) */
            T__model_19_m_0_m_m_1_Add_output_0,
            T__model_19_m_0_Concat_output_0,
            T__model_19_m_0_cv3_act_Mul_output_0,
            T__model_19_Concat_output_0,
            T__model_19_cv2_act_Mul_output_0
        };
        c3k2_b(A, W, &g);
    }

    /* m20 Conv 3x3 s2 -> 128x10x10 */
    conv2d(&TVi(T__model_19_cv2_act_Mul_output_0), &TVi(T__model_20_act_Mul_output_0),
           CV_model_20, W);

    /* m21 Concat [m20 | m10] */
    {
        const Tensor *parts[2] = { &TVi(T__model_20_act_Mul_output_0),
                                   &TVi(T__model_10_cv2_act_Mul_output_0) };
        concat_ch(parts, 2, &TVi(T__model_21_Concat_output_0));
    }

    /* m22 C3k2 C -> P5 */
    c3k2_c_module22();


    /* ============= Segment26 ============= */
    /* Cada nivel: box(cls/coef) -> reshape -> concat 2100 -> decode -> topk */

    /* nivel 0: P3 = t117 (64x40x40) */
    {
        HeadBranch_Cfg g = {
            CV_model_23_one2one_cv2_0_0,
            CV_model_23_one2one_cv2_0_1,
            CV_model_23_one2one_cv2_0_2,
            CV_model_23_one2one_cv3_0_0_0,
            CV_model_23_one2one_cv3_0_0_1,
            CV_model_23_one2one_cv3_0_1_0,
            CV_model_23_one2one_cv3_0_1_1,
            CV_model_23_one2one_cv3_0_2,
            CV_model_23_one2one_cv4_0_0,
            CV_model_23_one2one_cv4_0_1,
            CV_model_23_one2one_cv4_0_2,
            T__model_16_cv2_act_Mul_output_0,
            T__model_23_one2one_cv2_0_one2one_cv2_0_0_act_Mul_output_0,
            T__model_23_one2one_cv2_0_one2one_cv2_0_1_act_Mul_output_0,
            T__model_23_one2one_cv2_0_one2one_cv2_0_2_Conv_output_0,
            T__model_23_one2one_cv3_0_one2one_cv3_0_0_one2one_cv3_0_0_0_act_Mul_output_0,
            T__model_23_one2one_cv3_0_one2one_cv3_0_0_one2one_cv3_0_0_1_act_Mul_output_0,
            T__model_23_one2one_cv3_0_one2one_cv3_0_1_one2one_cv3_0_1_0_act_Mul_output_0,
            T__model_23_one2one_cv3_0_one2one_cv3_0_1_one2one_cv3_0_1_1_act_Mul_output_0,
            T__model_23_one2one_cv3_0_one2one_cv3_0_2_Conv_output_0,
            T__model_23_one2one_cv4_0_one2one_cv4_0_0_act_Mul_output_0,
            T__model_23_one2one_cv4_0_one2one_cv4_0_1_act_Mul_output_0,
            T__model_23_one2one_cv4_0_one2one_cv4_0_2_Conv_output_0,
            T__model_23_Reshape_output_0,
            T__model_23_Reshape_3_output_0,
            T__model_23_Reshape_6_output_0,
            40, 40
        };
        head_branch(A, W, &g);
    }
    {
        Tensor tb = TVi(T__model_23_one2one_cv2_0_one2one_cv2_0_2_Conv_output_0);
        Tensor tc = TVi(T__model_23_one2one_cv3_0_one2one_cv3_0_2_Conv_output_0);
        Tensor tk = TVi(T__model_23_one2one_cv4_0_one2one_cv4_0_2_Conv_output_0);
    }

    /* nivel 1: P4 */
    {
        HeadBranch_Cfg g = {
            CV_model_23_one2one_cv2_1_0,
            CV_model_23_one2one_cv2_1_1,
            CV_model_23_one2one_cv2_1_2,
            CV_model_23_one2one_cv3_1_0_0,
            CV_model_23_one2one_cv3_1_0_1,
            CV_model_23_one2one_cv3_1_1_0,
            CV_model_23_one2one_cv3_1_1_1,
            CV_model_23_one2one_cv3_1_2,
            CV_model_23_one2one_cv4_1_0,
            CV_model_23_one2one_cv4_1_1,
            CV_model_23_one2one_cv4_1_2,
            T__model_19_cv2_act_Mul_output_0,
            T__model_23_one2one_cv2_1_one2one_cv2_1_0_act_Mul_output_0,
            T__model_23_one2one_cv2_1_one2one_cv2_1_1_act_Mul_output_0,
            T__model_23_one2one_cv2_1_one2one_cv2_1_2_Conv_output_0,
            T__model_23_one2one_cv3_1_one2one_cv3_1_0_one2one_cv3_1_0_0_act_Mul_output_0,
            T__model_23_one2one_cv3_1_one2one_cv3_1_0_one2one_cv3_1_0_1_act_Mul_output_0,
            T__model_23_one2one_cv3_1_one2one_cv3_1_1_one2one_cv3_1_1_0_act_Mul_output_0,
            T__model_23_one2one_cv3_1_one2one_cv3_1_1_one2one_cv3_1_1_1_act_Mul_output_0,
            T__model_23_one2one_cv3_1_one2one_cv3_1_2_Conv_output_0,
            T__model_23_one2one_cv4_1_one2one_cv4_1_0_act_Mul_output_0,
            T__model_23_one2one_cv4_1_one2one_cv4_1_1_act_Mul_output_0,
            T__model_23_one2one_cv4_1_one2one_cv4_1_2_Conv_output_0,
            T__model_23_Reshape_1_output_0,
            T__model_23_Reshape_4_output_0,
            T__model_23_Reshape_7_output_0,
            20, 20
        };
        head_branch(A, W, &g);
    }
    {
        Tensor tb = TVi(T__model_23_one2one_cv2_1_one2one_cv2_1_2_Conv_output_0);
        Tensor tc = TVi(T__model_23_one2one_cv3_1_one2one_cv3_1_2_Conv_output_0);
        Tensor tk = TVi(T__model_23_one2one_cv4_1_one2one_cv4_1_2_Conv_output_0);
    }

    /* nivel 2: P5 */
    {
        HeadBranch_Cfg g = {
            CV_model_23_one2one_cv2_2_0,
            CV_model_23_one2one_cv2_2_1,
            CV_model_23_one2one_cv2_2_2,
            CV_model_23_one2one_cv3_2_0_0,
            CV_model_23_one2one_cv3_2_0_1,
            CV_model_23_one2one_cv3_2_1_0,
            CV_model_23_one2one_cv3_2_1_1,
            CV_model_23_one2one_cv3_2_2,
            CV_model_23_one2one_cv4_2_0,
            CV_model_23_one2one_cv4_2_1,
            CV_model_23_one2one_cv4_2_2,
            T__model_22_cv2_act_Mul_output_0,
            T__model_23_one2one_cv2_2_one2one_cv2_2_0_act_Mul_output_0,
            T__model_23_one2one_cv2_2_one2one_cv2_2_1_act_Mul_output_0,
            T__model_23_one2one_cv2_2_one2one_cv2_2_2_Conv_output_0,
            T__model_23_one2one_cv3_2_one2one_cv3_2_0_one2one_cv3_2_0_0_act_Mul_output_0,
            T__model_23_one2one_cv3_2_one2one_cv3_2_0_one2one_cv3_2_0_1_act_Mul_output_0,
            T__model_23_one2one_cv3_2_one2one_cv3_2_1_one2one_cv3_2_1_0_act_Mul_output_0,
            T__model_23_one2one_cv3_2_one2one_cv3_2_1_one2one_cv3_2_1_1_act_Mul_output_0,
            T__model_23_one2one_cv3_2_one2one_cv3_2_2_Conv_output_0,
            T__model_23_one2one_cv4_2_one2one_cv4_2_0_act_Mul_output_0,
            T__model_23_one2one_cv4_2_one2one_cv4_2_1_act_Mul_output_0,
            T__model_23_one2one_cv4_2_one2one_cv4_2_2_Conv_output_0,
            T__model_23_Reshape_2_output_0,
            T__model_23_Reshape_5_output_0,
            T__model_23_Reshape_8_output_0,
            10, 10
        };
        head_branch(A, W, &g);
    }

    {
        Tensor tb = TVi(T__model_23_one2one_cv2_2_one2one_cv2_2_2_Conv_output_0);
        Tensor tc = TVi(T__model_23_one2one_cv3_2_one2one_cv3_2_2_Conv_output_0);
        Tensor tk = TVi(T__model_23_one2one_cv4_2_one2one_cv4_2_2_Conv_output_0);
    }

    /* Concat dos 3 niveis: box [4,2100], cls [1,2100], coef [32,2100] */
    /* Box [4,2100] */
    {
        const Tensor *p[3] = {
            &TVi(T__model_23_one2one_cv2_0_one2one_cv2_0_2_Conv_output_0),
            &TVi(T__model_23_one2one_cv2_1_one2one_cv2_1_2_Conv_output_0),
            &TVi(T__model_23_one2one_cv2_2_one2one_cv2_2_2_Conv_output_0)
        };
        concat_w(p, 3, &TVi(T__model_23_Concat_output_0));
    }
    /* Cls [1,2100] */
    {
        const Tensor *p[3] = {
            &TVi(T__model_23_one2one_cv3_0_one2one_cv3_0_2_Conv_output_0),
            &TVi(T__model_23_one2one_cv3_1_one2one_cv3_1_2_Conv_output_0),
            &TVi(T__model_23_one2one_cv3_2_one2one_cv3_2_2_Conv_output_0)
        };
        concat_w(p, 3, &TVi(T__model_23_Concat_1_output_0));
    }
    /* Coef [32,2100] */
    {
        const Tensor *p[3] = {
            &TVi(T__model_23_one2one_cv4_0_one2one_cv4_0_2_Conv_output_0),
            &TVi(T__model_23_one2one_cv4_1_one2one_cv4_1_2_Conv_output_0),
            &TVi(T__model_23_one2one_cv4_2_one2one_cv4_2_2_Conv_output_0)
        };
        concat_w(p, 3, &TVi(T__model_23_Concat_2_output_0));
    }

    /* Decode + TopK -> output0 [300, 38] */
    head_decode_topk(A,
                     T__model_23_Concat_output_0,
                     T__model_23_Concat_1_output_0,
                     T__model_23_Concat_2_output_0,
                     T_output0);

    /* Proto26 -> output1 [32, 80, 80] */
    proto26(A, W,
            T__model_16_cv2_act_Mul_output_0,
            T__model_19_cv2_act_Mul_output_0,
            T__model_22_cv2_act_Mul_output_0,
            T__model_23_proto_feat_refine_0_act_Mul_output_0,
            T__model_23_proto_feat_refine_1_act_Mul_output_0,
            T__model_23_proto_Resize_output_0,
            T__model_23_proto_Resize_1_output_0,
            T__model_23_proto_Add_output_0,
            T__model_23_proto_feat_fuse_act_Mul_output_0,
            T__model_23_proto_cv1_act_Mul_output_0,
            T__model_23_proto_upsample_ConvTranspose_output_0,
            T__model_23_proto_cv2_act_Mul_output_0,
            T_output1);
}
