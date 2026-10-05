/* blocks.c -- C3k2 A/B/C, SPPF, C2PSA, Attention. */
#include "yolo26n_seg.h"

/* helper: view de um intervalo de canais de um tensor NCHW contiguo. */
static inline Tensor chan_view(const Tensor *t, int c0, int c_count){
    Tensor v;
    v.data = t->data + (size_t)c0 * t->h * t->w;
    v.c = c_count; v.h = t->h; v.w = t->w;
    return v;
}

/* ---------------- C3k2 variante A ---------------- */
void c3k2_a(float *A, const float *blob, const C3k2A_Cfg *g)
{
    Tensor Tin  = TV(A, g->t_in);
    Tensor Ty   = TV(A, g->t_y);
    Tensor Tb1  = TV(A, g->t_bcv1);
    Tensor Tb2  = TV(A, g->t_bcv2);
    Tensor Tad  = TV(A, g->t_add);
    Tensor Tcat = TV(A, g->t_cat);
    Tensor Tout = TV(A, g->t_out);

    conv2d(&Tin, &Ty, g->cv1, blob);

    int half = Ty.c / 2;
    Tensor a = chan_view(&Ty, 0,    half);
    Tensor b = chan_view(&Ty, half, half);

    conv2d(&b,   &Tb1, g->bcv1, blob);
    conv2d(&Tb1, &Tb2, g->bcv2, blob);
    add_tensor(&b, &Tb2, &Tad);

    const Tensor *parts[3] = { &a, &b, &Tad };
    concat_ch(parts, 3, &Tcat);

    conv2d(&Tcat, &Tout, g->cv2, blob);
}

/* ---------------- C3k2 variante B ---------------- */
void c3k2_b(float *A, const float *blob, const C3k2B_Cfg *g)
{
    Tensor Tin  = TV(A, g->t_in);
    Tensor Ty   = TV(A, g->t_y);
    Tensor Tu   = TV(A, g->t_u);
    Tensor Tv   = TV(A, g->t_v);
    Tensor Tm   = TV(A, g->t_m);
    Tensor Tcat3= TV(A, g->t_cat3);
    Tensor Tout = TV(A, g->t_out);
    Tensor T2   = TV(A, g->t_cat2);

    conv2d(&Tin, &Ty, g->cv1, blob);

    int half = Ty.c / 2;
    Tensor a = chan_view(&Ty, 0,    half);
    Tensor b = chan_view(&Ty, half, half);

    conv2d(&b, &Tu, g->mcv1, blob);
    conv2d(&b, &Tv, g->mcv2, blob);

    int cur_id = g->t_u;
    for (int i = 0; i < g->n_bot; i++){
        Tensor Tcur = TV(A, cur_id);
        if (i == 0){
            Tensor Tm1    = TV(A, g->t_m_bot1);
            Tensor Tcv2_1 = TV(A, g->t_cv2_bot1);
            Tensor Tb_out = TV(A, g->t_bot1_out);
            conv2d(&Tcur,  &Tm1,    g->bot[0][0], blob);
            conv2d(&Tm1,   &Tcv2_1, g->bot[0][1], blob);
            add_tensor(&Tcur, &Tcv2_1, &Tb_out);
            cur_id = g->t_bot1_out;
        } else {
            Tensor Tm2    = TV(A, g->t_m_bot2);
            Tensor Tcv2_2 = TV(A, g->t_cv2_bot2);
            Tensor Tb2_out= TV(A, g->t_bot2_out);
            conv2d(&Tcur,  &Tm2,    g->bot[1][0], blob);
            conv2d(&Tm2,   &Tcv2_2, g->bot[1][1], blob);
            add_tensor(&Tcur, &Tcv2_2, &Tb2_out);
            cur_id = g->t_bot2_out;
        }
    }
    {
        Tensor Tcur = TV(A, cur_id);
        const Tensor *parts[2] = { &Tcur, &Tv };
        concat_ch(parts, 2, &T2);
    }
    conv2d(&T2, &Tm, g->mcv3, blob);
    {
        const Tensor *parts[3] = { &a, &b, &Tm };
        concat_ch(parts, 3, &Tcat3);
    }
    conv2d(&Tcat3, &Tout, g->cv2, blob);
}

/* ---------------- Attention (PSABlock) ---------------- */
/* Estrutura esperada: qkv C->2*H*D, H=ATTN_HEADS, D=ATTN_HEAD_DIM/2? nao.
 * No modelo: qkv C=128 -> 256 (2*128). Por head: q(32),k(32),v(64).
 * t_qkv:      [256, 10, 10] (saida da conv qkv)
 * t_v_resh:   [128, 10, 10] (v apos reshape, para pe)
 * t_pe:       [128, 10, 10] (saida da dwconv pe)
 * t_o_sum:    [128, 10, 10] (o + pe)
 * t_proj_out: [128, 10, 10] (saida da proj, = attention out)
 */
void psa_attention(float *A, const float *blob,
                   ConvId qkv, ConvId pe, ConvId proj,
                   int t_in, int t_qkv, int t_v_reshaped,
                   int t_pe, int t_o_sum, int t_proj_out)
{
    const int H = ATTN_HEADS;      /* 2 */
    const int KD = ATTN_KEY_DIM;   /* 32 */
    const int HD = ATTN_HEAD_DIM;  /* 64 */
    const int T  = ATTN_TOKENS;    /* 100 */
    const float scale = ATTN_SCALE;

    Tensor Tin  = TV(A, t_in);
    Tensor Tqkv = TV(A, t_qkv);
    Tensor Tpe  = TV(A, t_pe);
    Tensor Tsum = TV(A, t_o_sum);
    Tensor Tprj = TV(A, t_proj_out);

    /* 1) qkv conv -> Tqkv [256,10,10] */
    conv2d(&Tin, &Tqkv, qkv, blob);

    /* 2) montar por cabeca q[KD,T], k[KD,T], v[HD,T] */
    /*    no layout ONNX: reshape [256,10,10] -> [2,128,100] significa head-major:
     *    head0 = canais 0..127, head1 = canais 128..255
     *    dentro do head: q(32) k(32) v(64)
     *    reshape para [100] e por (h*w) row-major.
     */
    const int HW = T;  /* 100 tokens */
    /* ponteiros para cabeca 0 e 1 */
    const float *h0 = Tqkv.data;                    /* canais 0..127 */
    const float *h1 = Tqkv.data + 128*HW;           /* canais 128..255 */

    /* buffers por head: q[32,T], k[32,T], v[64,T] */
    float Q[2][32*100];
    float K[2][32*100];
    float V[2][64*100];

    for (int hd = 0; hd < 2; hd++){
        const float *hh = hd == 0 ? h0 : h1;
        for (int t = 0; t < T; t++){
            for (int i = 0; i < 32; i++) Q[hd][i*T + t] = hh[(0  + i)*HW + t];
            for (int i = 0; i < 32; i++) K[hd][i*T + t] = hh[(32 + i)*HW + t];
            for (int i = 0; i < 64; i++) V[hd][i*T + t] = hh[(64 + i)*HW + t];
        }
        /* escala em Q */
        for (int i = 0; i < 32*T; i++) Q[hd][i] *= scale;
    }

    /* 3) escrever V reshaped em T_v_reshaped [128,10,10] para a pe */
    Tensor Tv = TV(A, t_v_reshaped);
    for (int hd = 0; hd < 2; hd++)
        for (int c = 0; c < 64; c++)
            for (int t = 0; t < T; t++)
                Tv.data[(hd*64 + c)*HW + t] = V[hd][c*T + t];

    /* 4) pe conv (depthwise 3x3, sem SiLU) */
    conv2d(&Tv, &Tpe, pe, blob);

    /* 5) attn = softmax(Q^T K / 1) [T,T] por head;  o = V @ attn^T */
    for (int hd = 0; hd < 2; hd++){
        float Attn[100*100];
        float O[64*100];
        /* scores[qi,tj] = sum_k Q[k,qi]*K[k,tj] -> com Q^T @ K: (Q^T)[qi,k] = Q[k,qi] */
        matmul_batched(Q[hd], K[hd], Attn, 1, T, 32, T);  /* A[T,32] @ B[32,T] */
        /* softmax por linha (qi): cols = tj = T */
        for (int qi = 0; qi < T; qi++){
            float *row = Attn + qi*T;
            float m = row[0];
            for (int j = 1; j < T; j++) if (row[j] > m) m = row[j];
            float s = 0.f;
            for (int j = 0; j < T; j++){ row[j] = expf(row[j]-m); s += row[j]; }
            float inv = 1.f/s;
            for (int j = 0; j < T; j++) row[j] *= inv;
        }
        /* O = V @ Attn^T:  O[c,qi] = sum_j V[c,j] * Attn[qi,j] */
        matmul_nt(V[hd], Attn, O, 1, 64, T, T);
        /* escrever O no Tsum (mesmo layout que Tv) */
        for (int c = 0; c < 64; c++)
            for (int t = 0; t < T; t++)
                Tsum.data[(hd*64 + c)*HW + t] = O[c*T + t];
    }

    /* 6) o + pe */
    add_inplace(&Tsum, &Tpe);

    /* 7) proj conv */
    conv2d(&Tsum, &Tprj, proj, blob);
}


/* ---------------- SPPF ---------------- */
void sppf(float *A, const float *blob, const SPPF_Cfg *g)
{
    Tensor Tin  = TV(A, g->t_in);
    Tensor Tc1  = TV(A, g->t_cv1);
    Tensor Tp1  = TV(A, g->t_p1);
    Tensor Tp2  = TV(A, g->t_p2);
    Tensor Tp3  = TV(A, g->t_p3);
    Tensor Tcat = TV(A, g->t_cat);
    Tensor Tc2  = TV(A, g->t_cv2);
    Tensor Tout = TV(A, g->t_out);

    conv2d(&Tin, &Tc1, g->cv1, blob);   /* sem SiLU */
    maxpool5s1p2(&Tc1, &Tp1);
    maxpool5s1p2(&Tp1, &Tp2);
    maxpool5s1p2(&Tp2, &Tp3);
    {
        const Tensor *parts[4] = { &Tc1, &Tp1, &Tp2, &Tp3 };
        concat_ch(parts, 4, &Tcat);
    }
    conv2d(&Tcat, &Tc2, g->cv2, blob);
    add_tensor(&Tin, &Tc2, &Tout);      /* shortcut */
}
