/* head.c -- Segment26 (branch + decode + topk) e Proto26. */
#include "yolo26n_seg.h"


/* ------------- Uma ramificacao box/cls/coef por nivel ------------- */
void head_branch(float *A, const float *blob, const HeadBranch_Cfg *g)
{
    Tensor Tin   = TV(A, g->t_in);
    Tensor Tbox0 = TV(A, g->t_box0);
    Tensor Tbox1 = TV(A, g->t_box1);
    Tensor Tboxo = TV(A, g->t_box_out);

    Tensor Tcls0 = TV(A, g->t_cls0);
    Tensor Tcls1 = TV(A, g->t_cls1);
    Tensor Tcls2 = TV(A, g->t_cls2);
    Tensor Tcls3 = TV(A, g->t_cls3);
    Tensor Tcls4 = TV(A, g->t_cls4);

    Tensor Tcoef0 = TV(A, g->t_coef0);
    Tensor Tcoef1 = TV(A, g->t_coef1);
    Tensor Tcoefo = TV(A, g->t_coef_out);


    conv2d(&Tin,   &Tbox0, g->box_cv0, blob);
    conv2d(&Tbox0, &Tbox1, g->box_cv1, blob);
    conv2d(&Tbox1, &Tboxo, g->box_cv2, blob);

    conv2d(&Tin,   &Tcls0, g->cls_cv0, blob);
    conv2d(&Tcls0, &Tcls1, g->cls_cv1, blob);
    conv2d(&Tcls1, &Tcls2, g->cls_cv2, blob);
    conv2d(&Tcls2, &Tcls3, g->cls_cv3, blob);
    conv2d(&Tcls3, &Tcls4, g->cls_cv4, blob);

    conv2d(&Tin,   &Tcoef0, g->coef_cv0, blob);
    conv2d(&Tcoef0, &Tcoef1, g->coef_cv1, blob);
    conv2d(&Tcoef1, &Tcoefo, g->coef_cv2, blob);
}

/* ------------- Decode + TopK -------------
 * t_box_2100:  [4, 2100]
 * t_score_2100:[1, 2100]   (logits)
 * t_coef_2100: [32, 2100]
 * t_output0:   [300, 38]
 */
void head_decode_topk(float *A, int t_box_2100, int t_score_2100, int t_coef_2100,
                      int t_output0)
{
    Tensor Tb = TV(A, t_box_2100);
    Tensor Ts = TV(A, t_score_2100);
    /* Tabelas analiticas: anchors (cx,cy) por nivel e strides. */
    static const int SZ[3]    = { 40, 20, 10 };
    static const float STR[3] = { 8.f, 16.f, 32.f };
    enum { N = N_ANCHORS };   /* agora sim é constante de compilação */
    static float CX[N], CY[N], ST[N];

    static int   inited = 0;
    if (!inited){
        int idx = 0;
        for (int lvl = 0; lvl < 3; lvl++){
            int H = SZ[lvl], W = SZ[lvl];
            for (int y = 0; y < H; y++){
                for (int x = 0; x < W; x++){
                    CX[idx] = x + 0.5f;
                    CY[idx] = y + 0.5f;
                    ST[idx] = STR[lvl];
                    idx++;
                }
            }
        }
        inited = 1;
    }

    const float *B = TV(A, t_box_2100).data;   /* [4,2100] */
    const float *C = TV(A, t_coef_2100).data;  /* [32,2100] */
    float       *O = TV(A, t_output0).data;    /* [300,38] */

    /* 1) sigmoid no score — copiando para buffer local estatico para
     *    eliminar qualquer chance de aliasing com O */
    static float S_local[2100];
    {
        const float *S_in = TV(A, t_score_2100).data;
        for (int i = 0; i < N; i++) S_local[i] = 1.0f/(1.0f + expf(-S_in[i]));
    }
    float *S = S_local;

    /* 2) TopK sobre S */
    static float topv[TOPK];
    static int   topi[TOPK];
    topk_desc(S, N, TOPK, topv, topi);

    /* 3) construir saida [300, 38] = [x1,y1,x2,y2, score, cls=0, coef[32]] */
    for (int t = 0; t < TOPK; t++){
        int i = topi[t];
        float l = B[0*N + i];
        float tp= B[1*N + i];
        float r = B[2*N + i];
        float b = B[3*N + i];
        float cx = CX[i], cy = CY[i], s = ST[i];
        float x1 = (cx - l) * s;
        float y1 = (cy - tp) * s;
        float x2 = (cx + r) * s;
        float y2 = (cy + b) * s;

        float *row = O + t*OUT0_COLS;
        row[0] = x1; row[1] = y1; row[2] = x2; row[3] = y2;
        row[4] = S[i];       /* score */
        row[5] = 0.0f;       /* cls: nc=1, sempre 0 */
        for (int k = 0; k < NM; k++) row[6 + k] = C[k*N + i];
    }
}

/* ------------- Proto26 -------------
 * P3: [64, 40, 40]
 * P4: [128, 20, 20]
 * P5: [256, 10, 10]
 * saida: output1 [32, 80, 80]
 */
void proto26(float *A, const float *blob,
             int t_p3, int t_p4, int t_p5,
             int t_p4_ref, int t_p5_ref,
             int t_p4_up, int t_p5_up,
             int t_sum_p3r4, int t_fuse, int t_cv1, int t_convT,
             int t_cv2, int t_output1)
{
    Tensor Tp3       = TV(A, t_p3);
    Tensor Tp4       = TV(A, t_p4);
    Tensor Tp5       = TV(A, t_p5);
    Tensor Tp4_ref   = TV(A, t_p4_ref);
    Tensor Tp5_ref   = TV(A, t_p5_ref);
    Tensor Tp4_up    = TV(A, t_p4_up);
    Tensor Tp5_up    = TV(A, t_p5_up);
    Tensor Tsum_p3r4 = TV(A, t_sum_p3r4);
    Tensor Tfuse     = TV(A, t_fuse);
    Tensor Tcv1      = TV(A, t_cv1);
    Tensor TconvT    = TV(A, t_convT);
    Tensor Tcv2      = TV(A, t_cv2);
    Tensor Toutput1  = TV(A, t_output1);

    conv2d(&Tp4, &Tp4_ref, CV_model_23_proto_feat_refine_0, blob);
    resize_nearest(&Tp4_ref, &Tp4_up, 2);

    conv2d(&Tp5, &Tp5_ref, CV_model_23_proto_feat_refine_1, blob);
    resize_nearest(&Tp5_ref, &Tp5_up, 4);

    add_tensor(&Tp3, &Tp4_up, &Tsum_p3r4);
    add_inplace(&Tsum_p3r4, &Tp5_up);

    conv2d(&Tsum_p3r4, &Tfuse, CV_model_23_proto_feat_fuse, blob);
    conv2d(&Tfuse,     &Tcv1,  CV_model_23_proto_cv1, blob);
    convT2x2s2(&Tcv1,  &TconvT, CV_model_23_proto_upsample, blob);
    conv2d(&TconvT,    &Tcv2,  CV_model_23_proto_cv2, blob);
    conv2d(&Tcv2,      &Toutput1, CV_model_23_proto_cv3, blob);
}
