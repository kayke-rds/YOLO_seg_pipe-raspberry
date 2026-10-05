/* kernels.c -- primitivas de calculo. */
#include "yolo26n_seg.h"

static inline float silu_f(float x){ return x / (1.0f + expf(-x)); }

/* Conv2d + bias + (SiLU opcional). Cobre 1x1, 3x3, 3x3s2 e DW. */
void conv2d(const Tensor *in, Tensor *out, ConvId id, const float *blob)
{
    const ConvDesc *d = &YOLO_CONV[id];
    const float *W = CONV_W(blob, id);
    const float *B = CONV_B(blob, id);
    const int Cin = d->cin, Cout = d->cout;
    const int K = d->k, S = d->stride, P = d->pad, G = d->groups;
    const int H = out->h, W_ = out->w;
    const int OH = in->h, OW = in->w;
    const int Cin_g = Cin / G, Cout_g = Cout / G;
    const int HW = H * W_, OHW = OH * OW;

    /* bias init */
    for (int co = 0; co < Cout; co++){
        float *o = out->data + co * HW;
        float b = B[co];
        for (int i = 0; i < HW; i++) o[i] = b;
    }

    /* 1x1 rapida (K==1) */
    if (K == 1 && S == 1 && P == 0 && G == 1) {
        for (int co = 0; co < Cout; co++){
            float *o = out->data + co * HW;
            const float *w = W + co * Cin;
            for (int ci = 0; ci < Cin; ci++){
                const float *ip = in->data + ci * OHW;
                float wv = w[ci];
                for (int i = 0; i < HW; i++) o[i] += wv * ip[i];
            }
        }
    } else {
        for (int g = 0; g < G; g++){
            int ci_base = g * Cin_g;
            int co_base = g * Cout_g;
            for (int co = 0; co < Cout_g; co++){
                float *o = out->data + (co_base + co) * HW;
                const float *wb = W + (co_base + co) * Cin_g * K * K;
                for (int ci = 0; ci < Cin_g; ci++){
                    const float *ip   = in->data + (ci_base + ci) * OHW;
                    const float *wp   = wb + ci * K * K;
                    for (int ky = 0; ky < K; ky++){
                        int iy0 = ky - P;
                        /* limites válidos de y para este ky */
                        int y_start = 0;
                        int y_end   = H;
                        if (S == 1){
                            if (iy0       < 0) y_start = -iy0;
                            if (iy0 + H   > OH) y_end = OH - iy0;
                        } else {
                            /* y tal que 0 <= iy0 + y*S < OH */
                            int lo = (-iy0 + S - 1) / S;
                            if (lo > y_start) y_start = lo;
                            int hi = (OH - iy0 + S - 1) / S;
                            if (hi < y_end) y_end = hi;
                        }
                        if (y_start >= y_end) continue;
                        for (int kx = 0; kx < K; kx++){
                            float wv = wp[ky*K + kx];
                            if (wv == 0.0f) continue;
                            int ix0 = kx - P;
                            int x_start = 0;
                            int x_end   = W_;
                            if (S == 1){
                                if (ix0       < 0) x_start = -ix0;
                                if (ix0 + W_  > OW) x_end = OW - ix0;
                            } else {
                                int lo = (-ix0 + S - 1) / S;
                                if (lo > x_start) x_start = lo;
                                int hi = (OW - ix0 + S - 1) / S;
                                if (hi < x_end) x_end = hi;
                            }
                            if (x_start >= x_end) continue;
                            int len = x_end - x_start;
                            for (int y = y_start; y < y_end; y++){
                                float *orow = o + y * W_;
                                const float *irow = ip + (iy0 + y*S)*OW + ix0 + x_start*S;
                                if (S == 1){
                                    for (int x = 0; x < len; x++)
                                        orow[x_start + x] += wv * irow[x];
                                } else {
                                    for (int x = 0; x < len; x++)
                                        orow[x_start + x] += wv * irow[x * S];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (d->act == ACT_SILU){
        int N = Cout * HW;
        for (int i = 0; i < N; i++) out->data[i] = silu_f(out->data[i]);
    }
}

/* ConvTranspose 2x2 stride 2, sem overlap. W:[Cin][Cout][2][2], B:[Cout]. */
void convT2x2s2(const Tensor *in, Tensor *out, ConvId id, const float *blob)
{
    const ConvDesc *d = &YOLO_CONV[id];
    const float *W = CONV_W(blob, id);
    const float *B = CONV_B(blob, id);
    const int Cin = d->cin, Cout = d->cout;
    const int OH = in->h, OW = in->w;

    for (int co = 0; co < Cout; co++){
        float b = B[co];
        for (int y = 0; y < OH; y++){
            for (int x = 0; x < OW; x++){
                float a00 = b, a01 = b, a10 = b, a11 = b;
                for (int ci = 0; ci < Cin; ci++){
                    float v = in->data[ci * OH*OW + y*OW + x];
                    const float *w = W + ((ci*Cout + co) * 4);
                    a00 += v * w[0];
                    a01 += v * w[1];
                    a10 += v * w[2];
                    a11 += v * w[3];
                }
                int oy = y*2, ox = x*2;
                out->data[co*out->h*out->w + (oy  )*out->w + (ox  )] = a00;
                out->data[co*out->h*out->w + (oy  )*out->w + (ox+1)] = a01;
                out->data[co*out->h*out->w + (oy+1)*out->w + (ox  )] = a10;
                out->data[co*out->h*out->w + (oy+1)*out->w + (ox+1)] = a11;
            }
        }
    }
}

/* MaxPool 5x5 stride 1 pad 2. Padding = -inf. */
void maxpool5s1p2(const Tensor *in, Tensor *out)
{
    int C = in->c, H = in->h, W = in->w;
    for (int c = 0; c < C; c++){
        const float *ip = in->data + c * H * W;
        float *op = out->data + c * H * W;
        for (int y = 0; y < H; y++){
            for (int x = 0; x < W; x++){
                float m = -INFINITY;
                for (int ky = 0; ky < 5; ky++){
                    int iy = y - 2 + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < 5; kx++){
                        int ix = x - 2 + kx;
                        if (ix < 0 || ix >= W) continue;
                        float v = ip[iy*W + ix];
                        if (v > m) m = v;
                    }
                }
                op[y*W + x] = m;
            }
        }
    }
}

/* Resize nearest (asymmetric, floor). scale 2 ou 4. */
void resize_nearest(const Tensor *in, Tensor *out, int scale)
{
    int C = in->c, H = in->h, W = in->w;
    int OH = out->h, OW = out->w;
    for (int c = 0; c < C; c++){
        const float *ip = in->data + c * H * W;
        float *op = out->data + c * OH * OW;
        for (int y = 0; y < OH; y++){
            int iy = y / scale;
            for (int x = 0; x < OW; x++){
                int ix = x / scale;
                op[y*OW + x] = ip[iy*W + ix];
            }
        }
    }
}

void add_tensor(const Tensor *a, const Tensor *b, Tensor *out)
{
    int N = tsz(a);
    for (int i = 0; i < N; i++) out->data[i] = a->data[i] + b->data[i];
}
void add_inplace(Tensor *a, const Tensor *b)
{
    int N = tsz(a);
    for (int i = 0; i < N; i++) a->data[i] += b->data[i];
}

void concat_ch(const Tensor *const *ins, int n, Tensor *out)
{
    int HW = out->h * out->w;
    int c = 0;
    for (int i = 0; i < n; i++){
        int ci = ins[i]->c;
        memcpy(out->data + c*HW, ins[i]->data, (size_t)ci*HW*sizeof(float));
        c += ci;
    }
}

/* Concat no ultimo eixo (W). Cada canal contribui com um segmento. */
void concat_w(const Tensor *const *ins, int n, Tensor *out)
{
    int C       = out->c;
    int HW_out  = out->h * out->w;
    int offset  = 0;
    for (int i = 0; i < n; i++) {
        int HW_i = ins[i]->h * ins[i]->w;
        for (int c = 0; c < C; c++) {
            memcpy(out->data + (size_t)c * HW_out + offset,
                   ins[i]->data + (size_t)c * HW_i,
                   (size_t)HW_i * sizeof(float));
        }
        offset += HW_i;
    }
}

void silu_tensor(Tensor *t){
    int N = tsz(t);
    for (int i = 0; i < N; i++) t->data[i] = silu_f(t->data[i]);
}
void sigmoid_tensor(Tensor *t){
    int N = tsz(t);
    for (int i = 0; i < N; i++) t->data[i] = 1.0f/(1.0f + expf(-t->data[i]));
}
void mul_scalar(Tensor *t, float s){
    int N = tsz(t);
    for (int i = 0; i < N; i++) t->data[i] *= s;
}

/* Softmax sobre a ultima dimensao (w). rows = c*h, cols = w. */
void softmax_last(Tensor *t){
    int rows = t->c * t->h, cols = t->w;
    for (int r = 0; r < rows; r++){
        float *row = t->data + r*cols;
        float m = row[0];
        for (int i = 1; i < cols; i++) if (row[i] > m) m = row[i];
        float s = 0.f;
        for (int i = 0; i < cols; i++){ row[i] = expf(row[i]-m); s += row[i]; }
        float inv = 1.0f/s;
        for (int i = 0; i < cols; i++) row[i] *= inv;
    }
}

/* C = A @ B, A:[batch,M,K], B:[batch,K,N], C:[batch,M,N] */
void matmul_batched(const float *A, const float *B, float *C, int batch, int M, int K, int N)
{
    for (int b = 0; b < batch; b++){
        const float *Ab = A + b*M*K;
        const float *Bb = B + b*K*N;
        float *Cb = C + b*M*N;
        for (int m = 0; m < M; m++){
            float *Crow = Cb + m*N;
            const float *Arow = Ab + m*K;
            for (int n = 0; n < N; n++) Crow[n] = 0.f;
            for (int k = 0; k < K; k++){
                float a = Arow[k];
                if (a == 0.f) continue;
                const float *Brow = Bb + k*N;
                for (int n = 0; n < N; n++) Crow[n] += a * Brow[n];
            }
        }
    }
}

/* C = A @ B^T, A:[batch,M,K], B:[batch,N,K], C:[batch,M,N] */
void matmul_nt(const float *A, const float *B, float *C, int batch, int M, int K, int N)
{
    for (int b = 0; b < batch; b++){
        const float *Ab = A + b*M*K;
        const float *Bb = B + b*N*K;
        float *Cb = C + b*M*N;
        for (int m = 0; m < M; m++){
            const float *Arow = Ab + m*K;
            float *Crow = Cb + m*N;
            for (int n = 0; n < N; n++){
                const float *Brow = Bb + n*K;
                float s = 0.f;
                for (int k = 0; k < K; k++) s += Arow[k] * Brow[k];
                Crow[n] = s;
            }
        }
    }
}

/* TopK descendente simples (k << n). */
void topk_desc(const float *x, int n, int k, float *val, int *idx)
{
    static uint8_t used[N_ANCHORS];   /* n <= N_ANCHORS */
    memset(used, 0, n);
    for (int t = 0; t < k; t++) {
        int best = -1;
        float bv = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            if (x[i] > bv || (x[i] == bv && i < best)) {  /* desempate por menor indice */
                bv = x[i];
                best = i;
            }
        }
        val[t] = bv; idx[t] = best;
        used[best] = 1;
    }
}
