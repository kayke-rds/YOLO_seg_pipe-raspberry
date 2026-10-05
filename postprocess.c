/* postprocess.c -- Decode de mascara e comparacao com ground truth.
 *
 * Segue a implementacao oficial da Ultralytics:
 *   process_mask_native(protos, masks_in, bboxes, shape)
 *
 * Referencias:
 *   - ultralytics/utils/ops.py (process_mask_native, crop_mask)
 *   - YOLOv5 utils/segment/general.py
 */
#include "postprocess.h"
#include <math.h>
#include <string.h>

void decode_mask_for_detection(const float *proto, const float *output0,
                               int det, int img_h, int img_w,
                               uint8_t *out_mask)
{
    const int PROTO_C = 32, PROTO_H = 80, PROTO_W = 80;

    const float *row = output0 + (size_t)det * 38;
    const float *coeffs = row + 6;   /* 32 coeficientes */

    /* --- 1) Combinacao linear + sigmoid em 80x80 --- */
    static float mask80[80 * 80];
    for (int i = 0; i < PROTO_H * PROTO_W; i++)
        mask80[i] = 0.0f;

    for (int k = 0; k < PROTO_C; k++) {
        float c = coeffs[k];
        if (c == 0.0f) continue;
        const float *p = proto + (size_t)k * PROTO_H * PROTO_W;
        for (int i = 0; i < PROTO_H * PROTO_W; i++)
            mask80[i] += c * p[i];
    }
    for (int i = 0; i < PROTO_H * PROTO_W; i++)
        mask80[i] = 1.0f / (1.0f + expf(-mask80[i]));

    /* --- 2) Upsample bilinear para img_h x img_w (align_corners=false) --- */
    float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
    float scale_y = (float)PROTO_H / (float)img_h;
    float scale_x = (float)PROTO_W / (float)img_w;

    for (int y = 0; y < img_h; y++) {
        float src_y = ((float)y + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(src_y);
        float dy = src_y - (float)y0;
        int y1i = y0 + 1;
        if (y0 < 0) { y0 = 0; dy = 0.0f; y1i = 0; }
        if (y1i >= PROTO_H) { y1i = PROTO_H - 1; dy = 0.0f; }

        for (int x = 0; x < img_w; x++) {
            float src_x = ((float)x + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(src_x);
            float dx = src_x - (float)x0;
            int x1i = x0 + 1;
            if (x0 < 0) { x0 = 0; dx = 0.0f; x1i = 0; }
            if (x1i >= PROTO_W) { x1i = PROTO_W - 1; dx = 0.0f; }

            float v00 = mask80[(size_t)y0 * PROTO_W + x0];
            float v01 = mask80[(size_t)y0 * PROTO_W + x1i];
            float v10 = mask80[(size_t)y1i * PROTO_W + x0];
            float v11 = mask80[(size_t)y1i * PROTO_W + x1i];
            float top = v00 * (1.0f - dx) + v01 * dx;
            float bot = v10 * (1.0f - dx) + v11 * dx;
            float val = top * (1.0f - dy) + bot * dy;

            /* --- 3) crop_mask: zera fora da bbox --- */
            if (x < x1 || x > x2 || y < y1 || y > y2)
                val = 0.0f;

            /* --- 4) Binarizacao --- */
            out_mask[(size_t)y * img_w + x] = (val > 0.5f) ? 1 : 0;
        }
    }
}

float mask_iou(const uint8_t *a, const uint8_t *b, int n) {
    int inter = 0, uni = 0;
    for (int i = 0; i < n; i++) {
        int va = a[i] != 0, vb = b[i] != 0;
        inter += (va && vb);
        uni   += (va || vb);
    }
    return uni == 0 ? 0.0f : (float)inter / (float)uni;
}

float mask_dice(const uint8_t *a, const uint8_t *b, int n) {
    int inter = 0, sum = 0;
    for (int i = 0; i < n; i++) {
        int va = a[i] != 0, vb = b[i] != 0;
        inter += (va && vb);
        sum   += va + vb;
    }
    return sum == 0 ? 1.0f : 2.0f * (float)inter / (float)sum;
}

void mask_precision_recall(const uint8_t *pred, const uint8_t *gt, int n,
                           float *precision, float *recall)
{
    int tp = 0, fp = 0, fn = 0;
    for (int i = 0; i < n; i++) {
        int p = pred[i] != 0, g = gt[i] != 0;
        if (p && g) tp++;
        else if (p && !g) fp++;
        else if (!p && g) fn++;
    }
    *precision = (tp + fp) > 0 ? (float)tp / (float)(tp + fp) : 0.0f;
    *recall    = (tp + fn) > 0 ? (float)tp / (float)(tp + fn) : 0.0f;
}

void visualize_comparison(const uint8_t *pred, const uint8_t *gt,
                          int w, int h, uint8_t *rgb_out)
{
    for (int i = 0; i < w * h; i++) {
        int p = pred[i] != 0, g = gt[i] != 0;
        uint8_t *px = rgb_out + (size_t)i * 3;
        if (p && g)      { px[0] =   0; px[1] = 255; px[2] =   0; } /* TP verde */
        else if (p && !g){ px[0] = 255; px[1] =   0; px[2] =   0; } /* FP vermelho */
        else if (!p && g){ px[0] =   0; px[1] =   0; px[2] = 255; } /* FN azul */
        else             { px[0] =   0; px[1] =   0; px[2] =   0; } /* TN preto */
    }
}
