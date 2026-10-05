/* preprocess.c -- Letterbox para imagem e mascara GT.
 *
 * Reproduz o pipeline padrao da Ultralytics:
 *   r = min(dst_h/src_h, dst_w/src_w)
 *   new_h = round(src_h * r), new_w = round(src_w * r)
 *   dw = (dst_w - new_w) / 2,  dh = (dst_h - new_h) / 2
 *   padding cinza (114,114,114) centralizado
 *
 * Para a mascara GT, usa-se o mesmo r/dw/dh e padding 0.
 * A interpolacao da mascara e NEAREST (nao bilinear) para
 * nao criar valores fracionarios que destruiriam a binariedade.
 */
#include "preprocess.h"
#include <math.h>
#include <string.h>

void letterbox_image_rgb_to_nchw(const uint8_t *img, int src_h, int src_w,
                                 float *out_nchw, int dst_h, int dst_w,
                                 LetterboxInfo *info)
{
    float r = fminf((float)dst_h / (float)src_h,
                    (float)dst_w / (float)src_w);
    int new_h = (int)roundf((float)src_h * r);
    int new_w = (int)roundf((float)src_w * r);
    int dw = (dst_w - new_w) / 2;
    int dh = (dst_h - new_h) / 2;

    /* Preenche todo o tensor NCHW com o valor de padding (114/255) */
    const float pad = 114.0f / 255.0f;
    for (int i = 0; i < 3 * dst_h * dst_w; i++)
        out_nchw[i] = pad;

    /* Resize bilinear da imagem original para new_h x new_w
     * e escreve no offset (dh, dw) de cada canal. */
    for (int y = 0; y < new_h; y++) {
        float sy = (y + 0.5f) * (float)src_h / (float)new_h - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) { y0 = 0; fy = 0.0f; y1 = 0; }
        if (y1 >= src_h) { y1 = src_h - 1; fy = 0.0f; }

        for (int x = 0; x < new_w; x++) {
            float sx = (x + 0.5f) * (float)src_w / (float)new_w - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) { x0 = 0; fx = 0.0f; x1 = 0; }
            if (x1 >= src_w) { x1 = src_w - 1; fx = 0.0f; }

            int dy = y + dh;
            int dx = x + dw;

            for (int c = 0; c < 3; c++) {
                float v00 = img[((size_t)y0 * src_w + x0) * 3 + c];
                float v01 = img[((size_t)y0 * src_w + x1) * 3 + c];
                float v10 = img[((size_t)y1 * src_w + x0) * 3 + c];
                float v11 = img[((size_t)y1 * src_w + x1) * 3 + c];
                float top = v00 * (1.0f - fx) + v01 * fx;
                float bot = v10 * (1.0f - fx) + v11 * fx;
                float v = (top * (1.0f - fy) + bot * fy) / 255.0f;
                out_nchw[(size_t)c * dst_h * dst_w + (size_t)dy * dst_w + dx] = v;
            }
        }
    }

    info->r = r;
    info->dw = dw;
    info->dh = dh;
    info->new_h = new_h;
    info->new_w = new_w;
}

void letterbox_mask_nearest(const uint8_t *mask, int src_h, int src_w,
                            uint8_t *out, int dst_h, int dst_w,
                            const LetterboxInfo *info)
{
    /* Padding = fundo (0) */
    memset(out, 0, (size_t)dst_h * dst_w);

    /* Nearest: para cada pixel de destino, mapeia de volta para a origem
     * via sy = y / r. Isso e o inverso exato do letterbox bilinear. */
    for (int y = 0; y < info->new_h; y++) {
        int sy = (int)((float)y / info->r);
        if (sy >= src_h) sy = src_h - 1;
        if (sy < 0) sy = 0;

        for (int x = 0; x < info->new_w; x++) {
            int sx = (int)((float)x / info->r);
            if (sx >= src_w) sx = src_w - 1;
            if (sx < 0) sx = 0;

            /* Binariza: qualquer valor > 127 vira 1 */
            uint8_t v = (mask[(size_t)sy * src_w + sx] > 127) ? 1 : 0;
            out[(size_t)(y + info->dh) * dst_w + (x + info->dw)] = v;
        }
    }
}
