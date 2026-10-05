#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <stdint.h>

/* Metadados do letterbox: necessarios para reverter coordenadas depois
 * e para aplicar o MESMO letterbox na mascara GT. */
typedef struct {
    float r;           /* fator de escala (new / old)           */
    int   dw, dh;      /* padding horizontal e vertical (pixels) */
    int   new_h, new_w;/* dimensoes apos resize (sem padding)    */
} LetterboxInfo;

/* Letterbox da imagem RGB -> NCHW float [3, dst_h, dst_w], normalizado /255.
 * Preenche 'info' com r, dw, dh para uso posterior. */
void letterbox_image_rgb_to_nchw(const uint8_t *img, int src_h, int src_w,
                                 float *out_nchw, int dst_h, int dst_w,
                                 LetterboxInfo *info);

/* Letterbox da mascara GT -> uint8 [dst_h, dst_w] com valores 0/1.
 * Usa o MESMO r, dw, dh da imagem para alinhamento pixel-perfect.
 * Interpolacao NEAREST (preserva natureza binaria).
 * Padding = 0 (fundo). */
void letterbox_mask_nearest(const uint8_t *mask, int src_h, int src_w,
                            uint8_t *out, int dst_h, int dst_w,
                            const LetterboxInfo *info);

#endif
