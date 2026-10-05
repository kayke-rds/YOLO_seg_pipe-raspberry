#ifndef POSTPROCESS_H
#define POSTPROCESS_H
#include <stdint.h>

/* Decodifica a mascara de UMA deteccao (indice det) para o tamanho da
 * imagem (320x320 letterboxed).
 *
 * Pipeline (equivalente a Ultralytics process_mask_native):
 *   1. masks = sigmoid(coeffs @ proto_flat)   -> [80, 80]
 *   2. bilinear upsample -> [img_h, img_w]
 *   3. crop_mask: zera fora da bbox [x1,y1,x2,y2]
 *   4. binariza > 0.5 -> uint8 (0/1)
 *
 * proto:    yolo_output1() [32, 80, 80]
 * output0:  yolo_output0() [300, 38]  (x1,y1,x2,y2,score,cls,coef[32])
 * det:      indice da deteccao (0 = maior score)
 * out_mask: buffer pre-alocado [img_h * img_w] uint8
 */
void decode_mask_for_detection(const float *proto, const float *output0,
                               int det, int img_h, int img_w,
                               uint8_t *out_mask);

/* Metricas de comparacao entre mascaras binarias (0/1). */
float mask_iou(const uint8_t *a, const uint8_t *b, int n);
float mask_dice(const uint8_t *a, const uint8_t *b, int n);
void  mask_precision_recall(const uint8_t *pred, const uint8_t *gt, int n,
                            float *precision, float *recall);

/* Gera imagem RGB de comparacao:
 *   TP (acerto)   -> verde
 *   FP (falso pos)-> vermelho
 *   FN (falso neg)-> azul
 *   TN (acerto)   -> preto
 */
void visualize_comparison(const uint8_t *pred, const uint8_t *gt,
                          int w, int h, uint8_t *rgb_out);

#endif
