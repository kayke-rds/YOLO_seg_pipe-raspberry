/* yolo26n_seg_mem.h -- GERADO por plan_memory.py. NAO EDITAR.
 * Mapa de memoria estatica para a arena de ativacoes do YOLO26n-seg.
 * Offsets em floats (4 bytes). Arena alinhada a 64 B. */
#ifndef YOLO26N_SEG_MEM_H
#define YOLO26N_SEG_MEM_H
#include <stdint.h>

#define YOLO_ARENA_FLOATS  2837312u
#define YOLO_ARENA_BYTES   11349248u

typedef enum {
    T_images = 0,
    T__model_0_act_Mul_output_0 = 1,
    T__model_1_act_Mul_output_0 = 2,
    T__model_2_cv1_act_Mul_output_0 = 3,
    T__model_2_Split_output_0 = 4,
    T__model_2_Split_output_1 = 5,
    T__model_2_m_0_cv1_act_Mul_output_0 = 6,
    T__model_2_m_0_cv2_act_Mul_output_0 = 7,
    T__model_2_m_0_Add_output_0 = 8,
    T__model_2_Concat_output_0 = 9,
    T__model_2_cv2_act_Mul_output_0 = 10,
    T__model_3_act_Mul_output_0 = 11,
    T__model_4_cv1_act_Mul_output_0 = 12,
    T__model_4_Split_output_0 = 13,
    T__model_4_Split_output_1 = 14,
    T__model_4_m_0_cv1_act_Mul_output_0 = 15,
    T__model_4_m_0_cv2_act_Mul_output_0 = 16,
    T__model_4_m_0_Add_output_0 = 17,
    T__model_4_Concat_output_0 = 18,
    T__model_4_cv2_act_Mul_output_0 = 19,
    T__model_5_act_Mul_output_0 = 20,
    T__model_6_cv1_act_Mul_output_0 = 21,
    T__model_6_Split_output_0 = 22,
    T__model_6_Split_output_1 = 23,
    T__model_6_m_0_cv1_act_Mul_output_0 = 24,
    T__model_6_m_0_cv2_act_Mul_output_0 = 25,
    T__model_6_m_0_m_m_0_cv1_act_Mul_output_0 = 26,
    T__model_6_m_0_m_m_0_cv2_act_Mul_output_0 = 27,
    T__model_6_m_0_m_m_0_Add_output_0 = 28,
    T__model_6_m_0_m_m_1_cv1_act_Mul_output_0 = 29,
    T__model_6_m_0_m_m_1_cv2_act_Mul_output_0 = 30,
    T__model_6_m_0_m_m_1_Add_output_0 = 31,
    T__model_6_m_0_Concat_output_0 = 32,
    T__model_6_m_0_cv3_act_Mul_output_0 = 33,
    T__model_6_Concat_output_0 = 34,
    T__model_6_cv2_act_Mul_output_0 = 35,
    T__model_7_act_Mul_output_0 = 36,
    T__model_8_cv1_act_Mul_output_0 = 37,
    T__model_8_Split_output_0 = 38,
    T__model_8_Split_output_1 = 39,
    T__model_8_m_0_cv1_act_Mul_output_0 = 40,
    T__model_8_m_0_cv2_act_Mul_output_0 = 41,
    T__model_8_m_0_m_m_0_cv1_act_Mul_output_0 = 42,
    T__model_8_m_0_m_m_0_cv2_act_Mul_output_0 = 43,
    T__model_8_m_0_m_m_0_Add_output_0 = 44,
    T__model_8_m_0_m_m_1_cv1_act_Mul_output_0 = 45,
    T__model_8_m_0_m_m_1_cv2_act_Mul_output_0 = 46,
    T__model_8_m_0_m_m_1_Add_output_0 = 47,
    T__model_8_m_0_Concat_output_0 = 48,
    T__model_8_m_0_cv3_act_Mul_output_0 = 49,
    T__model_8_Concat_output_0 = 50,
    T__model_8_cv2_act_Mul_output_0 = 51,
    T__model_9_cv1_conv_Conv_output_0 = 52,
    T__model_9_m_MaxPool_output_0 = 53,
    T__model_9_m_1_MaxPool_output_0 = 54,
    T__model_9_m_2_MaxPool_output_0 = 55,
    T__model_9_Concat_output_0 = 56,
    T__model_9_cv2_act_Mul_output_0 = 57,
    T__model_9_Add_output_0 = 58,
    T__model_10_cv1_act_Mul_output_0 = 59,
    T__model_10_Split_output_0 = 60,
    T__model_10_Split_output_1 = 61,
    T__model_10_m_m_0_attn_qkv_conv_Conv_output_0 = 62,
    T__model_10_m_m_0_attn_Reshape_output_0 = 63,
    T__model_10_m_m_0_attn_Split_output_0 = 64,
    T__model_10_m_m_0_attn_Split_output_1 = 65,
    T__model_10_m_m_0_attn_Split_output_2 = 66,
    T__model_10_m_m_0_attn_Mul_output_0 = 67,
    T__model_10_m_m_0_attn_Reshape_2_output_0 = 68,
    T__model_10_m_m_0_attn_Transpose_output_0 = 69,
    T__model_10_m_m_0_attn_pe_conv_Conv_output_0 = 70,
    T__model_10_m_m_0_attn_MatMul_output_0 = 71,
    T__model_10_m_m_0_attn_Softmax_output_0 = 72,
    T__model_10_m_m_0_attn_Transpose_1_output_0 = 73,
    T__model_10_m_m_0_attn_MatMul_1_output_0 = 74,
    T__model_10_m_m_0_attn_Reshape_1_output_0 = 75,
    T__model_10_m_m_0_attn_Add_output_0 = 76,
    T__model_10_m_m_0_attn_proj_conv_Conv_output_0 = 77,
    T__model_10_m_m_0_Add_output_0 = 78,
    T__model_10_m_m_0_ffn_ffn_0_act_Mul_output_0 = 79,
    T__model_10_m_m_0_ffn_ffn_1_conv_Conv_output_0 = 80,
    T__model_10_m_m_0_Add_1_output_0 = 81,
    T__model_10_Concat_output_0 = 82,
    T__model_10_cv2_act_Mul_output_0 = 83,
    T__model_11_Resize_output_0 = 84,
    T__model_12_Concat_output_0 = 85,
    T__model_13_cv1_act_Mul_output_0 = 86,
    T__model_13_Split_output_0 = 87,
    T__model_13_Split_output_1 = 88,
    T__model_13_m_0_cv1_act_Mul_output_0 = 89,
    T__model_13_m_0_cv2_act_Mul_output_0 = 90,
    T__model_13_m_0_m_m_0_cv1_act_Mul_output_0 = 91,
    T__model_13_m_0_m_m_0_cv2_act_Mul_output_0 = 92,
    T__model_13_m_0_m_m_0_Add_output_0 = 93,
    T__model_13_m_0_m_m_1_cv1_act_Mul_output_0 = 94,
    T__model_13_m_0_m_m_1_cv2_act_Mul_output_0 = 95,
    T__model_13_m_0_m_m_1_Add_output_0 = 96,
    T__model_13_m_0_Concat_output_0 = 97,
    T__model_13_m_0_cv3_act_Mul_output_0 = 98,
    T__model_13_Concat_output_0 = 99,
    T__model_13_cv2_act_Mul_output_0 = 100,
    T__model_14_Resize_output_0 = 101,
    T__model_15_Concat_output_0 = 102,
    T__model_16_cv1_act_Mul_output_0 = 103,
    T__model_16_Split_output_0 = 104,
    T__model_16_Split_output_1 = 105,
    T__model_16_m_0_cv1_act_Mul_output_0 = 106,
    T__model_16_m_0_cv2_act_Mul_output_0 = 107,
    T__model_16_m_0_m_m_0_cv1_act_Mul_output_0 = 108,
    T__model_16_m_0_m_m_0_cv2_act_Mul_output_0 = 109,
    T__model_16_m_0_m_m_0_Add_output_0 = 110,
    T__model_16_m_0_m_m_1_cv1_act_Mul_output_0 = 111,
    T__model_16_m_0_m_m_1_cv2_act_Mul_output_0 = 112,
    T__model_16_m_0_m_m_1_Add_output_0 = 113,
    T__model_16_m_0_Concat_output_0 = 114,
    T__model_16_m_0_cv3_act_Mul_output_0 = 115,
    T__model_16_Concat_output_0 = 116,
    T__model_16_cv2_act_Mul_output_0 = 117,
    T__model_17_act_Mul_output_0 = 118,
    T__model_23_one2one_cv2_0_one2one_cv2_0_0_act_Mul_output_0 = 119,
    T__model_23_one2one_cv3_0_one2one_cv3_0_0_one2one_cv3_0_0_0_act_Mul_output_0 = 120,
    T__model_23_one2one_cv4_0_one2one_cv4_0_0_act_Mul_output_0 = 121,
    T__model_18_Concat_output_0 = 122,
    T__model_23_one2one_cv2_0_one2one_cv2_0_1_act_Mul_output_0 = 123,
    T__model_23_one2one_cv3_0_one2one_cv3_0_0_one2one_cv3_0_0_1_act_Mul_output_0 = 124,
    T__model_23_one2one_cv4_0_one2one_cv4_0_1_act_Mul_output_0 = 125,
    T__model_19_cv1_act_Mul_output_0 = 126,
    T__model_23_one2one_cv2_0_one2one_cv2_0_2_Conv_output_0 = 127,
    T__model_23_one2one_cv3_0_one2one_cv3_0_1_one2one_cv3_0_1_0_act_Mul_output_0 = 128,
    T__model_23_one2one_cv4_0_one2one_cv4_0_2_Conv_output_0 = 129,
    T__model_19_Split_output_0 = 130,
    T__model_19_Split_output_1 = 131,
    T__model_23_Reshape_output_0 = 132,
    T__model_23_Reshape_6_output_0 = 133,
    T__model_19_m_0_cv1_act_Mul_output_0 = 134,
    T__model_19_m_0_cv2_act_Mul_output_0 = 135,
    T__model_23_one2one_cv3_0_one2one_cv3_0_1_one2one_cv3_0_1_1_act_Mul_output_0 = 136,
    T__model_19_m_0_m_m_0_cv1_act_Mul_output_0 = 137,
    T__model_23_one2one_cv3_0_one2one_cv3_0_2_Conv_output_0 = 138,
    T__model_23_Reshape_3_output_0 = 139,
    T__model_19_m_0_m_m_0_cv2_act_Mul_output_0 = 140,
    T__model_19_m_0_m_m_0_Add_output_0 = 141,
    T__model_19_m_0_m_m_1_cv1_act_Mul_output_0 = 142,
    T__model_19_m_0_m_m_1_cv2_act_Mul_output_0 = 143,
    T__model_19_m_0_m_m_1_Add_output_0 = 144,
    T__model_19_m_0_Concat_output_0 = 145,
    T__model_19_m_0_cv3_act_Mul_output_0 = 146,
    T__model_19_Concat_output_0 = 147,
    T__model_19_cv2_act_Mul_output_0 = 148,
    T__model_20_act_Mul_output_0 = 149,
    T__model_23_one2one_cv2_1_one2one_cv2_1_0_act_Mul_output_0 = 150,
    T__model_23_one2one_cv3_1_one2one_cv3_1_0_one2one_cv3_1_0_0_act_Mul_output_0 = 151,
    T__model_23_one2one_cv4_1_one2one_cv4_1_0_act_Mul_output_0 = 152,
    T__model_23_proto_feat_refine_0_act_Mul_output_0 = 153,
    T__model_21_Concat_output_0 = 154,
    T__model_23_one2one_cv2_1_one2one_cv2_1_1_act_Mul_output_0 = 155,
    T__model_23_one2one_cv3_1_one2one_cv3_1_0_one2one_cv3_1_0_1_act_Mul_output_0 = 156,
    T__model_23_one2one_cv4_1_one2one_cv4_1_1_act_Mul_output_0 = 157,
    T__model_23_proto_Resize_output_0 = 158,
    T__model_22_cv1_act_Mul_output_0 = 159,
    T__model_23_proto_Add_output_0 = 160,
    T__model_23_one2one_cv2_1_one2one_cv2_1_2_Conv_output_0 = 161,
    T__model_23_one2one_cv3_1_one2one_cv3_1_1_one2one_cv3_1_1_0_act_Mul_output_0 = 162,
    T__model_23_one2one_cv4_1_one2one_cv4_1_2_Conv_output_0 = 163,
    T__model_22_Split_output_0 = 164,
    T__model_22_Split_output_1 = 165,
    T__model_23_Reshape_1_output_0 = 166,
    T__model_23_Reshape_7_output_0 = 167,
    T__model_22_m_0_m_0_0_cv1_act_Mul_output_0 = 168,
    T__model_23_one2one_cv3_1_one2one_cv3_1_1_one2one_cv3_1_1_1_act_Mul_output_0 = 169,
    T__model_22_m_0_m_0_0_cv2_act_Mul_output_0 = 170,
    T__model_23_one2one_cv3_1_one2one_cv3_1_2_Conv_output_0 = 171,
    T__model_23_Reshape_4_output_0 = 172,
    T__model_22_m_0_m_0_0_Add_output_0 = 173,
    T__model_22_m_0_m_0_1_attn_qkv_conv_Conv_output_0 = 174,
    T__model_22_m_0_m_0_1_attn_Reshape_output_0 = 175,
    T__model_22_m_0_m_0_1_attn_Split_output_0 = 176,
    T__model_22_m_0_m_0_1_attn_Split_output_1 = 177,
    T__model_22_m_0_m_0_1_attn_Split_output_2 = 178,
    T__model_22_m_0_m_0_1_attn_Mul_output_0 = 179,
    T__model_22_m_0_m_0_1_attn_Reshape_2_output_0 = 180,
    T__model_22_m_0_m_0_1_attn_Transpose_output_0 = 181,
    T__model_22_m_0_m_0_1_attn_pe_conv_Conv_output_0 = 182,
    T__model_22_m_0_m_0_1_attn_MatMul_output_0 = 183,
    T__model_22_m_0_m_0_1_attn_Softmax_output_0 = 184,
    T__model_22_m_0_m_0_1_attn_Transpose_1_output_0 = 185,
    T__model_22_m_0_m_0_1_attn_MatMul_1_output_0 = 186,
    T__model_22_m_0_m_0_1_attn_Reshape_1_output_0 = 187,
    T__model_22_m_0_m_0_1_attn_Add_output_0 = 188,
    T__model_22_m_0_m_0_1_attn_proj_conv_Conv_output_0 = 189,
    T__model_22_m_0_m_0_1_Add_output_0 = 190,
    T__model_22_m_0_m_0_1_ffn_ffn_0_act_Mul_output_0 = 191,
    T__model_22_m_0_m_0_1_ffn_ffn_1_conv_Conv_output_0 = 192,
    T__model_22_m_0_m_0_1_Add_1_output_0 = 193,
    T__model_22_Concat_output_0 = 194,
    T__model_22_cv2_act_Mul_output_0 = 195,
    T__model_23_one2one_cv2_2_one2one_cv2_2_0_act_Mul_output_0 = 196,
    T__model_23_one2one_cv3_2_one2one_cv3_2_0_one2one_cv3_2_0_0_act_Mul_output_0 = 197,
    T__model_23_one2one_cv4_2_one2one_cv4_2_0_act_Mul_output_0 = 198,
    T__model_23_proto_feat_refine_1_act_Mul_output_0 = 199,
    T__model_23_one2one_cv2_2_one2one_cv2_2_1_act_Mul_output_0 = 200,
    T__model_23_one2one_cv3_2_one2one_cv3_2_0_one2one_cv3_2_0_1_act_Mul_output_0 = 201,
    T__model_23_one2one_cv4_2_one2one_cv4_2_1_act_Mul_output_0 = 202,
    T__model_23_proto_Resize_1_output_0 = 203,
    T__model_23_proto_Add_1_output_0 = 204,
    T__model_23_proto_feat_fuse_act_Mul_output_0 = 205,
    T__model_23_one2one_cv2_2_one2one_cv2_2_2_Conv_output_0 = 206,
    T__model_23_one2one_cv3_2_one2one_cv3_2_1_one2one_cv3_2_1_0_act_Mul_output_0 = 207,
    T__model_23_one2one_cv4_2_one2one_cv4_2_2_Conv_output_0 = 208,
    T__model_23_Reshape_2_output_0 = 209,
    T__model_23_Reshape_8_output_0 = 210,
    T__model_23_Concat_output_0 = 211,
    T__model_23_Concat_2_output_0 = 212,
    T__model_23_proto_cv1_act_Mul_output_0 = 213,
    T__model_23_one2one_cv3_2_one2one_cv3_2_1_one2one_cv3_2_1_1_act_Mul_output_0 = 214,
    T__model_23_Slice_output_0 = 215,
    T__model_23_Slice_1_output_0 = 216,
    T__model_23_Sub_output_0 = 217,
    T__model_23_Add_1_output_0 = 218,
    T__model_23_proto_upsample_ConvTranspose_output_0 = 219,
    T__model_23_Concat_3_output_0 = 220,
    T__model_23_one2one_cv3_2_one2one_cv3_2_2_Conv_output_0 = 221,
    T__model_23_proto_cv2_act_Mul_output_0 = 222,
    T__model_23_Mul_2_output_0 = 223,
    T__model_23_Reshape_5_output_0 = 224,
    T__model_23_Concat_1_output_0 = 225,
    T__model_23_Sigmoid_output_0 = 226,
    T_output1 = 227,
    T__model_23_Concat_4_output_0 = 228,
    T__model_23_Transpose_output_0 = 229,
    T__model_23_Split_output_0 = 230,
    T__model_23_Split_output_1 = 231,
    T__model_23_Split_output_2 = 232,
    T__model_23_ReduceMax_output_0 = 233,
    T__model_23_TopK_output_0 = 234,
    T__model_23_TopK_output_1 = 235,
    T__model_23_Expand_output_0 = 236,
    T__model_23_GatherElements_output_0 = 237,
    T__model_23_Flatten_output_0 = 238,
    T__model_23_TopK_1_output_0 = 239,
    T__model_23_Cast_1_output_0 = 240,
    T__model_23_Unsqueeze_2_output_0 = 241,
    T__model_23_Mod_output_0 = 242,
    T__model_23_Unsqueeze_1_output_0 = 243,
    T__model_23_Unsqueeze_3_output_0 = 244,
    T__model_23_GatherElements_1_output_0 = 245,
    T__model_23_Cast_2_output_0 = 246,
    T__model_23_Expand_1_output_0 = 247,
    T__model_23_Expand_2_output_0 = 248,
    T__model_23_GatherElements_2_output_0 = 249,
    T__model_23_GatherElements_3_output_0 = 250,
    T_output0 = 251,
    T_COUNT
} TensorId;

typedef struct {
    uint32_t off;   /* offset em floats */
    uint32_t size;  /* tamanho em floats (sem padding) */
    uint16_t c, h, w; /* shape do tensor */
} TensorDesc;

static const TensorDesc YOLO_TENSOR[T_COUNT] = {
    {0u, 307200u, 3, 320, 320},  /* t0: 1x3x320x320 */
    {1105600u, 409600u, 16, 160, 160},  /* t1: 1x16x160x160 */
    {0u, 204800u, 32, 80, 80},  /* t2: 1x32x80x80 */
    {1105600u, 204800u, 32, 80, 80},  /* t3: 1x32x80x80 */
    {0u, 102400u, 16, 80, 80},  /* t4: 1x16x80x80 */
    {102400u, 102400u, 16, 80, 80},  /* t5: 1x16x80x80 */
    {204800u, 51200u, 8, 80, 80},  /* t6: 1x8x80x80 */
    {1310400u, 102400u, 16, 80, 80},  /* t7: 1x16x80x80 */
    {204800u, 102400u, 16, 80, 80},  /* t8: 1x16x80x80 */
    {1310400u, 307200u, 48, 80, 80},  /* t9: 1x48x80x80 */
    {1617600u, 409600u, 64, 80, 80},  /* t10: 1x64x80x80 */
    {0u, 102400u, 64, 40, 40},  /* t11: 1x64x40x40 */
    {102400u, 102400u, 64, 40, 40},  /* t12: 1x64x40x40 */
    {0u, 51200u, 32, 40, 40},  /* t13: 1x32x40x40 */
    {51200u, 51200u, 32, 40, 40},  /* t14: 1x32x40x40 */
    {204800u, 25600u, 16, 40, 40},  /* t15: 1x16x40x40 */
    {230400u, 51200u, 32, 40, 40},  /* t16: 1x32x40x40 */
    {1105600u, 51200u, 32, 40, 40},  /* t17: 1x32x40x40 */
    {1156800u, 153600u, 96, 40, 40},  /* t18: 1x96x40x40 */
    {0u, 204800u, 128, 40, 40},  /* t19: 1x128x40x40 */
    {204800u, 51200u, 128, 20, 20},  /* t20: 1x128x20x20 */
    {256000u, 51200u, 128, 20, 20},  /* t21: 1x128x20x20 */
    {204800u, 25600u, 64, 20, 20},  /* t22: 1x64x20x20 */
    {230400u, 25600u, 64, 20, 20},  /* t23: 1x64x20x20 */
    {1105600u, 12800u, 32, 20, 20},  /* t24: 1x32x20x20 */
    {1118400u, 12800u, 32, 20, 20},  /* t25: 1x32x20x20 */
    {1131200u, 12800u, 32, 20, 20},  /* t26: 1x32x20x20 */
    {1144000u, 12800u, 32, 20, 20},  /* t27: 1x32x20x20 */
    {1131200u, 12800u, 32, 20, 20},  /* t28: 1x32x20x20 */
    {1105600u, 12800u, 32, 20, 20},  /* t29: 1x32x20x20 */
    {1144000u, 12800u, 32, 20, 20},  /* t30: 1x32x20x20 */
    {1105600u, 12800u, 32, 20, 20},  /* t31: 1x32x20x20 */
    {1131200u, 25600u, 64, 20, 20},  /* t32: 1x64x20x20 */
    {1105600u, 25600u, 64, 20, 20},  /* t33: 1x64x20x20 */
    {1131200u, 76800u, 192, 20, 20},  /* t34: 1x192x20x20 */
    {204800u, 51200u, 128, 20, 20},  /* t35: 1x128x20x20 */
    {256000u, 25600u, 256, 10, 10},  /* t36: 1x256x10x10 */
    {281600u, 25600u, 256, 10, 10},  /* t37: 1x256x10x10 */
    {256000u, 12800u, 128, 10, 10},  /* t38: 1x128x10x10 */
    {268800u, 12800u, 128, 10, 10},  /* t39: 1x128x10x10 */
    {1105600u, 6400u, 64, 10, 10},  /* t40: 1x64x10x10 */
    {1112000u, 6400u, 64, 10, 10},  /* t41: 1x64x10x10 */
    {1118400u, 6400u, 64, 10, 10},  /* t42: 1x64x10x10 */
    {1124800u, 6400u, 64, 10, 10},  /* t43: 1x64x10x10 */
    {1118400u, 6400u, 64, 10, 10},  /* t44: 1x64x10x10 */
    {1105600u, 6400u, 64, 10, 10},  /* t45: 1x64x10x10 */
    {1124800u, 6400u, 64, 10, 10},  /* t46: 1x64x10x10 */
    {1105600u, 6400u, 64, 10, 10},  /* t47: 1x64x10x10 */
    {1118400u, 12800u, 128, 10, 10},  /* t48: 1x128x10x10 */
    {1105600u, 12800u, 128, 10, 10},  /* t49: 1x128x10x10 */
    {1118400u, 38400u, 384, 10, 10},  /* t50: 1x384x10x10 */
    {256000u, 25600u, 256, 10, 10},  /* t51: 1x256x10x10 */
    {281600u, 12800u, 128, 10, 10},  /* t52: 1x128x10x10 */
    {294400u, 12800u, 128, 10, 10},  /* t53: 1x128x10x10 */
    {1105600u, 12800u, 128, 10, 10},  /* t54: 1x128x10x10 */
    {1118400u, 12800u, 128, 10, 10},  /* t55: 1x128x10x10 */
    {1131200u, 51200u, 512, 10, 10},  /* t56: 1x512x10x10 */
    {281600u, 25600u, 256, 10, 10},  /* t57: 1x256x10x10 */
    {1105600u, 25600u, 256, 10, 10},  /* t58: 1x256x10x10 */
    {460800u, 25600u, 256, 10, 10},  /* t59: 1x256x10x10 */
    {256000u, 12800u, 128, 10, 10},  /* t60: 1x128x10x10 */
    {268800u, 12800u, 128, 10, 10},  /* t61: 1x128x10x10 */
    {486400u, 25600u, 256, 10, 10},  /* t62: 1x256x10x10 */
    {512000u, 25600u, 2, 128, 100},  /* t63: 1x2x128x100 */
    {1054400u, 6400u, 2, 32, 100},  /* t64: 1x2x32x100 */
    {1060800u, 6400u, 2, 32, 100},  /* t65: 1x2x32x100 */
    {785600u, 12800u, 2, 64, 100},  /* t66: 1x2x64x100 */
    {1067200u, 6400u, 2, 32, 100},  /* t67: 1x2x32x100 */
    {798400u, 12800u, 128, 10, 10},  /* t68: 1x128x10x10 */
    {1073600u, 6400u, 2, 100, 32},  /* t69: 1x2x100x32 */
    {811200u, 12800u, 128, 10, 10},  /* t70: 1x128x10x10 */
    {665600u, 20000u, 2, 100, 100},  /* t71: 1x2x100x100 */
    {685600u, 20000u, 2, 100, 100},  /* t72: 1x2x100x100 */
    {705600u, 20000u, 2, 100, 100},  /* t73: 1x2x100x100 */
    {824000u, 12800u, 2, 64, 100},  /* t74: 1x2x64x100 */
    {836800u, 12800u, 128, 10, 10},  /* t75: 1x128x10x10 */
    {849600u, 12800u, 128, 10, 10},  /* t76: 1x128x10x10 */
    {862400u, 12800u, 128, 10, 10},  /* t77: 1x128x10x10 */
    {875200u, 12800u, 128, 10, 10},  /* t78: 1x128x10x10 */
    {537600u, 25600u, 256, 10, 10},  /* t79: 1x256x10x10 */
    {888000u, 12800u, 128, 10, 10},  /* t80: 1x128x10x10 */
    {900800u, 12800u, 128, 10, 10},  /* t81: 1x128x10x10 */
    {268800u, 25600u, 256, 10, 10},  /* t82: 1x256x10x10 */
    {1105600u, 25600u, 256, 10, 10},  /* t83: 1x256x10x10 */
    {1131200u, 102400u, 256, 20, 20},  /* t84: 1x256x20x20 */
    {1233600u, 153600u, 384, 20, 20},  /* t85: 1x384x20x20 */
    {204800u, 51200u, 128, 20, 20},  /* t86: 1x128x20x20 */
    {256000u, 25600u, 64, 20, 20},  /* t87: 1x64x20x20 */
    {281600u, 25600u, 64, 20, 20},  /* t88: 1x64x20x20 */
    {1131200u, 12800u, 32, 20, 20},  /* t89: 1x32x20x20 */
    {1144000u, 12800u, 32, 20, 20},  /* t90: 1x32x20x20 */
    {1156800u, 12800u, 32, 20, 20},  /* t91: 1x32x20x20 */
    {1169600u, 12800u, 32, 20, 20},  /* t92: 1x32x20x20 */
    {1156800u, 12800u, 32, 20, 20},  /* t93: 1x32x20x20 */
    {1131200u, 12800u, 32, 20, 20},  /* t94: 1x32x20x20 */
    {1169600u, 12800u, 32, 20, 20},  /* t95: 1x32x20x20 */
    {1131200u, 12800u, 32, 20, 20},  /* t96: 1x32x20x20 */
    {1156800u, 25600u, 64, 20, 20},  /* t97: 1x64x20x20 */
    {1131200u, 25600u, 64, 20, 20},  /* t98: 1x64x20x20 */
    {1156800u, 76800u, 192, 20, 20},  /* t99: 1x192x20x20 */
    {204800u, 51200u, 128, 20, 20},  /* t100: 1x128x20x20 */
    {1131200u, 204800u, 128, 40, 40},  /* t101: 1x128x40x40 */
    {1336000u, 409600u, 256, 40, 40},  /* t102: 1x256x40x40 */
    {0u, 102400u, 64, 40, 40},  /* t103: 1x64x40x40 */
    {102400u, 51200u, 32, 40, 40},  /* t104: 1x32x40x40 */
    {153600u, 51200u, 32, 40, 40},  /* t105: 1x32x40x40 */
    {256000u, 25600u, 16, 40, 40},  /* t106: 1x16x40x40 */
    {281600u, 25600u, 16, 40, 40},  /* t107: 1x16x40x40 */
    {1131200u, 25600u, 16, 40, 40},  /* t108: 1x16x40x40 */
    {1156800u, 25600u, 16, 40, 40},  /* t109: 1x16x40x40 */
    {1131200u, 25600u, 16, 40, 40},  /* t110: 1x16x40x40 */
    {256000u, 25600u, 16, 40, 40},  /* t111: 1x16x40x40 */
    {1156800u, 25600u, 16, 40, 40},  /* t112: 1x16x40x40 */
    {256000u, 25600u, 16, 40, 40},  /* t113: 1x16x40x40 */
    {1131200u, 51200u, 32, 40, 40},  /* t114: 1x32x40x40 */
    {256000u, 51200u, 32, 40, 40},  /* t115: 1x32x40x40 */
    {1131200u, 153600u, 96, 40, 40},  /* t116: 1x96x40x40 */
    {307200u, 102400u, 64, 40, 40},  /* t117: 1x64x40x40 */
    {0u, 25600u, 64, 20, 20},  /* t118: 1x64x20x20 */
    {179200u, 25600u, 16, 40, 40},  /* t119: 1x16x40x40 */
    {25600u, 102400u, 64, 40, 40},  /* t120: 1x64x40x40 */
    {128000u, 51200u, 32, 40, 40},  /* t121: 1x32x40x40 */
    {1523200u, 76800u, 192, 20, 20},  /* t122: 1x192x20x20 */
    {1489600u, 25600u, 16, 40, 40},  /* t123: 1x16x40x40 */
    {1131200u, 102400u, 64, 40, 40},  /* t124: 1x64x40x40 */
    {256000u, 51200u, 32, 40, 40},  /* t125: 1x32x40x40 */
    {204800u, 51200u, 128, 20, 20},  /* t126: 1x128x20x20 */
    {1515200u, 6400u, 4, 40, 40},  /* t127: 1x4x40x40 */
    {1233600u, 102400u, 64, 40, 40},  /* t128: 1x64x40x40 */
    {1438400u, 51200u, 32, 40, 40},  /* t129: 1x32x40x40 */
    {0u, 25600u, 64, 20, 20},  /* t130: 1x64x20x20 */
    {1523200u, 25600u, 64, 20, 20},  /* t131: 1x64x20x20 */
    {1548800u, 6400u, 4, 1, 1600},  /* t132: 1x4x1600 */
    {1555200u, 51200u, 32, 1, 1600},  /* t133: 1x32x1600 */
    {1606400u, 12800u, 32, 20, 20},  /* t134: 1x32x20x20 */
    {1619200u, 12800u, 32, 20, 20},  /* t135: 1x32x20x20 */
    {1336000u, 102400u, 64, 40, 40},  /* t136: 1x64x40x40 */
    {1632000u, 12800u, 32, 20, 20},  /* t137: 1x32x20x20 */
    {1521600u, 1600u, 1, 40, 40},  /* t138: 1x1x40x40 */
    {1644800u, 1600u, 1, 1, 1600},  /* t139: 1x1x1600 */
    {1646400u, 12800u, 32, 20, 20},  /* t140: 1x32x20x20 */
    {1632000u, 12800u, 32, 20, 20},  /* t141: 1x32x20x20 */
    {1606400u, 12800u, 32, 20, 20},  /* t142: 1x32x20x20 */
    {1646400u, 12800u, 32, 20, 20},  /* t143: 1x32x20x20 */
    {1606400u, 12800u, 32, 20, 20},  /* t144: 1x32x20x20 */
    {1646400u, 25600u, 64, 20, 20},  /* t145: 1x64x20x20 */
    {1606400u, 25600u, 64, 20, 20},  /* t146: 1x64x20x20 */
    {1646400u, 76800u, 192, 20, 20},  /* t147: 1x192x20x20 */
    {409600u, 51200u, 128, 20, 20},  /* t148: 1x128x20x20 */
    {0u, 12800u, 128, 10, 10},  /* t149: 1x128x10x10 */
    {1684800u, 6400u, 16, 20, 20},  /* t150: 1x16x20x20 */
    {204800u, 51200u, 128, 20, 20},  /* t151: 1x128x20x20 */
    {12800u, 12800u, 32, 20, 20},  /* t152: 1x32x20x20 */
    {1699600u, 25600u, 64, 20, 20},  /* t153: 1x64x20x20 */
    {1725200u, 38400u, 384, 10, 10},  /* t154: 1x384x10x10 */
    {1691200u, 6400u, 16, 20, 20},  /* t155: 1x16x20x20 */
    {1523200u, 25600u, 64, 20, 20},  /* t156: 1x64x20x20 */
    {1632000u, 12800u, 32, 20, 20},  /* t157: 1x32x20x20 */
    {1763600u, 102400u, 64, 40, 40},  /* t158: 1x64x40x40 */
    {1105600u, 25600u, 256, 10, 10},  /* t159: 1x256x10x10 */
    {1866000u, 102400u, 64, 40, 40},  /* t160: 1x64x40x40 */
    {1697600u, 1600u, 4, 20, 20},  /* t161: 1x4x20x20 */
    {1606400u, 25600u, 64, 20, 20},  /* t162: 1x64x20x20 */
    {1672000u, 12800u, 32, 20, 20},  /* t163: 1x32x20x20 */
    {0u, 12800u, 128, 10, 10},  /* t164: 1x128x10x10 */
    {1699600u, 12800u, 128, 10, 10},  /* t165: 1x128x10x10 */
    {1712400u, 1600u, 4, 1, 400},  /* t166: 1x4x400 */
    {1714000u, 12800u, 32, 1, 400},  /* t167: 1x32x400 */
    {1726800u, 6400u, 64, 10, 10},  /* t168: 1x64x10x10 */
    {1646400u, 25600u, 64, 20, 20},  /* t169: 1x64x20x20 */
    {1733200u, 12800u, 128, 10, 10},  /* t170: 1x128x10x10 */
    {1699200u, 400u, 1, 20, 20},  /* t171: 1x1x20x20 */
    {1726800u, 400u, 1, 1, 400},  /* t172: 1x1x400 */
    {913600u, 12800u, 128, 10, 10},  /* t173: 1x128x10x10 */
    {563200u, 25600u, 256, 10, 10},  /* t174: 1x256x10x10 */
    {588800u, 25600u, 2, 128, 100},  /* t175: 1x2x128x100 */
    {1080000u, 6400u, 2, 32, 100},  /* t176: 1x2x32x100 */
    {1086400u, 6400u, 2, 32, 100},  /* t177: 1x2x32x100 */
    {926400u, 12800u, 2, 64, 100},  /* t178: 1x2x64x100 */
    {1092800u, 6400u, 2, 32, 100},  /* t179: 1x2x32x100 */
    {939200u, 12800u, 128, 10, 10},  /* t180: 1x128x10x10 */
    {1099200u, 6400u, 2, 100, 32},  /* t181: 1x2x100x32 */
    {952000u, 12800u, 128, 10, 10},  /* t182: 1x128x10x10 */
    {725600u, 20000u, 2, 100, 100},  /* t183: 1x2x100x100 */
    {745600u, 20000u, 2, 100, 100},  /* t184: 1x2x100x100 */
    {765600u, 20000u, 2, 100, 100},  /* t185: 1x2x100x100 */
    {964800u, 12800u, 2, 64, 100},  /* t186: 1x2x64x100 */
    {977600u, 12800u, 128, 10, 10},  /* t187: 1x128x10x10 */
    {990400u, 12800u, 128, 10, 10},  /* t188: 1x128x10x10 */
    {1003200u, 12800u, 128, 10, 10},  /* t189: 1x128x10x10 */
    {1016000u, 12800u, 128, 10, 10},  /* t190: 1x128x10x10 */
    {614400u, 25600u, 256, 10, 10},  /* t191: 1x256x10x10 */
    {1028800u, 12800u, 128, 10, 10},  /* t192: 1x128x10x10 */
    {1041600u, 12800u, 128, 10, 10},  /* t193: 1x128x10x10 */
    {1727200u, 38400u, 384, 10, 10},  /* t194: 1x384x10x10 */
    {640000u, 25600u, 256, 10, 10},  /* t195: 1x256x10x10 */
    {1730400u, 1600u, 16, 10, 10},  /* t196: 1x16x10x10 */
    {1105600u, 25600u, 256, 10, 10},  /* t197: 1x256x10x10 */
    {1706000u, 3200u, 32, 10, 10},  /* t198: 1x32x10x10 */
    {1734112u, 6400u, 64, 10, 10},  /* t199: 1x64x10x10 */
    {1732000u, 1600u, 16, 10, 10},  /* t200: 1x16x10x10 */
    {0u, 6400u, 64, 10, 10},  /* t201: 1x64x10x10 */
    {1709200u, 3200u, 32, 10, 10},  /* t202: 1x32x10x10 */
    {1740512u, 102400u, 64, 40, 40},  /* t203: 1x64x40x40 */
    {1968400u, 102400u, 64, 40, 40},  /* t204: 1x64x40x40 */
    {1734112u, 102400u, 64, 40, 40},  /* t205: 1x64x40x40 */
    {1733600u, 400u, 4, 10, 10},  /* t206: 1x4x10x10 */
    {6400u, 6400u, 64, 10, 10},  /* t207: 1x64x10x10 */
    {1727200u, 3200u, 32, 10, 10},  /* t208: 1x32x10x10 */
    {1836512u, 400u, 4, 1, 100},  /* t209: 1x4x100 */
    {1836912u, 3200u, 32, 1, 100},  /* t210: 1x32x100 */
    {1840112u, 8400u, 4, 1, 2100},  /* t211: 1x4x2100 */
    {1848512u, 67200u, 32, 1, 2100},  /* t212: 1x32x2100 */
    {1915712u, 102400u, 64, 40, 40},  /* t213: 1x64x40x40 */
    {1699600u, 6400u, 64, 10, 10},  /* t214: 1x64x10x10 */
    {1548800u, 4200u, 2, 1, 2100},  /* t215: 1x2x2100 */
    {1553008u, 4200u, 2, 1, 2100},  /* t216: 1x2x2100 */
    {1557216u, 4200u, 2, 1, 2100},  /* t217: 1x2x2100 */
    {1548800u, 4200u, 2, 1, 2100},  /* t218: 1x2x2100 */
    {2018112u, 409600u, 64, 80, 80},  /* t219: 1x64x80x80 */
    {1561424u, 8400u, 4, 1, 2100},  /* t220: 1x4x2100 */
    {1734000u, 100u, 1, 10, 10},  /* t221: 1x1x10x10 */
    {2427712u, 409600u, 64, 80, 80},  /* t222: 1x64x80x80 */
    {1548800u, 8400u, 4, 1, 2100},  /* t223: 1x4x2100 */
    {1557200u, 100u, 1, 1, 100},  /* t224: 1x1x100 */
    {1557312u, 2100u, 1, 1, 2100},  /* t225: 1x1x2100 */
    {0u, 2100u, 1, 1, 2100},  /* t226: 1x1x2100 */
    {2112u, 204800u, 32, 80, 80},  /* t227: 1x32x80x80 */
    {206912u, 77700u, 37, 1, 2100},  /* t228: 1x37x2100 */
    {1105600u, 77700u, 2100, 1, 37},  /* t229: 1x2100x37 */
    {274112u, 8400u, 2100, 1, 4},  /* t230: 1x2100x4 */
    {0u, 2100u, 2100, 1, 1},  /* t231: 1x2100x1 */
    {206912u, 67200u, 2100, 1, 32},  /* t232: 1x2100x32 */
    {282512u, 2100u, 1, 1, 2100},  /* t233: 1x2100 */
    {284624u, 300u, 1, 1, 300},  /* t234: 1x300 */
    {284928u, 300u, 1, 1, 300},  /* t235: 1x300 */
    {282512u, 300u, 300, 1, 1},  /* t236: 1x300x1 */
    {282816u, 300u, 300, 1, 1},  /* t237: 1x300x1 */
    {0u, 300u, 1, 1, 300},  /* t238: 1x300 */
    {304u, 300u, 1, 1, 300},  /* t239: 1x300 */
    {608u, 300u, 1, 1, 300},  /* t240: 1x300 */
    {0u, 300u, 300, 1, 1},  /* t241: 1x300x1 */
    {912u, 300u, 1, 1, 300},  /* t242: 1x300 */
    {1216u, 300u, 300, 1, 1},  /* t243: 1x300x1 */
    {1520u, 300u, 300, 1, 1},  /* t244: 1x300x1 */
    {282816u, 300u, 300, 1, 1},  /* t245: 1x300x1 */
    {608u, 300u, 300, 1, 1},  /* t246: 1x300x1 */
    {912u, 1200u, 300, 1, 4},  /* t247: 1x300x4 */
    {284928u, 9600u, 300, 1, 32},  /* t248: 1x300x32 */
    {282512u, 1200u, 300, 1, 4},  /* t249: 1x300x4 */
    {294528u, 9600u, 300, 1, 32},  /* t250: 1x300x32 */
    {206912u, 11400u, 300, 1, 38},  /* t251: 1x300x38 */
};

#define TENSOR(arena, id) ((arena) + YOLO_TENSOR[id].off)

#endif /* YOLO26N_SEG_MEM_H */
