#ifndef IMAGE_IO_H
#define IMAGE_IO_H
#include <stdint.h>

#include "stb_image.h"
#include "stb_image_write.h"

uint8_t *load_image_rgb(const char *path, int *w, int *h);
uint8_t *load_image_gray(const char *path, int *w, int *h);
void     free_image(uint8_t *data);
void     save_mask_png(const char *path, const uint8_t *mask, int w, int h);
void     save_rgb_png (const char *path, const uint8_t *rgb,  int w, int h);

#endif
