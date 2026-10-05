#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image_io.h"
#include <stdlib.h>

uint8_t *load_image_rgb(const char *path, int *w, int *h) {
    int channels;
    return stbi_load(path, w, h, &channels, 3);
}

uint8_t *load_image_gray(const char *path, int *w, int *h) {
    int channels;
    return stbi_load(path, w, h, &channels, 1);
}

void free_image(uint8_t *data) { stbi_image_free(data); }

void save_mask_png(const char *path, const uint8_t *mask, int w, int h) {
    uint8_t *buf = (uint8_t *)malloc((size_t)w * h);
    if (!buf) return;
    for (int i = 0; i < w * h; i++) buf[i] = mask[i] ? 255 : 0;
    stbi_write_png(path, w, h, 1, buf, w);
    free(buf);
}

void save_rgb_png(const char *path, const uint8_t *rgb, int w, int h) {
    stbi_write_png(path, w, h, 3, rgb, w * 3);
}
