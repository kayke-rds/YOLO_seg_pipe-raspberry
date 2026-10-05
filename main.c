/* main.c -- Benchmark em lote: roda o forward em todas as imagens de uma
 * pasta, acumula TP/FP/TN/FN globalmente e imprime IoU/Dice/Prec/Rec
 * globais + metricas de performance (FPS, RAM pico, CPU media).
 *
 * Uso:
 *   ./yolo_seg <dir_imgs> <dir_masks> [pesos.bin]
 *
 * As mascaras GT sao procuradas por <stem>.png em <dir_masks>.
 */
#include "yolo26n_seg.h"
#include "image_io.h"
#include "preprocess.h"
#include "postprocess.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>
#include <sys/resource.h>

/* ---------- helpers ---------- */

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Tempo de CPU acumulado (user + sys), em segundos. */
static double cpu_sec(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_stime.tv_sec
         + ((double)ru.ru_utime.tv_usec + (double)ru.ru_stime.tv_usec) * 1e-6;
}

/* Pico de RSS desde o inicio do processo (Linux: KB). */
static double peak_ram_mb(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1024.0;
}

static int has_ext(const char *name, const char *ext)
{
    size_t ln = strlen(name), le = strlen(ext);
    if (ln < le) return 0;
    for (size_t i = 0; i < le; i++) {
        char a = name[ln - le + i];
        if (a >= 'A' && a <= 'Z') a = (char)(a + 32);
        if (a != ext[i]) return 0;
    }
    return 1;
}

static int is_image(const char *name)
{
    return has_ext(name, ".jpg")  || has_ext(name, ".jpeg")
        || has_ext(name, ".png")  || has_ext(name, ".bmp")
        || has_ext(name, ".tif")  || has_ext(name, ".tiff");
}

/* "abc.jpg" -> "abc" */
static void stem_of(const char *name, char *out, size_t cap)
{
    const char *dot = strrchr(name, '.');
    size_t n = dot ? (size_t)(dot - name) : strlen(name);
    if (n >= cap) n = cap - 1;
    memcpy(out, name, n);
    out[n] = '\0';
}

static int load_weights(const char *path)
{
    float *w = (float *)yolo_weights();
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(w, sizeof(float), YOLO_WEIGHTS_FLOATS, f);
    fclose(f);
    return (n == YOLO_WEIGHTS_FLOATS) ? 0 : -1;
}

/* Acumula TP/FP/TN/FN para a classe "pipe" (foreground). */
static void accumulate_confusion(const uint8_t *pred, const uint8_t *gt, int n,
                                 long *tp, long *fp, long *tn, long *fn)
{
    for (int i = 0; i < n; i++) {
        int p = pred[i] != 0, g = gt[i] != 0;
        if      ( p &&  g) (*tp)++;
        else if ( p && !g) (*fp)++;
        else if (!p &&  g) (*fn)++;
        else                (*tn)++;
    }
}

/* ---------- main ---------- */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "uso: %s <dir_imgs> <dir_masks> [pesos.bin]\n", argv[0]);
        return 1;
    }

    const char *imgs_dir     = argv[1];
    const char *masks_dir    = argv[2];
    const char *weights_path = (argc > 3) ? argv[3] : "yolo26n_seg_weights.bin";

    if (load_weights(weights_path) != 0) {
        fprintf(stderr, "falha ao carregar pesos: %s\n", weights_path);
        return 1;
    }

    /* Coleta lista de imagens. */
    DIR *d = opendir(imgs_dir);
    if (!d) { perror("opendir"); return 1; }

    char  **files     = NULL;
    size_t  n_files   = 0;
    size_t  cap_files = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.')   continue;
        if (!is_image(ent->d_name))  continue;
        if (n_files == cap_files) {
            cap_files = cap_files ? cap_files * 2 : 64;
            files = (char **)realloc(files, cap_files * sizeof(char *));
        }
        files[n_files++] = strdup(ent->d_name);
    }
    closedir(d);

    if (n_files == 0) {
        fprintf(stderr, "nenhuma imagem encontrada em %s\n", imgs_dir);
        return 1;
    }
    qsort(files, n_files, sizeof(char *), cmp_str);

    printf("imagens: %zu\n", n_files);
    printf("pesos:   %s\n\n", weights_path);

    /* Acumuladores globais. */
    long   tp = 0, fp = 0, tn = 0, fn = 0;
    size_t n_processed = 0;

    /* Performance. */
    double t_infer_total = 0.0;    /* soma dos wall-times (letterbox+forward+decode) */
    double cpu_sum_pct   = 0.0;    /* soma das amostras de %CPU por imagem */
    double cpu_prev      = cpu_sec();
    double wall_prev     = now_sec();

    /* Buffers reutilizados entre imagens. */
    static float   input_nchw[3 * 320 * 320];
    static uint8_t gt_320[320 * 320];
    static uint8_t pred_mask[320 * 320];

    for (size_t i = 0; i < n_files; i++) {
        char img_path[1024], mask_path[1024], stem[256];
        snprintf(img_path,  sizeof(img_path),  "%s/%s",    imgs_dir, files[i]);
        stem_of(files[i], stem, sizeof(stem));
        snprintf(mask_path, sizeof(mask_path), "%s/%s.png", masks_dir, stem);

        int src_w, src_h;
        uint8_t *img = load_image_rgb(img_path, &src_w, &src_h);
        if (!img) {
            fprintf(stderr, "\n[skip] imagem: %s\n", img_path);
            continue;
        }

        int gt_w, gt_h;
        uint8_t *gt_rgb = load_image_rgb(mask_path, &gt_w, &gt_h);
        if (!gt_rgb) {
            fprintf(stderr, "\n[skip] mascara: %s\n", mask_path);
            free_image(img);
            continue;
        }

        /* --- Inferencia: letterbox + forward + decode da mascara --- */
        double t0 = now_sec();

        LetterboxInfo lb;
        letterbox_image_rgb_to_nchw(img, src_h, src_w, input_nchw, 320, 320, &lb);
        yolo_forward(input_nchw);

        const float *out0  = yolo_output0();
        const float *proto = yolo_output1();
        decode_mask_for_detection(proto, out0, 0, 320, 320, pred_mask);

        double t1 = now_sec();
        t_infer_total += t1 - t0;

        free_image(img);

        /* --- GT: binariza + letterbox --- */
        uint8_t *gt_gray = (uint8_t *)malloc((size_t)gt_w * gt_h);
        if (!gt_gray) { fprintf(stderr, "\nOOM\n"); free_image(gt_rgb); return 1; }
        for (int k = 0; k < gt_w * gt_h; k++) {
            uint8_t r = gt_rgb[k*3 + 0];
            uint8_t g = gt_rgb[k*3 + 1];
            uint8_t b = gt_rgb[k*3 + 2];
            gt_gray[k] = (r > 60 && r > g + 30 && r > b + 30) ? 255 : 0;
        }
        free_image(gt_rgb);
        letterbox_mask_nearest(gt_gray, gt_h, gt_w, gt_320, 320, 320, &lb);
        free(gt_gray);

        /* --- Acumula confusao --- */
        accumulate_confusion(pred_mask, gt_320, 320 * 320, &tp, &fp, &tn, &fn);
        n_processed++;

        /* --- Amostra de CPU apos esta imagem (analoga a psutil.cpu_percent) --- */
        double cpu_now  = cpu_sec();
        double wall_now = now_sec();
        double cpu_dt   = cpu_now  - cpu_prev;
        double wall_dt  = wall_now - wall_prev;
        if (wall_dt > 0.0) cpu_sum_pct += 100.0 * cpu_dt / wall_dt;
        cpu_prev  = cpu_now;
        wall_prev = wall_now;

        printf("\rimagem %zu/%zu", i + 1, n_files);
        fflush(stdout);
    }
    printf("\n\n");

    for (size_t i = 0; i < n_files; i++) free(files[i]);
    free(files);

    if (n_processed == 0) {
        fprintf(stderr, "nenhuma imagem processada\n");
        return 1;
    }

    /* --- Metricas globais (acumuladas, nao media de medias) --- */
    const double eps = 1e-7;
    double iou  = (double)tp / ((double)tp + (double)fp + (double)fn + eps);
    double dice = 2.0 * (double)tp / (2.0 * (double)tp + (double)fp + (double)fn + eps);
    double prec = (double)tp / ((double)tp + (double)fp + eps);
    double rec  = (double)tp / ((double)tp + (double)fn + eps);

    /* --- Performance --- */
    double fps           = (double)n_processed / t_infer_total;
    double peak_ram      = peak_ram_mb();
    double avg_cpu_usage = cpu_sum_pct / (double)n_processed;

    printf("=== Resultado (n=%zu) ===\n", n_processed);
    printf("IoU=%.4f  Dice=%.4f  Prec=%.4f  Rec=%.4f\n",
           iou, dice, prec, rec);
    printf("TP=%ld  FP=%ld  TN=%ld  FN=%ld\n", tp, fp, tn, fn);
    printf("FPS=%.2f  RAM_pico=%.1f MB  CPU=%.1f%%\n",
           fps, peak_ram, avg_cpu_usage);

    return 0;
}
