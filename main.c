/* main.c -- Benchmark em lote, multiprocesso via fork.
 *
 * Uso:
 *   ./yolo_seg <dir_imgs> <dir_masks> [pesos.bin] [n_workers]
 *
 * Cada worker é um processo independente com sua própria arena.
 * Os pesos são compartilhados via copy-on-write (read-only apos fork).
 * TP/FP/TN/FN parciais são agregados via mmap compartilhado.
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
#include <unistd.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/mman.h>

/* ---------- helpers (mesmos de antes) ---------- */

static int cmp_str(const void *a, const void *b)
{ return strcmp(*(const char *const *)a, *(const char *const *)b); }

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double cpu_sec(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_stime.tv_sec
         + ((double)ru.ru_utime.tv_usec + (double)ru.ru_stime.tv_usec) * 1e-6;
}

static double peak_ram_mb(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1024.0;
}

static int has_ext(const char *n, const char *e)
{
    size_t ln = strlen(n), le = strlen(e);
    if (ln < le) return 0;
    for (size_t i = 0; i < le; i++) {
        char a = n[ln - le + i];
        if (a >= 'A' && a <= 'Z') a = (char)(a + 32);
        if (a != e[i]) return 0;
    }
    return 1;
}

static int is_image(const char *n)
{
    return has_ext(n, ".jpg") || has_ext(n, ".jpeg") || has_ext(n, ".png")
        || has_ext(n, ".bmp") || has_ext(n, ".tif") || has_ext(n, ".tiff");
}

static void stem_of(const char *n, char *o, size_t cap)
{
    const char *d = strrchr(n, '.');
    size_t k = d ? (size_t)(d - n) : strlen(n);
    if (k >= cap) k = cap - 1;
    memcpy(o, n, k); o[k] = '\0';
}

static int load_weights(const char *p)
{
    float *w = (float *)yolo_weights();
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    size_t n = fread(w, sizeof(float), YOLO_WEIGHTS_FLOATS, f);
    fclose(f);
    return (n == YOLO_WEIGHTS_FLOATS) ? 0 : -1;
}

static void accumulate(const uint8_t *pred, const uint8_t *gt, int n,
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

/* ---------- estatisticas por worker (em mmap compartilhado) ---------- */

typedef struct {
    long    tp, fp, tn, fn;
    size_t  processed;
    double  cpu_sum_pct;
    double  t_infer_total;
} Stats;

/* ---------- corpo do worker ---------- */

static void run_worker(int wid, int n_workers,
                       char **files, size_t n_files,
                       const char *imgs_dir, const char *masks_dir,
                       Stats *st)
{
    memset(st, 0, sizeof(*st));

    static float   input_nchw[3 * 320 * 320];
    static uint8_t gt_320[320 * 320];
    static uint8_t pred_mask[320 * 320];

    double cpu_prev  = cpu_sec();
    double wall_prev = now_sec();

    for (size_t i = (size_t)wid; i < n_files; i += (size_t)n_workers) {
        char img_path[1024], mask_path[1024], stem[256];
        snprintf(img_path,  sizeof(img_path),  "%s/%s",    imgs_dir, files[i]);
        stem_of(files[i], stem, sizeof(stem));
        snprintf(mask_path, sizeof(mask_path), "%s/%s.png", masks_dir, stem);

        int sw, sh;
        uint8_t *img = load_image_rgb(img_path, &sw, &sh);
        if (!img) continue;

        int gw, gh;
        uint8_t *gt_rgb = load_image_rgb(mask_path, &gw, &gh);
        if (!gt_rgb) { free_image(img); continue; }

        /* inferencia */
        double t0 = now_sec();
        LetterboxInfo lb;
        letterbox_image_rgb_to_nchw(img, sh, sw, input_nchw, 320, 320, &lb);
        yolo_forward(input_nchw);
        decode_mask_for_detection(yolo_output1(), yolo_output0(),
                                  0, 320, 320, pred_mask);
        double t1 = now_sec();
        st->t_infer_total += t1 - t0;

        free_image(img);

        /* GT */
        uint8_t *gt_gray = (uint8_t *)malloc((size_t)gw * gh);
        if (!gt_gray) { free_image(gt_rgb); continue; }
        for (int k = 0; k < gw * gh; k++) {
            uint8_t r = gt_rgb[k*3 + 0], g = gt_rgb[k*3 + 1], b = gt_rgb[k*3 + 2];
            gt_gray[k] = (r > 60 && r > g + 30 && r > b + 30) ? 255 : 0;
        }
        free_image(gt_rgb);
        letterbox_mask_nearest(gt_gray, gh, gw, gt_320, 320, 320, &lb);
        free(gt_gray);

        accumulate(pred_mask, gt_320, 320 * 320,
                   &st->tp, &st->fp, &st->tn, &st->fn);
        st->processed++;

        /* CPU amostrado */
        double cpu_now = cpu_sec(), wall_now = now_sec();
        double wall_dt = wall_now - wall_prev;
        if (wall_dt > 0.0) st->cpu_sum_pct += 100.0 * (cpu_now - cpu_prev) / wall_dt;
        cpu_prev  = cpu_now;
        wall_prev = wall_now;
    }
}

/* ---------- main ---------- */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "uso: %s <dir_imgs> <dir_masks> [pesos.bin] [n_workers]\n", argv[0]);
        return 1;
    }

    const char *imgs_dir     = argv[1];
    const char *masks_dir    = argv[2];
    const char *weights_path = (argc > 3) ? argv[3] : "yolo26n_seg_weights.bin";
    int n_workers            = (argc > 4) ? atoi(argv[4]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n_workers < 1) n_workers = 1;
    if (n_workers > 32) n_workers = 32;

    if (load_weights(weights_path) != 0) {
        fprintf(stderr, "falha ao carregar pesos: %s\n", weights_path);
        return 1;
    }

    /* enumera imagens */
    DIR *d = opendir(imgs_dir);
    if (!d) { perror("opendir"); return 1; }

    char **files = NULL;
    size_t n_files = 0, cap = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!is_image(ent->d_name)) continue;
        if (n_files == cap) {
            cap = cap ? cap * 2 : 64;
            files = (char **)realloc(files, cap * sizeof(char *));
        }
        files[n_files++] = strdup(ent->d_name);
    }
    closedir(d);

    if (n_files == 0) {
        fprintf(stderr, "nenhuma imagem em %s\n", imgs_dir);
        return 1;
    }
    qsort(files, n_files, sizeof(char *), cmp_str);

    if (n_workers > (int)n_files) n_workers = (int)n_files;

    printf("imagens:   %zu\n", n_files);
    printf("pesos:     %s\n", weights_path);
    printf("workers:   %d\n\n", n_workers);

    /* area compartilhada para estatisticas */
    Stats *stats = mmap(NULL, n_workers * sizeof(Stats),
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (stats == MAP_FAILED) { perror("mmap"); return 1; }

    double wall_start = now_sec();

    pid_t *pids = (pid_t *)malloc(n_workers * sizeof(pid_t));
    for (int w = 0; w < n_workers; w++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 1; }
        if (pid == 0) {
            /* filho: roda e sai */
            run_worker(w, n_workers, files, n_files, imgs_dir, masks_dir, &stats[w]);
            _exit(0);
        }
        pids[w] = pid;
    }

    /* pai espera */
    int rc_any = 0;
    for (int w = 0; w < n_workers; w++) {
        int status;
        waitpid(pids[w], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) rc_any = 1;
    }

    double wall_total = now_sec() - wall_start;

    /* agrega */
    long   tp = 0, fp = 0, tn = 0, fn = 0;
    size_t n_processed   = 0;
    double t_infer_total = 0.0;
    double cpu_sum       = 0.0;
    for (int w = 0; w < n_workers; w++) {
        tp += stats[w].tp;
        fp += stats[w].fp;
        tn += stats[w].tn;
        fn += stats[w].fn;
        n_processed   += stats[w].processed;
        t_infer_total += stats[w].t_infer_total;
        cpu_sum       += stats[w].cpu_sum_pct;
    }

    for (size_t i = 0; i < n_files; i++) free(files[i]);
    free(files);
    free(pids);

    if (n_processed == 0) { fprintf(stderr, "nada processado\n"); return 1; }

    const double eps = 1e-7;
    double iou  = (double)tp / ((double)tp + (double)fp + (double)fn + eps);
    double dice = 2.0 * (double)tp / (2.0 * (double)tp + (double)fp + (double)fn + eps);
    double prec = (double)tp / ((double)tp + (double)fp + eps);
    double rec  = (double)tp / ((double)tp + (double)fn + eps);

    /* FPS agregado: usa wall total, nao soma de tempos dos workers.
     * t_infer_total/N aqui seria "FPS por worker"; o que queremos e
     * quantas imagens por segundo o conjunto entregou. */
    double fps_aggregate = (double)n_processed / wall_total;
    double cpu_avg       = cpu_sum / (double)n_workers;
    double peak_ram      = peak_ram_mb() / 1024.0;  /* MB */

    printf("=== Resultado (n=%zu, workers=%d) ===\n", n_processed, n_workers);
    printf("IoU=%.4f  Dice=%.4f  Prec=%.4f  Rec=%.4f\n", iou, dice, prec, rec);
    printf("TP=%ld  FP=%ld  TN=%ld  FN=%ld\n", tp, fp, tn, fn);
    printf("FPS=%.2f  wall=%.2fs  RAM_pico=%.1f MB  CPU_medio=%.1f%%\n",
           fps_aggregate, wall_total, peak_ram, cpu_avg);

    return rc_any;
}
