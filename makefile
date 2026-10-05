CC      := gcc
CFLAGS  := -std=gnu11 -O3 -march=native -funroll-loops \
           -Wall -Wextra -Wno-missing-field-initializers
LDFLAGS := -lm

TARGET  := yolo_seg

SRC     := main.c forward.c head.c blocks.c kernels.c \
           image_io.c preprocess.c postprocess.c yolo26n_seg_mem.c

HDR     := yolo26n_seg.h yolo26n_seg_layout.h yolo26n_seg_mem.h \
           image_io.h preprocess.h postprocess.h

IMG_DIR   := ./images
MASK_DIR  := ./masks
WEIGHTS   := yolo26n_seg_weights.bin

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(IMG_DIR) $(MASK_DIR) $(WEIGHTS)

clean:
	rm -f $(TARGET)
