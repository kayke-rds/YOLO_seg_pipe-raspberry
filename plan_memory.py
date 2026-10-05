#!/usr/bin/env python3
"""
plan_memory.py -- Planejamento de memoria estatica para YOLO26n-seg.

Le yolo26n_seg_tensors.csv (e opcionalmente yolo26n_seg_ops.csv) e gera
yolo26n_seg_mem.h com:

  - YOLO_ARENA_FLOATS / YOLO_ARENA_BYTES
  - enum TensorId com um ID por tensor
  - YOLO_TENSOR[] com offset (floats) e tamanho (floats)
  - macro TENSOR(arena, id)

Alocacao: first-fit sobre intervalos de vida [birth, death].
Alinhamento: 16 floats (64 B).

Uso:  python3 plan_memory.py <pasta_com_CSVs> [pasta_saida]
"""
import sys, os, csv

ALIGN_FLOATS = 16  # 64 bytes


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else src

    # ---------- 1) Ler tensores ----------
    tensors = {}
    with open(os.path.join(src, "yolo26n_seg_tensors.csv")) as f:
        for row in csv.DictReader(f):
            tid = int(row["tid"])
            elems = int(row["elems"])
            prod = row["producer_op"]
            cons = row["consumer_ops"]
            io = row["io"]
            birth = int(prod) if prod else -1
            death = max(int(c) for c in cons.split()) if cons else 10**9
            if io == "output":
                death = 10**9
            tensors[tid] = {
                "tid": tid,
                "name": row["onnx_name"],
                "shape": row["shape"],
                "elems": elems,
                "birth": birth,
                "death": death,
                "io": io,
            }

    # ---------- 1.b) Aliasing: Reshape/Unsqueeze/Flatten nao criam tensor novo.
    # O C usa o tensor de entrada diretamente. A liveness do input precisa
    # cobrir os consumidores do output. Sem isto, t127 morre em 121 (op do
    # Reshape) e o concat_w que o le em ~197 le lixo.
    with open(os.path.join(src, "yolo26n_seg_ops.csv")) as f:
        for row in csv.DictReader(f):
            if row["op"] not in ("reshape", "unsqueeze", "flatten"):
                continue
            ins  = [int(x[1:]) for x in row["in_tensors"].split()]
            outs = [int(x[1:]) for x in row["out_tensors"].split()]
            for i in ins:
                for o in outs:
                    tensors[i]["death"] = max(tensors[i]["death"],
                                                tensors[o]["death"])

    # ---------- 1.c) Aliasing: Split nao cria tensor novo.
    # Idem para os splits (chan_view sobre t62, t86, t159 etc).
    with open(os.path.join(src, "yolo26n_seg_ops.csv")) as f:
        for row in csv.DictReader(f):
            if row["op"] != "split":
                continue
            ins  = [int(x[1:]) for x in row["in_tensors"].split()]
            outs = [int(x[1:]) for x in row["out_tensors"].split()]
            max_death = max([tensors[i]["death"] for i in ins] +
                            [tensors[o]["death"] for o in outs])
            for i in ins:
                tensors[i]["death"] = max(tensors[i]["death"], max_death)

    # ---------- 1.d) Aliasing: leituras fora de ordem do forward.c ----------
    # P3/P4/P5 sao lidos por head_branch e proto26, que rodam no final do C.
    # Forcamos birth=-1 e death=10^9 para esses tres, garantindo que sejam
    # alocados primeiro e ninguem invada seus offsets.
    HEAD_INPUTS = {
        "/model.16/cv2/act/Mul_output_0",   # P3  (t117)
        "/model.19/cv2/act/Mul_output_0",   # P4  (t148)
        "/model.22/cv2/act/Mul_output_0",   # P5  (t195)
    }
    for t in tensors.values():
        if t["name"] in HEAD_INPUTS:
            t["birth"] = -1
            t["death"] = 10**9

    # ---------- 1.e) Blocos atomicos do head ----------
    # O C executa cada nivel do head como um bloco contiguo:
    #   head_branch(0): box0 -> box1 -> box_out -> cls0..cls4 -> coef0..coef_out
    # No ONNX, essas ops estavam intercaladas (ex.: box0, cls0, coef0, box1, ...)
    # Isso faz o planner permitir que t120 (cls0) e t127 (box_out) compartilhem
    # offset quando suas vidas ONNX nao se sobrepoem — mas no C, t127 e escrito
    # ANTES de t120, entao t120 clobberia t127. Forcamos todos os tensores de
    # um mesmo nivel a terem a mesma janela [min_birth, max_death].
    def head_level(name):
        p = name.split("/")
        if len(p) >= 4 and p[1] == "model.23" and p[2].startswith("one2one_cv"):
            return p[2].split(".")[-1]
        return None

    levels = {}
    for t in tensors.values():
        lv = head_level(t["name"])
        if lv is not None:
            levels.setdefault(lv, []).append(t)

    for lv, ts in levels.items():
        mb = min(t["birth"] for t in ts)
        md = max(t["death"] for t in ts)
        for t in ts:
            t["birth"] = mb
            t["death"] = md

    # ---------- 1.f) Blocos atomicos dos PSABlock (m10 e m22) ----------
    # O C executa cada PSABlock como bloco contiguo, com o input x modificado
    # in-place (add_inplace) e alguns tensores reusados como scratch (proj,
    # ffn0). Isso estende a vida real desses tensores bem além do que o ONNX
    # declara. Aplicamos birth=-1, death=10^9 em toda a familia do PSABlock
    # mais o input.
    PSABLOCK_PREFIXES = (
        "/model.10/m/m.0/attn/",
        "/model.10/m/m.0/ffn/",
        "/model.22/m.0/m.0.1/attn/",
        "/model.22/m.0/m.0.1/ffn/",
    )
    PSABLOCK_EXTRA = {
        "/model.10/cv1/act/Mul_output_0",        # t59  entrada (view do x)
        "/model.10/m/m.0/Add_output_0",          # t78
        "/model.10/m/m.0/Add_1_output_0",        # t81
        "/model.22/m.0/m.0.0/Add_output_0",      # t173 entrada (x)
        "/model.22/m.0/m.0.1/Add_output_0",      # t190
        "/model.22/m.0/m.0.1/Add_1_output_0",    # t193
    }
    for t in tensors.values():
        if (any(t["name"].startswith(p) for p in PSABLOCK_PREFIXES)
                or t["name"] in PSABLOCK_EXTRA):
            t["birth"] = -1
            t["death"] = 10**9

    # ---------- 2) Ordenar por birth ----------
    order = sorted(tensors.values(), key=lambda t: (t["birth"], -t["elems"]))

    # ---------- 3) Alocacao first-fit ----------
    allocated = []  # (offset, size_padded, death, tid)

    def fits(off, size, birth):
        for (o, s, d, _) in allocated:
            if d >= birth:
                if not (off + size <= o or o + s <= off):
                    return False
        return True

    def find_offset(size, birth):
        off = 0
        while True:
            if fits(off, size, birth):
                return off
            off += ALIGN_FLOATS

    total_arena = 0
    for t in order:
        size = t["elems"]
        size_padded = ((size + ALIGN_FLOATS - 1) // ALIGN_FLOATS) * ALIGN_FLOATS
        off = find_offset(size_padded, t["birth"])
        t["offset"] = off
        t["size_padded"] = size_padded
        allocated.append((off, size_padded, t["death"], t["tid"]))
        total_arena = max(total_arena, off + size_padded)

    # ---------- 4) Gerar header ----------
    hdr = []
    A = hdr.append
    A("/* yolo26n_seg_mem.h -- GERADO por plan_memory.py. NAO EDITAR.")
    A(" * Mapa de memoria estatica para a arena de ativacoes do YOLO26n-seg.")
    A(" * Offsets em floats (4 bytes). Arena alinhada a 64 B. */")
    A("#ifndef YOLO26N_SEG_MEM_H")
    A("#define YOLO26N_SEG_MEM_H")
    A("#include <stdint.h>")
    A("")
    A(f"#define YOLO_ARENA_FLOATS  {total_arena}u")
    A(f"#define YOLO_ARENA_BYTES   {total_arena * 4}u")
    A("")
    A("typedef enum {")
    for tid in sorted(tensors.keys()):
        name = tensors[tid]["name"]
        # sanitize: / . - -> _
        s = "T_" + name.replace("/", "_").replace(".", "_").replace("-", "_")
        if s[2].isdigit():
            s = s[:2] + "_" + s[2:]
        A(f"    {s} = {tid},")
    A("    T_COUNT")
    A("} TensorId;")
    A("")
    A("typedef struct {")
    A("    uint32_t off;   /* offset em floats */")
    A("    uint32_t size;  /* tamanho em floats (sem padding) */")
    A("    uint16_t c, h, w; /* shape do tensor */")
    A("} TensorDesc;")
    A("")
    A("static const TensorDesc YOLO_TENSOR[T_COUNT] = {")
    for tid in sorted(tensors.keys()):
        t = tensors[tid]
        # shape vem como "1x3x320x320" ou "1x300x38"
        parts = t["shape"].split("x")
        # normaliza para (c,h,w) — ignora batch
        if len(parts) == 4:      # 1 x C x H x W
            _, c, h, w = parts
        elif len(parts) == 3:    # 1 x C x N  ->  trata N como w, h=1
            _, c, w = parts
            h = 1
        elif len(parts) == 2:    # C x N
            c, w = parts
            h = 1
        else:
            c = h = w = 1
        A(f"    {{{t['offset']}u, {t['elems']}u, {c}, {h}, {w}}},"
          f"  /* t{tid}: {t['shape']} */")
    A("};")
    A("")
    A("#define TENSOR(arena, id) ((arena) + YOLO_TENSOR[id].off)")
    A("")
    A("#endif /* YOLO26N_SEG_MEM_H */")

    with open(os.path.join(out, "yolo26n_seg_mem.h"), "w") as f:
        f.write("\n".join(hdr) + "\n")

    print(f"ok: arena = {total_arena} floats ({total_arena*4} bytes), "
          f"{len(tensors)} tensores")


if __name__ == "__main__":
    main()
