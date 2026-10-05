#!/usr/bin/env python3
"""
Extrai do ONNX do YOLO26n-seg (BN já fundida) tudo o que a reimplementação em C precisa:

  yolo26n_seg_weights.bin   blob float32 (little-endian) com pesos+bias na ordem canônica de execução
  yolo26n_seg_layout.h      enum + tabela ConvDesc[] (offsets em floats, geometria, ativação) + constantes
  yolo26n_seg_layers.csv    uma linha por camada com parâmetros (118)
  yolo26n_seg_ops.csv       grafo fundido em ordem topológica (Conv+Sigmoid+Mul -> 1 operador)
  yolo26n_seg_tensors.csv   tensores de ativação (shape, produtor, consumidores)

Uso:  python3 export_yolo26n_seg.py best.onnx [pasta_saida]
Requer: numpy, onnx
"""
import sys, os, csv, collections
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

ALIGN_FLOATS = 16            # 64 bytes = linha de cache do Raspberry Pi 4/5 (e múltiplo de NEON 16 B)
MODULE_TYPES = ["Conv", "Conv", "C3k2", "Conv", "C3k2", "Conv", "C3k2", "Conv", "C3k2", "SPPF", "C2PSA",
                "Upsample", "Concat", "C3k2", "Upsample", "Concat", "C3k2", "Conv", "Concat", "C3k2",
                "Conv", "Concat", "C3k2", "Segment26"]


def attrs_of(n):
    d = {}
    for a in n.attribute:
        if a.type == onnx.AttributeProto.INTS:
            d[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.INT:
            d[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            d[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            d[a.name] = a.s.decode()
    return d


def block_of(name):
    """'/model.6/m.0/cv1/conv/Conv' -> 6"""
    return int(name.split("/")[1].split(".")[1])


def canon_group(wname):
    """Ordem canônica do blob: módulo; na cabeça, nível-a-nível (box, cls, coef) e por fim proto."""
    p = wname.split(".")
    if p[1] != "23":
        return (int(p[1]), 0)
    if p[2] == "proto":
        return (23, 100)
    branch = {"one2one_cv2": 0, "one2one_cv3": 1, "one2one_cv4": 2}[p[2]]
    return (23, int(p[3]) * 10 + branch)


def cname(wname):
    s = wname[:-len(".weight")]
    if s.endswith(".conv"):
        s = s[:-5]
    return s.replace(".", "_")


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out, exist_ok=True)
    model = shape_inference.infer_shapes(onnx.load(src))
    g = model.graph
    init = {i.name: i for i in g.initializer}
    shp = {}
    for v in list(g.value_info) + list(g.input) + list(g.output):
        shp[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    for i in g.initializer:
        shp[i.name] = list(i.dims)
    meta = {p.key: p.value for p in model.metadata_props}

    # ---------- 1) grafo fundido: Conv + Sigmoid + Mul(x, sigmoid(x)) -> conv com act=silu ----------
    nodes = list(g.node)
    cons = collections.defaultdict(list)
    for n in nodes:
        for i in n.input:
            cons[i].append(n)
    ops, skip = [], set()
    for n in nodes:
        if id(n) in skip:
            continue
        a = attrs_of(n)
        if n.op_type == "Conv":
            co, act, y = n.output[0], "none", n.output[0]
            sg = [x for x in cons[co] if x.op_type == "Sigmoid"]
            if sg:
                mul = [x for x in cons[sg[0].output[0]] if x.op_type == "Mul"][0]
                assert len(cons[co]) == 2 and len(cons[sg[0].output[0]]) == 1
                skip |= {id(sg[0]), id(mul)}
                act, y = "silu", mul.output[0]
            ops.append(dict(op="conv", name=n.name, ins=[n.input[0]], outs=[y], w=n.input[1], b=n.input[2],
                            kind="conv", k=a["kernel_shape"][0], s=a["strides"][0], p=a["pads"][0],
                            g=a["group"], act=act))
        elif n.op_type == "ConvTranspose":
            ops.append(dict(op="convT", name=n.name, ins=[n.input[0]], outs=[n.output[0]], w=n.input[1],
                            b=n.input[2], kind="convT", k=a["kernel_shape"][0], s=a["strides"][0], p=0,
                            g=a["group"], act="none"))
        else:
            ops.append(dict(op=n.op_type.lower(), name=n.name, ins=[i for i in n.input if i not in init],
                            outs=list(n.output), at=a, consts=[i for i in n.input if i in init]))

    # ---------- 2) tensores ----------
    tid, T = {}, []

    def tensor(name):
        if name not in tid:
            tid[name] = len(T)
            T.append(dict(id=len(T), onnx=name, shape=shp.get(name), prod=None, cons=[]))
        return tid[name]

    tensor("images")
    for k, o in enumerate(ops):
        o["i"] = k
        o["in_t"] = [tensor(x) for x in o["ins"]]
        o["out_t"] = [tensor(x) for x in o["outs"]]
        for t in o["out_t"]:
            T[t]["prod"] = k
        for t in o["in_t"]:
            T[t]["cons"].append(k)

    # ---------- 3) camadas com parâmetros, em ordem canônica, e layout do blob ----------
    layers = [o for o in ops if o["op"] in ("conv", "convT")]
    layers.sort(key=lambda o: canon_group(o["w"]))          # sort estável: preserva a ordem de execução
    names = [cname(o["w"]) for o in layers]
    assert len(set(names)) == len(names)
    blob, cursor = [], 0

    def put(arr):
        nonlocal cursor
        flat = np.ascontiguousarray(arr, dtype="<f4").ravel()
        off = cursor
        pad = (-flat.size) % ALIGN_FLOATS
        blob.append(flat)
        if pad:
            blob.append(np.zeros(pad, dtype="<f4"))
        cursor += flat.size + pad
        return off, flat.size

    rows = []
    for idx, (o, nm) in enumerate(zip(layers, names)):
        w = numpy_helper.to_array(init[o["w"]])
        b = numpy_helper.to_array(init[o["b"]])
        xs, ys = T[o["in_t"][0]]["shape"], T[o["out_t"][0]]["shape"]
        if o["kind"] == "conv":
            cout, cin_g, kh, kw = w.shape
            cin = cin_g * o["g"]
            macs = ys[2] * ys[3] * cout * cin_g * kh * kw
        else:                                              # ConvTranspose: pesos [Cin][Cout][kh][kw]
            cin, cout, kh, kw = w.shape
            macs = xs[2] * xs[3] * cin * cout * kh * kw
        assert xs[1] == cin and ys[1] == cout and kh == kw == o["k"] and xs[2] == xs[3] and ys[2] == ys[3]
        w_off, n_w = put(w)
        b_off, n_b = put(b)
        blk = block_of(o["name"])
        rows.append(dict(idx=idx, cid=f"CV_{nm}", weight=o["w"], bias=o["b"], module=f"model.{blk}",
                         mtype=MODULE_TYPES[blk], kind=o["kind"], cin=cin, cout=cout, k=o["k"], stride=o["s"],
                         pad=o["p"], groups=o["g"],
                         dw=int(o["g"] > 1 and o["g"] == cin == cout), act=o["act"],
                         h_in=xs[2], w_in=xs[3], h_out=ys[2], w_out=ys[3], w_off=w_off, b_off=b_off, n_w=n_w,
                         n_b=n_b, macs=macs, op_idx=o["i"], t_in=o["in_t"][0], t_out=o["out_t"][0]))
    flat_blob = np.concatenate(blob)
    flat_blob.tofile(os.path.join(out, "yolo26n_seg_weights.bin"))

    # ---------- 4) header C ----------
    hdr = []
    A = hdr.append
    A("/* yolo26n_seg_layout.h -- GERADO por export_yolo26n_seg.py. NAO EDITAR.")
    A(f" * Origem: {os.path.basename(src)}  ({meta.get('description', '')})")
    A(" * Blob: float32 little-endian, tensores alinhados a 64 B; offsets em NUMERO DE FLOATS a partir do inicio. */")
    A("#ifndef YOLO26N_SEG_LAYOUT_H\n#define YOLO26N_SEG_LAYOUT_H\n#include <stdint.h>\n")
    A('#define YOLO_WEIGHTS_FILE   "yolo26n_seg_weights.bin"')
    A(f"#define YOLO_WEIGHTS_FLOATS {flat_blob.size}u")
    A(f"#define YOLO_WEIGHTS_BYTES  {flat_blob.size * 4}u")
    A(f"#define YOLO_ALIGN_FLOATS   {ALIGN_FLOATS}\n")
    A("/* ---- constantes do modelo (todas lidas do ONNX) ---- */")
    A("#define IN_C 3\n#define IN_H 320\n#define IN_W 320")
    A("#define NC 1                 /* classes ('pipe') */\n#define NM 32                /* coeficientes de mascara */")
    A("#define N_ANCHORS 2100       /* 40*40 + 20*20 + 10*10 */\n#define TOPK 300             /* deteccoes de saida */")
    A("#define OUT0_COLS 38         /* x1,y1,x2,y2,score,cls,coef[32] */\n#define PROTO_C 32\n#define PROTO_H 80\n#define PROTO_W 80")
    A("#define ATTN_HEADS 2\n#define ATTN_KEY_DIM 32\n#define ATTN_HEAD_DIM 64\n#define ATTN_TOKENS 100\n#define ATTN_SCALE 0.17677669529663687f /* 32^-0.5 */\n")
    A("typedef enum { ACT_NONE = 0, ACT_SILU = 1 } Act;")
    A("typedef enum { KIND_CONV = 0, KIND_CONVT = 1 } ConvKind;\n")
    A("/* Pesos conv : [cout][cin/groups][k][k]  (OIHW, igual ao ONNX)\n * Pesos convT: [cin][cout][k][k]\n * Bias       : [cout]                                         */")
    A("typedef struct {\n    uint32_t w_off, b_off;       /* offsets (floats) de pesos e bias */")
    A("    uint16_t cin, cout;          /* canais totais */\n    uint16_t groups;             /* 1 = normal; =cin=cout -> depthwise (ate 256) */")
    A("    uint8_t  k, stride, pad;\n    uint8_t  act, kind;          /* Act, ConvKind */\n    uint16_t h_in, w_in, h_out, w_out;\n} ConvDesc;\n")
    A("typedef enum {")
    for r in rows:
        A(f"    {r['cid']} = {r['idx']},")
    A(f"    CV_COUNT = {len(rows)}\n}} ConvId;\n")
    A("static const ConvDesc YOLO_CONV[CV_COUNT] = {")
    for r in rows:
        A(f"    /* {r['idx']:3d} {r['module']:<9} {r['weight'][:-7]} */ "
          f"{{{r['w_off']}u, {r['b_off']}u, {r['cin']}, {r['cout']}, {r['groups']}, {r['k']}, {r['stride']}, {r['pad']}, "
          f"{'ACT_SILU' if r['act'] == 'silu' else 'ACT_NONE'}, {'KIND_CONVT' if r['kind'] == 'convT' else 'KIND_CONV'}, "
          f"{r['h_in']}, {r['w_in']}, {r['h_out']}, {r['w_out']}}},")
    A("};\n")
    A("#define CONV_W(blob, id) ((blob) + YOLO_CONV[id].w_off)")
    A("#define CONV_B(blob, id) ((blob) + YOLO_CONV[id].b_off)\n")
    A("#endif /* YOLO26N_SEG_LAYOUT_H */")
    with open(os.path.join(out, "yolo26n_seg_layout.h"), "w") as f:
        f.write("\n".join(hdr) + "\n")

    # ---------- 5) CSVs ----------
    def write_csv(fname, cols, data):
        with open(os.path.join(out, fname), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(data)

    lc = ["idx", "cid", "weight_name", "module", "module_type", "kind", "cin", "cout", "k", "stride", "pad",
          "groups", "depthwise", "act", "h_in", "w_in", "h_out", "w_out", "w_off", "b_off", "n_w", "n_b", "macs",
          "op_idx", "t_in", "t_out"]
    write_csv("yolo26n_seg_layers.csv", lc, [[r["idx"], r["cid"], r["weight"], r["module"], r["mtype"], r["kind"],
              r["cin"], r["cout"], r["k"], r["stride"], r["pad"], r["groups"], r["dw"], r["act"], r["h_in"],
              r["w_in"], r["h_out"], r["w_out"], r["w_off"], r["b_off"], r["n_w"], r["n_b"], r["macs"],
              r["op_idx"], r["t_in"], r["t_out"]] for r in rows])

    def cinfo(o):
        out_ = []
        for c in o.get("consts", []):
            a = numpy_helper.to_array(init[c])
            out_.append(f"{c.split('/')[-1]}={a.tolist()}" if a.size <= 4 else f"{c.split('/')[-1]}{list(a.shape)}")
        return ";".join(out_)

    def shp_s(t):
        s = T[t]["shape"]
        return "x".join(map(str, s[1:])) if s and s[0] == 1 else "x".join(map(str, s))

    od = []
    for o in ops:
        blk = o["name"].split("/")[1] if o["name"].startswith("/") else ""
        extra = (f"k{o['k']}s{o['s']}g{o['g']}{'' if o['act'] == 'silu' else ' linear'}" if o["op"] in ("conv", "convT")
                 else ";".join(f"{k}={v}" for k, v in o.get("at", {}).items()))
        od.append([o["i"], o["op"], blk, o["name"], " ".join(f"t{t}" for t in o["in_t"]),
                   " ".join(f"t{t}" for t in o["out_t"]), "|".join(shp_s(t) for t in o["in_t"]),
                   "|".join(shp_s(t) for t in o["out_t"]), extra, cinfo(o) if o["op"] not in ("conv", "convT") else ""])
    write_csv("yolo26n_seg_ops.csv", ["op_idx", "op", "module", "onnx_name", "in_tensors", "out_tensors",
                                      "in_shapes", "out_shapes", "params", "consts"], od)

    outs = {v.name for v in g.output}
    td = []
    for t in T:
        s = t["shape"]
        td.append([t["id"], t["onnx"], "x".join(map(str, s)), int(np.prod(s)),
                   "" if t["prod"] is None else t["prod"], " ".join(map(str, t["cons"])), len(t["cons"]),
                   "input" if t["onnx"] == "images" else ("output" if t["onnx"] in outs else "")])
    write_csv("yolo26n_seg_tensors.csv", ["tid", "onnx_name", "shape", "elems", "producer_op", "consumer_ops",
                                          "n_consumers", "io"], td)

    # ---------- 6) auto-verificação: ler o blob de volta e comparar com o ONNX ----------
    chk = np.fromfile(os.path.join(out, "yolo26n_seg_weights.bin"), dtype="<f4")
    assert chk.size == flat_blob.size == cursor
    for r in rows:
        assert np.array_equal(chk[r["w_off"]:r["w_off"] + r["n_w"]], numpy_helper.to_array(init[r["weight"]]).ravel())
        assert np.array_equal(chk[r["b_off"]:r["b_off"] + r["n_b"]], numpy_helper.to_array(init[r["bias"]]).ravel())
    npar = sum(r["n_w"] + r["n_b"] for r in rows)
    print(f"ok: {len(rows)} camadas, {npar} parametros, blob {chk.size} floats ({chk.size * 4} bytes), "
          f"{len(ops)} operadores fundidos, {len(T)} tensores, {sum(r['macs'] for r in rows) / 1e9:.3f} GMACs")


if __name__ == "__main__":
    main()
