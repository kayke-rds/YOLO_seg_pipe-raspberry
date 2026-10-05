# YOLO26n-seg → C: mapeamento de estruturas e operações

Etapa 1 do projeto. Escopo: inventariar tudo que precisa ser reimplementado em C e definir o blob estático de pesos. Plano de memória das ativações (ping-pong e buffers residuais) fica para a etapa 2.

**Fontes:** `best.onnx` (opset 12, `simplify=True`, `nms=False`, `end2end=True`) e `best.pt` (Ultralytics 8.4, `SegmentationModel`, `imgsz=320`, 1 classe `pipe`).
**Arquivos gerados** (script `export_yolo26n_seg.py`): `yolo26n_seg_weights.bin`, `yolo26n_seg_layout.h`, `yolo26n_seg_layers.csv`, `yolo26n_seg_ops.csv`, `yolo26n_seg_tensors.csv`.

## 1. Resumo do modelo

| Item | Valor |
|---|---|
| Entrada | `images` 1×3×320×320, float32, NCHW |
| Saída 0 (`output0`) | 1×300×38 = 300 linhas `[x1,y1,x2,y2, score, cls, coef[32]]` |
| Saída 1 (`output1`) | 1×32×80×80 = protótipos de máscara (já com SiLU) |
| Camadas com parâmetros | 118 (117 Conv2d + 1 ConvTranspose2d) |
| Parâmetros no ONNX (pesos+bias) | 2.689.079 (10,26 MiB em float32) |
| Custo | 1,12 GMACs por inferência |
| Operadores no grafo fundido | 234 (432 nós ONNX; `Conv+Sigmoid+Mul` → 1 conv com SiLU) |
| Tensores de ativação | 252 (1 entrada, 2 saídas) |

Pontos que valem registrar:

- **BatchNorm já está fundida** nos pesos do ONNX (toda conv tem `weight` + `bias`). Conferi contra o `.pt` fundindo a BN à mão (`w·γ/√(var+ε)`, `b = β − μ·γ/√(var+ε)`, ε=1e-3): diferença máxima 7,3e-6 nas 118 camadas. Os dois arquivos são o mesmo modelo.
- O ONNX contém **apenas a cabeça `one2one`** (sem NMS). O `.pt` também guarda a cabeça `one2many` (`cv2/cv3/cv4`) e o ramo `semseg` do Proto26; ambos são só de treino e **não entram no blob**.
- `reg_max=1`: o módulo DFL é `Identity`. A regressão da caixa são 4 valores diretos (l,t,r,b), sem softmax/integral.

## 2. Convenções para o C

- Float32, NCHW, batch 1. Em NCHW com batch 1, **um corte de canais é uma região contígua**, então `Split` e `Concat` no eixo de canais são aritmética de ponteiro (ou escrita direta no offset de canal do destino), sem kernel próprio.
- SiLU: `x·σ(x)`, aplicada no epílogo da conv (flag `ACT_SILU`).
- Padding de conv: `k//2` (1 para 3×3, 0 para 1×1), simétrico. Conv 3×3 com stride 2 e pad 1 sobre entrada par dá saída `H/2`.
- Notação abaixo: **CBS(k)** = conv k×k + bias + SiLU; **CB(k)** = conv k×k + bias, sem ativação; **DW** = depthwise (`groups = Cin = Cout`).

## 3. Mapa do grafo por módulo

| # | Módulo | Entrada(s) | Saída | Convs | Params | MMACs | Observação |
|---|---|---|---|---|---|---|---|
| 0 | Conv 3×3 s2 | 3×320×320 | 16×160×160 | 1 | 448 | 11,1 | |
| 1 | Conv 3×3 s2 | m0 | 32×80×80 | 1 | 4.640 | 29,5 | |
| 2 | C3k2 (var. A) | m1 | 64×80×80 | 4 | 6.520 | 41,0 | |
| 3 | Conv 3×3 s2 | m2 | 64×40×40 | 1 | 36.928 | 59,0 | |
| 4 | C3k2 (var. A) | m3 | 128×40×40 | 4 | 25.840 | 41,0 | consumida em m15 |
| 5 | Conv 3×3 s2 | m4 | 128×20×20 | 1 | 147.584 | 59,0 | |
| 6 | C3k2 (var. B) | m5 | 128×20×20 | 9 | 86.528 | 34,4 | consumida em m12 |
| 7 | Conv 3×3 s2 | m6 | 256×10×10 | 1 | 295.168 | 29,5 | |
| 8 | C3k2 (var. B) | m7 | 256×10×10 | 9 | 345.088 | 34,4 | |
| 9 | SPPF | m8 | 256×10×10 | 2 | 164.224 | 16,4 | soma residual com a entrada (m8) |
| 10 | C2PSA | m9 | 256×10×10 | 7 | 248.320 | 24,7 | consumida em m11 e m21 |
| 11 | Upsample ×2 | m10 | 256×20×20 | – | – | – | `Resize` nearest |
| 12 | Concat | m11, m6 | 384×20×20 | – | – | – | ordem: m11, depois m6 |
| 13 | C3k2 (var. B) | m12 | 128×20×20 | 9 | 119.296 | 47,5 | consumida em m14 e m18 |
| 14 | Upsample ×2 | m13 | 128×40×40 | – | – | – | |
| 15 | Concat | m14, m4 | 256×40×40 | – | – | – | |
| 16 | C3k2 (var. B) | m15 | **P3** 64×40×40 | 9 | 34.048 | 54,1 | alimenta m17, cabeça (nível 0) e Proto |
| 17 | Conv 3×3 s2 | m16 | 64×20×20 | 1 | 36.928 | 14,7 | |
| 18 | Concat | m17, m13 | 192×20×20 | – | – | – | |
| 19 | C3k2 (var. B) | m18 | **P4** 128×20×20 | 9 | 94.720 | 37,7 | alimenta m20, cabeça (nível 1) e Proto |
| 20 | Conv 3×3 s2 | m19 | 128×10×10 | 1 | 147.584 | 14,7 | |
| 21 | Concat | m20, m10 | 384×10×10 | – | – | – | |
| 22 | C3k2 (var. C) | m21 | **P5** 256×10×10 | 9 | 461.504 | 46,0 | alimenta cabeça (nível 2) e Proto |
| 23 | Segment26 | P3, P4, P5 | `output0`, `output1` | 40 | 433.711 | 525,0 | ver §5.6 e §5.7 |

Segment26 é 47% dos MACs. Dentro dela (MMACs): box 30,8 · cls 23,0 · coef 73,1 · **Proto 398,1** (só `proto.cv2`, a 3×3 a 80×80, são 235,9).

## 4. Catálogo de operações

### 4.1 Kernels de cálculo

| Operação | Ocorrências | Parâmetros / semântica |
|---|---|---|
| **Conv2d + bias (+ SiLU)** | 117 | 1×1 SiLU: 45 · 1×1 linear: 16 · 3×3 s1 SiLU: 41 · 3×3 s2 SiLU: 7 · DW 3×3: 8 (6 SiLU + 2 lineares). Pesos OIHW `[Cout][Cin/g][k][k]` |
| **ConvTranspose2d 2×2 s2** (+bias, linear) | 1 | `proto.upsample`, 64→64, 40×40→80×80. Sem sobreposição: `out[co,2y+dy,2x+dx] = b[co] + Σ_ci in[ci,y,x]·W[ci,co,dy,dx]`. Pesos `[Cin][Cout][2][2]` |
| **Add** elemento a elemento | 23 | 13 residuais de Bottleneck, 1 do SPPF, 4 dos PSABlock (2 por bloco), 2 somas `o + pe` da atenção, 2 do Proto e 1 do decode (`anchor + rb`) |
| **MaxPool 5×5 s1 pad2** | 3 | SPPF; borda ignora o padding (como −∞) |
| **Resize nearest** | 4 | ×2 em m11, m14 e `proto` (P4); **×4** em `proto` (P5: 10→40). `asymmetric`/`floor`: `src = floor(dst/escala)` = replicação de pixel |
| **MatMul em lote** | 4 | 2 por bloco de atenção: `[2,100,32]·[2,32,100]` e `[2,64,100]·[2,100,100]` |
| **Softmax** | 2 | eixo = último (100), sobre `[2,100,100]` |
| **Mul por escalar** | 2 | escala da atenção `0.17677669 = 32^-0.5`, aplicada em `q` |
| **Decode da caixa** | 1 trecho | `Slice` 0:2 e 2:4, `Sub`, `Add`, `Concat`, `Mul` por stride (§5.6) |
| **Sigmoid** | 1 | só nos 2100 logits de classe (os outros 99 viraram SiLU na conv) |
| **ReduceMax / TopK / GatherElements / Mod / Cast** | 1 / 2 / 4 / 1 / 1 | seleção das 300 detecções (§5.6) |

### 4.2 Operações de layout (sem cálculo)

| Operação | Ocorrências | Observação |
|---|---|---|
| Split | 12 | 9 em canais (metades de C3k2/C2PSA), 2 na divisão q/k/v da atenção (`[2,128,100]` → 32/32/64) e 1 no head (`[2100,37]` → 4/1/32) |
| Concat | 25 | 19 em canais (blocos e neck); 6 no head (3 de níveis ao longo dos 2100 anchors, xyxy, `[box\|score\|coef]` e a saída final) |
| Reshape | 15 | reinterpretação de memória (`256×10×10` ↔ `2×128×100`, `C×H×W` ↔ `C×(H·W)`) |
| Transpose | 5 | 2 na atenção (`q` e a matriz de atenção) e 1 `[37,2100]→[2100,37]` no head. Dá para absorver na indexação do matmul |
| Unsqueeze, Flatten, Expand | 4, 1, 2 | só ajuste de shape para os `GatherElements` |

## 5. Blocos compostos (fluxo exato, conferido tensor a tensor)

### 5.1 C3k2, variante A (m2, m4; `c3k=False`)
```
y = CBS(1)(x)               # C → 2c
a, b = split(y, [c, c])
m = b + CBS(3)(CBS(3)(b))   # Bottleneck c → c/2 → c, residual com b
out = CBS(1)(concat(a, b, m))   # 3c → Cout
```
m2: c=16 (cv1 32→32, bottleneck 16→8→16, cv2 48→64). m4: c=32 (cv1 64→64, 32→16→32, cv2 96→128).

### 5.2 C3k2, variante B (m6, m8, m13, m16, m19; `c3k=True`)
```
y = CBS(1)(x); a, b = split(y, [c, c])
u = CBS(1)(b)                 # m.0.cv1: c → c/2
v = CBS(1)(b)                 # m.0.cv2: c → c/2   (ramo paralelo)
u = u + CBS(3)(CBS(3)(u))     # Bottleneck 1 (c/2)
u = u + CBS(3)(CBS(3)(u))     # Bottleneck 2 (c/2)
m = CBS(1)(concat(u, v))      # m.0.cv3: c → c
out = CBS(1)(concat(a, b, m)) # 3c → Cout
```

### 5.3 C3k2, variante C (m22; `c3k=True, attn=True`)
```
y = CBS(1)(x) (384→256); a, b = split(y, [128, 128])
m = b + CBS(3)(CBS(3)(b))     # Bottleneck 128 → 64 → 128, residual com b
m = PSABlock(m)
out = CBS(1)(concat(a, b, m)) # 384 → 256   (note: b original, não m pré-PSA)
```

### 5.4 SPPF (m9)
```
y0 = CB(1)(x)                 # 256→128, SEM SiLU
y1 = maxpool5(y0); y2 = maxpool5(y1); y3 = maxpool5(y2)   # em cascata
z  = CBS(1)(concat(y0, y1, y2, y3))   # 512 → 256
out = x + z                   # shortcut
```

### 5.5 C2PSA (m10) e PSABlock (também dentro de m22)
```
C2PSA:  y = CBS(1)(x); a, b = split(y, [128,128]);  b = PSABlock(b);  out = CBS(1)(concat(a, b))
PSABlock(x):  x1 = x + Attn(x);   out = x1 + CB(1)(CBS(1)(x1))    # FFN: 128→256 (SiLU), 256→128 (linear)
Attn(x):   (2 cabeças, 100 tokens = 10×10)
  qkv = CB(1)(x)                              # 128→256, linear
  reshape [2][128][100] → por cabeça: q[32][100], k[32][100], v[64][100]
  A   = softmax_j( (0.17677669·q)ᵀ @ k )      # [100×100]; linha i = query i
  o   = v @ Aᵀ                                # o[c,i] = Σ_j v[c,j]·A[i,j]  → [64][100] por cabeça
  o   = reshape → [128][10][10]               # canal = cabeça·64 + c
  pe  = DW3x3(v reformatado em [128][10][10]) # linear, pad 1
  Attn = CB(1)(o + pe)                        # proj 128→128, linear
```

### 5.6 Segment26: ramos, decode e seleção
Para cada nível i com Pi (Ci×Hi×Wi; Ci = 64/128/256; Hi = 40/20/10; stride = 8/16/32):
```
box : CBS(3)(Ci→16) → CBS(3)(16→16) → CB(1)(16→4)                      # (l,t,r,b)
cls : DW-CBS(3)(Ci) → CBS(1)(Ci→64) → DW-CBS(3)(64) → CBS(1)(64→64) → CB(1)(64→1)
coef: CBS(3)(Ci→32) → CBS(3)(32→32) → CB(1)(32→32)
```
Cada saída vira `[canais, Hi·Wi]` (índice `y·Wi + x`) e os 3 níveis são concatenados: 1600 + 400 + 100 = **2100 anchors**.

Decode (verificado contra as constantes do ONNX): anchor `(cx, cy) = (x + 0.5, y + 0.5)` e `s` = stride do nível:
`x1 = (cx−l)·s`, `y1 = (cy−t)·s`, `x2 = (cx+r)·s`, `y2 = (cy+b)·s`. As tabelas de anchors e strides do ONNX (`[1,2,2100]`, `[1,2100]`) são exatamente essa geração analítica (canal 0 = x, canal 1 = y), então **não precisam ir no blob**. `score = sigmoid(logit)`.

Seleção: monta `[2100, 37] = [box(4) | score(1) | coef(32)]`; `TopK(k=300)` sobre o score (descendente, sem limiar e sem NMS); `GatherElements` das linhas; um segundo `TopK(300)` sobre os 300 scores e `idx % NC` para a classe. Saída: `[x1,y1,x2,y2, score, cls, coef[32]]`.

### 5.7 Proto26
```
r4 = Resize×2( CBS(1)(P4: 128→64) )     # 20→40
r5 = Resize×4( CBS(1)(P5: 256→64) )     # 10→40
f  = P3 + r4 + r5                        # ordem: (P3 + r4) + r5
f  = CBS(3)(f)                           # proto.feat_fuse
f  = CBS(3)(f)                           # proto.cv1
f  = ConvT2x2s2(f)                       # 64→64, linear, 40→80
f  = CBS(3)(f)                           # proto.cv2  (80×80, maior custo do modelo)
output1 = CBS(1)(f)                      # proto.cv3: 64→32
```

### 5.8 Fora do grafo ONNX (a implementar em C à parte)
- **Pré-processamento:** letterbox para 320×320, RGB, divisão por 255, NCHW (padrão Ultralytics; não está no ONNX).
- **Pós-processamento das máscaras:** filtrar por limiar de confiança; `mask = sigmoid(coef[32] · proto[32×6400])`, recorte pela caixa e mapeamento de volta à imagem original. O ONNX só entrega `output0` e `output1`.

## 6. Blob estático de pesos

- **Arquivo:** `yolo26n_seg_weights.bin`, 2.689.168 floats = 10.756.672 bytes, float32 little-endian, sem cabeçalho (o tamanho está em `YOLO_WEIGHTS_BYTES`).
- **Ordem:** ordem de execução do C: módulos 0→22; no módulo 23, por nível (box, cls, coef do nível 0, depois 1, depois 2) e por fim Proto (`feat_refine.0`, `.1`, `feat_fuse`, `cv1`, `upsample`, `cv2`, `cv3`). Camadas 0–77 = backbone/neck, 78–117 = Segment26. A leitura dos pesos fica sequencial.
- **Por camada:** `pesos` seguidos de `bias`, cada tensor começando em múltiplo de 16 floats (64 B, linha de cache do Pi). Padding total: 89 floats.
- **Layout dos pesos = o do ONNX, sem reordenar:** conv `[Cout][Cin/g][k][k]`. Para 1×1 isso já é a matriz `[Cout][Cin]` e para 3×3 é `[Cout][Cin·9]`, servindo tanto a laço direto quanto a GEMM; DW é `[C][9]`. ConvT `[Cin][Cout][2][2]`. Qualquer reempacotamento para NEON pode ser feito na inicialização, na fase de otimização.
- **Acesso em C:** `yolo26n_seg_layout.h` define `enum ConvId` (`CV_model_6_m_0_cv1`, …, 118 entradas) e `YOLO_CONV[id]` com `w_off`, `b_off` (offsets **em floats**), `cin`, `cout`, `groups`, `k`, `stride`, `pad`, `act`, `kind`, `h_in/w_in/h_out/w_out`. `CONV_W(blob,id)` e `CONV_B(blob,id)` devolvem os ponteiros. Os nomes `CV_*` seguem o `state_dict` do PyTorch (rastreáveis ao `.pt`).
- **Constantes do modelo** (`NC`, `NM`, `N_ANCHORS`, `TOPK`, formas de entrada/protótipo, parâmetros da atenção) estão no mesmo header.

## 7. Tensores

`yolo26n_seg_tensors.csv` lista os 252 tensores (shape, nº de elementos, operador produtor, operadores consumidores) e `yolo26n_seg_ops.csv` os 234 operadores em ordem topológica, com ids de tensor de entrada/saída. São a base da análise de tempo de vida da etapa 2. O maior tensor tem 409.600 floats (1,6 MiB; ex.: 16×160×160 e 64×80×80).

## 8. Validação feita

1. Pesos do `.pt` (BN fundida à mão) × `.onnx`: máx. 7,3e-6, 118 camadas.
2. Grafo fundido executado em Python (convs, split/concat, atenção, decode, TopK) × `onnxruntime`, entrada aleatória: `output1` máx. 6e-6; `output0` com as mesmas 300 detecções como conjunto (erro máx. 2e-4 em valores de centenas de pixels). A ordem das linhas só muda onde scores empatam (entrada de ruído).
3. Blob relido via offsets do header e comparado bit a bit com os inicializadores do ONNX (118 camadas); header compilado em C11 com `-Wall -Wextra -Wconversion` sem avisos e testado lendo o blob com `fread`.

## 9. Pontos em aberto (não alterei nada; decisões suas)

- **nc=1:** o segundo `TopK` reordena 300 valores já ordenados e `idx % 1 = 0`. Mantive como está no grafo; dá para simplificar depois.
- **Carga do blob:** `fread` para um array estático alinhado a 64 B, ou embutir no binário (`.incbin`). O layout serve para os dois.
- **Anchors/strides:** gerar na inicialização (analítico, verificado) ou guardar tabelas estáticas.
