# Waymo DDW preparation

Папка владеет одним явным producer `ddw_preparation/v1`: выборкой кадров,
FRONT raster/LiDAR, MoGe metric depth, траекторией DDW, двумя warp-проходами
и model-ready артефактами. Реализованная поверхность:

- `spec.py` — строгие значения job/recipe без открытия данных и моделей;
- `selection.py` — ordered Waymo sample IDs, segment/start и DDW-смещение.
- `source.py` — точный 121-frame запрос существующему Waymo v2 reader,
  audited physical frame keys и одноразовый поток frame bundles без
  предварительной материализации.
- `raster.py` — concrete FRONT measured K/W2C, projection/support и все
  LiDAR returns в sparse camera-Z с неизменным optimizer/evaluation split;
  совпадающие alpha-zero/crop/remap и range-image→world переиспользуются из
  соседнего `waymo_depth/`.
- `depth.py` — один standalone MoGe-v1 request и per-frame relative-L1
  weighted-median scale только по non-heldout LiDAR; measured K/W2C
  сохраняются без model-predicted camera.
- `target_path.py` — чистая 121-frame cosine-траектория: lateral
  displacement и virtual W2C без sample identity, intrinsics и повторной
  проверки внутренних массивов.
- `warp.py` — один one-source/one-target NPY process boundary, который
  нормализует canonical uint8 RGB покадрово либо принимает фактический
  normalized результат первого прохода; result содержит только RGB/depth/
  known и телеметрию.
- `_warp_worker.py` — private owner literal Gen3C sanitization, reliable
  depth, unprojection, splat, chunk size 2 и hole convention. Torch и
  upstream Gen3C впервые импортируются только внутри worker-процесса.
- `condition.py` — явный A→B→A′ порядок: второй pass получает фактические
  RGB/depth/known первого, ones rectification, virtual source W2C и measured
  K/target W2C; постоянный результат содержит только A′ RGB/known и
  телеметрию.
- `bake.py` — начинается с временного condition transport: точный покадровый
  normalized TCHW → uint8 THWC round и little-endian pack/unpack known.
  Здесь же controller один раз на dataset запускает installed Gen3C
  empty-prompt worker, переводит один target/condition в batch scratch и
  запускает одним process path-only VAE tasks, уже сгруппированные
  `iter_vae_batches` по восемь; повторной проверки размера группы нет.
- `_prompt_worker.py` — one-shot T5 owner: `encode_prompts([""], 512)`, exact
  first-token mask, zero padded tail и прямой contiguous CPU BF16 prompt
  artifact без runtime scan upstream checkout.
- `_vae_worker.py` — bounded tokenizer/VAE owner: один model load на batch,
  BF16-first clean target, float32-first DDW condition, clean/source encode
  и pose encode с прямой записью `base.pt`/`pose.pt`. Condition читается
  без повторной проверки временного writer; target guard сохранён для
  самостоятельного внешнего входа model-ready диагностики.
- `artifacts.py` — четыре versioned `.pt` контракта: base latents, DDW pose,
  empty prompt и единственная постоянная depth-разметка — sparse measured
  FRONT-A LiDAR с исходным optimizer/evaluation split. Torch импортируется
  только при фактическом чтении или записи.
- `record.py` — один `novel-view/waymo-ddw-prepared/v1`: сохраняет selection
  order, 121 фактически прочитанный timestamp и относительные
  `base.pt`/`pose.pt`/`lidar-depth.pt`, без scan/reuse/prepared index.

`workflows/ddw_preparation.py` связывает эти владельцы одним явным
`run_v1()`, а runner выбирает literal `moge` и `inference_cp1`. Здесь не
создаются общие manager, registry, filesystem preflight, scan/reuse или
training/evaluation state.

`legacy/` — отдельные исторические DDW canary/probe/survey, перенесённые в
Stage 9.7. Они не подменяют этот поддерживаемый producer; совпадающие
траектория и one-source warp переиспользуются явно.
