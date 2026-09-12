# Пакет `novel_view`

Основные группы модулей:

- `config` — однократное чтение job/recipe; научные поля принадлежат
  конкретным сценариям, пути машины — хостовому профилю;
- `inputs.types`, `inputs.nuplan`, `inputs.euvs`, `inputs.waymo` — общий
  маленький `FrameRef`, строгая EUVS selection-схема без путей машины,
  concrete nuPlan/EUVS ingress и лёгкие физические
  Waymo camera/LiDAR/frame records, key-only Parquet index, соседние decoder
  и одноразовый streaming reader с устойчивыми ordered IDs без общего reader
  interface;
- `inputs.gaussian.spec`, `inputs.gaussian.reader` — пять явных Gaussian
  selection kinds с ordered clip/dense IDs и единственный concrete decoder
  точного `gaussian_depth_export/v1` JSON и владелец неизменяемых
  `GaussianFrame`/`GaussianExport`; он сохраняет producer-order, соединяет
  относительные frame paths с `info_file.parent` и открывает ровно запрошенный
  RGB/depth/mask triplet без полного preflight;
- `geometry.camera`, `geometry.trajectory`, `geometry.raster`,
  `geometry.polyline`, `geometry.sim3`, `geometry.depth` — чистые
  rigid-преобразования,
  численно устойчивые
  относительные W2C, кратчайшая SO(3)-интерполяция, reconstruction camera
  codec, cover/crop-геометрия, planar polyline projection и orientation-first
  Sim(3) camera packs и лёгкий численный `PosedDepthSequence` без
  dataset/model policy;
- `runtime.process` и `runtime.distributed` — общий no-timeout запуск
  одноразовых и torchrun worker в отдельных process group;
- `diagnostics` — явно вызываемые CP4/A100, Gen3C model-ready и Waymo
  ingress/raster проверки; импорт самого пакета остаётся холодным, а обычные
  CLI/plan/run эту область не загружают;
- `preparation.waymo_ddw.legacy` — исторические canary, gate, fit и audit;
  `preparation.waymo_depth` — прежняя трёхкамерная depth preparation.
  Переходный пакет `finetuning` удалён после переноса последних потребителей;
- `models.vggt` — единый source-neutral raw VGGT-Omega worker/protocol,
  постоянный record, explicit camera alignment и metric posed-depth adapter
  для EUVS и Waymo с фиксированным контейнерным Python и без общего timeout;
- `generation.euvs.source_views` — измеренный EUVS camera-pack,
  exact ordered source-tuple deduplication, source RGB/raster,
  source-neutral VGGT request и привязка model-owned record к физической
  source-последовательности; frozen DA3 получает тот же measured input;
- `workflows.euvs_source_views` — `euvs_source_views/v1`, literal
  `core`/`inference_cp1`, прямые attempt paths и последовательный model seam
  без внутренних GPU scheduler, scans/reuse и `check-only`;
- `generation.euvs.plan` — одна concrete связь физической EUVS-пары,
  отдельного source RGB, готовой source-геометрии и измеренных target `K/W2C`
  с mapping/timeline/conditioning и лёгким Gen3C request;
- `generation.euvs.record` — строгие читаемые EUVS generation records v1–v5
  на существующих `EuvsPairSelection/EuvsFrameSelection` и прямой writer
  прежних JSON fields/direction; v5 сохраняет ранее потерянную LoRA strength;
- `generation.gen3c.timeline` — exact target-якоря и ленивые окна временной
  оси закреплённого Gen3C;
- `generation.gen3c.conditioning` — общий anchor-local rebase, компактная
  direct query-траектория и projected-nearest single-source план над общей
  `geometry.trajectory`, материализуемый по одному окну без RGB/depth-тензоров;
- `generation.gen3c.windows`, `generation.gen3c.context_depth` — чистые
  решения по seed/seam каждого окна и численное выравнивание готовой MoGe
  depth с Cache4D reference без Gaussian/model/process-владения;
- `models.gen3c.cache4d` — холодные source-row типы, единый NPY-протокол,
  тяжёлая сборка/рендер официального Cache4D и отдельный диагностический
  backend для EUVS/Waymo-подобных численных запросов;
- `generation.gen3c.request`, `generation.gen3c.protocol`,
  `generation.gen3c.euvs_input` — тонкий generation request, только его
  дополнительные поля и явный EUVS-источник без файловых имён;
- `generation.gen3c.model_request` — конкретный base/full/LoRA request,
  общий для двух фактических потребителей без model registry;
- `generation.gaussian.full`, `generation.gaussian.clips`,
  `generation.gaussian.dense`, `generation.gaussian.handoff`,
  `generation.gaussian.record` — ordinary
  24-fps/camera/slot science, exact right-aligned independent clips,
  independent/overlap dense plans, previous RGB/depth/valid seam, exact
  record-to-native-PNG join и concrete Gaussian `run.json` v1–v3 без общего
  runtime либо model ownership;
- `preparation.waymo_ddw.{spec,selection,source,raster,depth,target_path,warp,condition,bake,artifacts,record}` — strict
  job/recipe/selection contracts `ddw_preparation/v1`, ordered
  устойчивые sample IDs и точный 121-frame seam существующего Waymo v2
  reader; concrete FRONT raster владеет alpha-zero 20:11 rectification,
  measured K/W2C и sparse LiDAR camera-Z с неизменным evaluation holdout.
  Concrete MoGe consumer вызывает существующий standalone backend и
  восстанавливает metric depth только по non-heldout LiDAR. Parquet и модель
  открываются лишь явными вызовами соответствующих функций. Чистый
  target-path owner строит точный cosine lateral displacement и меняет
  только X-трансляцию копии measured W2C, не владея intrinsics. Один
  concrete forward-warp boundary покадрово пишет normalized NPY, а private
  worker владеет literal Gen3C sanitization/unprojection/splat без SHA и
  предварительного filesystem audit или пути к runtime checkout. `condition`
  буквально связывает два
  прохода A→B→A′, используя фактические RGB/depth/known первого как источник
  второго и возвращая только итоговые RGB/known. Начало `bake` покадрово
  переводит их в точный uint8 THWC и little-endian packed-known scratch
  transport без постоянной pixel-публикации. `artifacts` напрямую читает и
  пишет versioned base/pose/prompt и sparse measured FRONT-A LiDAR `.pt`,
  не сохраняя dense MoGe либо synthetic B/A′ depth. Private prompt worker
  один раз вызывает установленный pinned T5 encoder, сохраняет exact
  mask/padded-tail semantics и пишет dataset-wide CPU BF16 artifact. Private
  VAE worker одним model load обрабатывает до восьми staged path descriptors,
  сначала публикует staged sparse LiDAR в Torch `.pt`, затем сохраняет разные
  target/condition normalizations и clean/source/pose order. Родительский
  core-процесс Torch не импортирует.
  `record` напрямую пишет и строго читает один ordered portable
  `prepared.json` без training state или filesystem index;
- `training.gen3c` — один поддерживаемый R4c LoRA baseline v1 и соседний
  экспериментальный LiDAR-depth v2: строгие split/spec, lazy prepared data,
  exact CP4 Kendall EDM, concrete LoRA method, PT/JSON checkpoint+resume и
  явная checkpoint selection без общего trainer, scan или compatibility
  gate; v1 не импортирует v2, а v2 добавляет frozen depth observer только
  поверх общего baseline forward;
- `evaluation.ddw` — matched `ddw_evaluation/v1`: exact PreparedRecord,
  validation split и v1/v2 checkpoint records, base/v1/v2 sampling на одном
  step/seed/schedule, общие VAE/MoGe загрузки, nonheldout scale fit,
  heldout depth/RGB metrics и постоянные JSON+MP4. Latents, decoded arrays и
  dense depth живут только во временной папке одного item;
- `evaluation.euvs.samples` — собственные малые типы точного generation
  record, записанный target/output/source order и target-only rasterization;
- `evaluation.euvs.masks` — direct token PNG cache, native/raster-grid
  dynamic masks и однозначная boolean polarity без sidecar/scan;
- `evaluation.euvs.support_projection`, `evaluation.euvs.support_views` —
  four-neighbour z-buffer и две независимые support-дорожки с сохранённым
  disocclusion;
- `models.grounded_sam2` — составной raw Grounding DINO + SAM2 worker без
  доступа к EUVS/cache/support; `evaluation.euvs.masks` владеет
  EUVS cache/order/polarity без generated RGB;
- `models.lpips` — точный in-process LPIPS AlexNet load/forward внутри
  существующего evaluation worker без собственного process или NPY;
- `models.dinov2` — точный in-process ViT-B/14 load, preprocess и raw patch
  features с кратковременным подключением закреплённого внешнего checkout;
- `metrics.image` — dataset-neutral exact uint8 PSNR, Gaussian SSIM и
  support-weighted readout raw LPIPS/DINO значений без model imports;
- `metrics.aggregate` — equal-camera macro и одинаковый вес полных
  pair/location rows;
- `evaluation.euvs.execute`, `evaluation.euvs.record` — одна композиция
  CPU/LPIPS/DINOv2 и строгий metric-result v1 без соседних artifact checks;
- `evaluation.euvs.benchmark` — paired base/tuned compatibility, frame/pair/
  location/global строки и три прямых CSV над готовыми records;
- `workflows.euvs_generation` — `euvs_generation/v1–v2`: одна strict selection,
  точный source-view attempt, selection-order пары, одна resident Gen3C
  session, явный autoregressive либо source-reseed order и прямые
  `pairs/<name>/{generated_rgb.npy,run.json}` для
  launcher-assigned CP1/CP2 без queue/scan/reuse/resume/check-only;
- `workflows.euvs_evaluation` — `euvs_evaluation/v1`: точные generation
  attempt/record references, последовательная masks/support/metrics
  композиция и прямые `metrics/<pair-name>.json` без discovery/reuse;
- `workflows.euvs_comparison` — `euvs_comparison/v1`: одна selection, две
  ready evaluation attempts, `core`/`cpu_test` и никакого скрытого запуска
  evaluation или models;
- `workflows.gaussian_generation` — `gaussian_generation/v1–v5`: один exact
  producer JSON, full sequence, ordered clips, dense independent либо
  overlap21, одна resident session и прямые NPY/MP4/record без
  scan/reuse/check-only; v4 выбирает literal `moge` и передаёт previous seam
  только следующему чанку, а pure-CPU v5 публикует две exact dense attempts
  как native PNG и `prepare.json`;
- `workflows.ddw_preparation` — один явный `ddw_preparation/v1`: selection,
  Waymo FRONT/LiDAR, MoGe, cosine path, A→B→A′, prompt/VAE bake и финальный
  ordered `prepared.json`; он выбирает literal `moge`/`inference_cp1`, пишет
  процессные логи в attempt и не вводит scan/reuse/staging;
- `workflows.gen3c_training`, `workflows.gen3c_checkpoint_selection` — две
  явные training-версии и их отдельные selection-контракты без model imports
  в parent-процессе;
- `workflows.ddw_evaluation` — один явный `ddw_evaluation/v1`, который
  последовательно связывает matched generation, decode, MoGe, heldout
  metrics и итоговые JSON+MP4 без discovery и старого request record;
- `cli/legacy.py` — только временная справка/версия с указателем на
  `./distil3d`; старые `stages`, `check-config` и `pipeline` удалены;
- Старые `euvs`, `gaussian_scene_gen3c` и compatibility
  `workflows.euvs_vggt` удалены после переключения всех production callers;
  historical record translation принадлежит конкретному evaluation reader.

Научную математику размещайте в тематических модулях с явными контрактами.
Замороженный DA3 находится вне устанавливаемого пакета в
`diffusion/code/research/da3_nested/`; production его не импортирует.
Связь стадий, файловые результаты и модельные окружения не должны проникать
в роль-независимые математические слои. Модельные адаптеры могут сохранять
собственные полные результаты, но перед общими последующими этапами обязаны
собрать один цельный `PosedDepthSequence`.

`geometry.sim3` решает orientation-first Sim(3) для predicted/measured
OpenCV W2C без dataset-специфичных типов.
`models.vggt.alignment` передаёт ему явные predicted/measured W2C: он не
меняет сетку depth, не выбирает камеры и не задаёт порог качества.

`models.vggt.posed_depth` принимает source-neutral raw prediction, цельный
alignment, source validity и ordered sequence ID. Он масштабирует camera-Z,
переносит balanced depth/confidence билинейно, пересчитывает predicted `K` на
source-grid и публикует согласованный predicted-camera-кандидат. Measured
nuPlan-камеры не подставляются без отдельной 3D-перепроекции.

`generation.euvs.plan` принимает такой цельный `PosedDepthSequence`,
отдельный source RGB и физическую пару. Он один раз сопоставляет их ordered
`sequence_id` и растровую размерность, затем добавляет measured target-камеры
в системе первой физической source-камеры.
`generation.gen3c.timeline` сопоставляет этот вход с уже существующим
target→source mapping по ordered source/target IDs и квантует только
относительные target image timestamps.
`generation.gen3c.conditioning` одним rebase переносит source и target в
систему выбранного anchor и строит query W2C/K с одним source index только
для запрошенного окна 121. Exact mapping и порядок сохраняются; промежуточный
выбор остаётся явно названным single-source baseline.

`models.gen3c.cache4d.backend` запускает standalone Cache4D через временные
NPY, а `runtime` собирает тяжёлые тензоры и вызывает сырой render. Составной
generation worker использует runtime напрямую: полные render-тензоры не
покидают его процесс, а seed/window/checkpoint остаются у generation.
