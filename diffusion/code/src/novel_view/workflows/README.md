# Рабочие сценарии

Папка содержит небольшие явные композиции уже типизированных предметных
модулей. Научная математика, model workers и форматы records принадлежат
своим областям, а workflow задаёт только порядок одного `run_vN()`.

`runner.py` лениво выбирает точную пару `(workflow.name, version)`, image
variant и execution preset, а затем ведёт operational attempt lifecycle.
Сейчас через него доступны:

- `_lifecycle_smoke/v1` — синтетическая проверка process/lifecycle;
- `euvs_source_views/v1` — first-seen ordered source tuples и прямые
  VGGT records в `source-views/<first-token>/`; внешний job принимает только
  `inference_cp1`;
- `euvs_generation/v1` — autoregressive Gen3C order;
- `euvs_generation/v2` — source-reseed Gen3C order;
- `euvs_evaluation/v1` — точные generation references, последовательная
  masks/support/metrics композиция и прямые `metrics/<pair-name>.json`.
- `euvs_comparison/v1` — две готовые evaluation attempts, точные metric/run
  records и прямые `frames.csv`, `pairs.csv`, `summary.csv` без моделей.
- `gaussian_generation/v1` — один strict producer/selection, ordinary
  full-sequence conditioning, одна resident Gen3C session и прямые
  `generated_rgb.npy`, `targets.mp4`, `run.json`.
- `gaussian_generation/v2` — exact ordered independent clip IDs, отдельные
  `clips/clip-NNN/`, right-aligned final clip и fresh source conditioning
  каждого запроса через одну resident session.
- `gaussian_generation/v3` — exact dense source range и reconstruction
  table, ordered independent chunks в `chunks/chunk-NNN/`, fresh source
  каждого чанка, hold-last хвоста и прямые Gaussian records v2 через одну
  resident session.
- `gaussian_generation/v4` — тот же exact dense input с
  `dense_overlap21` selection: 100 owned и до 20 trailing targets, fresh
  physical source каждого чанка, explicit previous RGB/depth/valid, одна
  resident Gen3C+MoGe session и Gaussian records v3 с seam MSE/PSNR. Image
  variant — literal `moge`.
- `gaussian_generation/v5` — две exact dense attempts, ordered handoff IDs,
  literal `core`/`cpu_test` и прямые `independent/selected/` и
  `overlap21/selected/` с native PNG, общей camera table и `prepare.json`.
- `waymo_keysets/v1` — исторические fit/dev/debug из official
  `training`, один central121 key на segment до RGB/геометрии и три
  прежних keyset JSON под текущим attempt. Он выбирает
  literal `core`/`cpu_test`; полный split остаётся в локальной selection.
- `waymo_depth_comparison/v1` — один explicit 121-frame clip, общий
  трёхкамерный raster/LiDAR, затем MoGe и VGGT. Outcome MoGe записывается до
  старта VGGT; неожиданная ошибка второго метода не стирает первый файл.
  Workflow выбирает `moge`/`inference_cp1`.
- `waymo_depth_selection/v1` — восемь ordered candidate v2 records и
  исходный heldout-count gate; известный отрицательный исход остаётся STOP.
- `waymo_depth_selection/v2` — те же восемь records, candidate-neutral
  evidence и исправленный gate. Обе версии выбирают `core`/`cpu_test`, не
  читают Git/SHA и пишут selection v4 с выбранным методом либо явным
  `scientific_stop`. Оба предметных исхода завершают вычисление с кодом 0;
  исключение чтения/процесса/кода доходит до runner.
- `waymo_ddw_canary/v1,v2` — отдельные исторические оси d1..4 и d2..4;
  v2 пишет первый scientific_stop и возвращает 2. V1 headroom error
  остаётся обычной ошибкой до reference.
- `waymo_ddw_probe/v1`, `waymo_ddw_survey/v1` — полная ось d3/4 либо
  d1..4, пустой reference только у failed candidate, итоговый код 0/2.
  Все четыре формы используют moge/CP1 и явный clip/depth-selection.
- `waymo_ddw_fit/v1` — full-keyset assignment, затем последовательный
  явный subset; accepted/rejected item records и accepted-only condition.
- `waymo_ddw_fit_collection/v1` — только явно названные clip/v2 records,
  порядок полной оси и complete/incomplete без чтения моделей/NPY. Payload
  остаётся привязан к папке исходного item record, не к collection.
- `waymo_ddw_audit/v1` — один rejected clip/v1 либо v2; отдельный audit
  gate до модели, rerender/сравнение, condition и preview. Accepted set не
  меняется. Fit/audit используют moge/CP1, collection — core/cpu_test.
- `ddw_preparation/v1` — ordered Waymo selection, существующий v2 reader,
  FRONT/LiDAR, MoGe metric depth, cosine target path, A→B→A′ и bounded
  prompt/VAE bake. Он выбирает literal `moge`/`inference_cp1`, сохраняет
  sparse measured LiDAR на item и пишет `prepared.json` последним.
- `gen3c_training/v1` — поддерживаемый R4c LoRA baseline: один явный
  `torchrun` запускает `_worker_v1.py`, который читает PreparedRecord/split,
  выполняет fresh либо exact resume и после каждой эпохи пишет соседние
  PT/JSON checkpoint. Core workflow не импортирует Torch/Cosmos.
- `gen3c_training/v2` запускает отдельный `_worker_v2.py`: fresh сначала
  подгоняет и замораживает depth observer, resume восстанавливает его без
  fit, затем отдельный loop выполняет прежний EDM forward, CP-global sparse
  depth loss и v2 PT/JSON checkpoints. Условия внутри v1 не добавлены.
- `gen3c_checkpoint_selection/v1` на `core`/`cpu_test` читает только явно
  перечисленные runs-relative checkpoint JSON, выбирает минимум по
  `(validation_loss, completed_step)` и пишет `selection.json`; каталог,
  stdout и binary PT он не просматривает.
- `gen3c_checkpoint_selection/v2` использует ту же явную границу, требует
  одну objective и lineage замороженного observer и выбирает минимум по
  `(validation_total_loss, completed_step)`.
- `ddw_evaluation/v1` читает exact PreparedRecord/validation split и явные
  matched v1/v2 checkpoint records, одним CP4 worker получает base/v1/v2,
  затем последовательно декодирует, запускает MoGe и сохраняет heldout
  JSON+MP4. Временные arrays принадлежат одному item и удаляются до
  следующего; discovery, request record и final-only gate отсутствуют.

Каждый EUVS workflow строго разбирает собственный job, выбирает literal image
и исполняет selection без внутренних GPU scheduler, locks, directory
discovery, reuse, resume либо `check-only`. Comparison использует `cpu_test`
с нулём GPU и не вызывает evaluation скрыто. Первая настоящая ошибка проходит
в runner и завершает attempt.

Canonical EUVS и Gaussian callers принадлежат перечисленным versioned
workflows; старые `euvs_vggt` и `gaussian_scene_gen3c` owners удалены.
Старые training/evaluation и исторические Waymo-кампании остаются только до
доказанного переключения последних потребителей; поддерживаемые preparation
и baseline training уже принадлежат versioned workflow.

Новый файл добавляется только вместе с реальным runner route. Другой научный
порядок получает новый `run_vN()`, а общий BaseWorkflow/registry/coordinator
не создаётся.
