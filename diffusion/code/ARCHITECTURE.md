# Архитектура

Этот документ описывает фактический компонент `diffusion/code` после
структурных этапов 1–9, упрощения 9A и документационной упаковки Stage 10.
Его Python-контракты не зависят от положения checkout:
поддерживаемые физические пути к тяжёлым ресурсам задаются profile нового
launcher. Исторические номера этапов ниже обозначают происхождение
реализованных изменений, а не дополнительные шаги запуска.

Собственная Apache-2.0 и границы распространения находятся в `LICENSE`
и `THIRD_PARTY_NOTICES.md` корня компонента. Это дистрибутивные копии
корневых документов: `pyproject.toml` включает их в wheel/sdist штатным
`license-files`; общий builder использует закреплённый setuptools 84.
Внешние модели и их код этим не перелицензируются.

Короткий вход находится в `README.md`, команды — в корневом
`docs/quickstart.md`. Этот документ описывает актуальные научные границы
модулей, а корневая архитектура — связь компонентов. Исторические отчёты
и планы в публичный исходный снимок не включены.

Информационная заметка `vggt_usage_note(checkpoint)` принадлежит
`models/vggt/spec.py`. Два конкретных workflow передают её через
`JobSelection.usage_notes` в JSON plan и явный doctor до открытия ресурсов.
Она не добавляет проверку лицензии, веса, сети или запрет исполнения.

Старая runtime/experiment-конфигурация и исполняющий CLI удалены в 9.9.
`novel_view.cli.legacy` оставляет только временную справку/версию:
`novel-view` и `python -m novel_view` направляют к `./distil3d`.
Информационная console удаляется только после завершения Stage 11.
Пакеты `novel_view.cli` и `novel_view.config` ничего не переэкспортируют.
Корневой launcher реализует явные host/job/Waymo doctor,
воспроизводимый `./distil3d test all`, включающий containerized pytest и
настоящий host Docker lifecycle, а также config-only `plan` для
`_lifecycle_smoke/v1`, всех перенесённых EUVS/Gaussian маршрутов,
исторических Waymo keysets/depth/DDW workflows, `ddw_preparation/v1`,
двух R4c training-конфигураций, checkpoint selection и `ddw_evaluation/v1`.

Новый лёгкий путь состоит из `config/load.py` и `config/job.py`, которые один
раз разбирают строгую внешнюю оболочку, `runtime/context.py` с фиксированными
container roots, буквальных preset `cpu_test`, `inference_cp1`,
`inference_cp2` и `training_cp4`, ленивого
`workflows/runner.py` и внутренних `cli/main.py`/`cli/jobs.py`. Предварительный
selector выполняется в `core --rm`, а окончательный plan — в выбранном
`core|moge` по точному image ID, с тремя read-only config mounts.
Data/model/run roots остаются непрозрачными
host-строками. План не создаёт ID, каталог attempt или `attempt.json`, не
открывает научные входы и не импортирует legacy/model stack. Фиксированные
production targets `core` и `moge` собираются отдельной явной командой;
обычные команды не строят образы и не требуют candidate.json. Final plan
сверяет предварительные variant/preset, launcher читает labels по image ID
и проверяет, что выбранный tag не изменился. `cpu-test` остаётся только
test-owned вариантом `_lifecycle_smoke`.

Один Bash-владелец профиля выбирает первые 0/1/2/4 GPU без опроса устройств.
`HostPlanInputs` получает готовые host IDs, container-local индексы, IPC и
memlock; `build_job_plan` показывает их в `execution` и будущем argv.
Фактический `create` использует тот же выбор. CPU и control containers
получают только `--network none`; GPU-attempt дополнительно получает
quoted device CSV, локальный `CUDA_VISIBLE_DEVICES`, IPC host и unlimited
memlock. Работа оборудования остаётся предметом Stage 11.

Внутреннее исполнение `_lifecycle_smoke v1` уже получает готовый
`RuntimeContext`, использует `novel_view.runs` как единственного Python-
writer собственного `attempt.json` и запускает synthetic worker через
`runtime/process.py` в отдельной process group. Прямая запись хранит только
resolved job, image metadata, ID, состояние и worker code; host-пути,
traceback и Docker-состояние в record не копируются. Python не вызывает
Docker. Host `run/resume` уже передаёт этому срезу готовые ID, image metadata,
attempt-root и необязательный checkpoint. Launcher создаёт контейнер через
`create → start`, а foreground возвращает фактический Docker exit code после
логов. Resume восстанавливает resolved job из записи предыдущей attempt
короткоживущим reader в `core`; Bash JSON не читает. Перед `run/resume`
окончательный plan строится в выбранном exact image ID. При resume он читает
только `/runs` read-only и сохраняет прежний run ID и дословный checkpoint
в будущей команде, не обращаясь к исходному YAML. Адресация `status/logs` и
явные `stop/remove` принадлежат host launcher. `status/logs` уже принимают
run-ref либо exact attempt-ref: единственная активная attempt выбирается,
несколько дают код 2 и точные candidates, а без активной используется
последняя записанная. Переносимую запись читает только короткоживущий
`core --rm` с одним read-only `/runs`; host отдельно читает labels и
`docker inspect`. Снимок ничего не записывает, показывает Python- и Docker-
состояния рядом и не выводит OOM только из кода 137. Container без записи
адресуется только exact ref. `stop` посылает Docker SIGTERM, который
`runtime/process.py` пересылает process group; graceful worker сохраняет
`stopped/143`, а Docker SIGKILL после grace оставляет stale `running/137`.
`remove` удаляет только завершённый container, не меняя attempt-root и
record.

Тестовый target `cpu-test` воспроизводит принятый Linux/amd64
`/opt/envs/core`, собирает один wheel этого компонента и устанавливает его
без editable-режима и `PYTHONPATH`. Текущие CPU `tests/unit`,
`tests/contract` и `tests/integration` запускаются из image без GPU, данных,
моделей и сети. Полная группа `all` затем на том же host Docker проверяет
настоящий attempt lifecycle и является единственной командой CPU workflow
GitHub Actions. Этот тестовый образ остаётся отдельным от двух production
targets. Его `LD_LIBRARY_PATH` выбирает C++ runtime из `/opt/envs/core/lib`,
чтобы SQLite/ICU и Torch/PyArrow не зависели от порядка импортов.

После Stage 9.1 все CPU model/data tests включены в обычные
unit/integration, в том числе R4c v1/v2. Native Gen3C/VGGT/SAM2 cases
явно помечены `native_api` и отделены от test all. Экспериментальный DA3
находится в `tests/integration/test_da3_native_api.py` с тем же marker,
но не входит в обязательную приёмку core/moge. Папка tests/model удалена.
Старые data operators перенесены
к предметным владельцам в9.5–9.9, папка tests/data удалена. CPU Torchvision и scikit-image
с зависимостями добавлены только в test-only lock; production не изменён.
На входе9.1 Linux/Docker сбор подтвердил609 CPU tests,417 subtests и lifecycle;
тогда13 native cases исключались из CPU-фазы. Текущий компактный native
набор находится в tests/integration/test_model_api.py, включая маленький
LoRA forward; точные команды описаны в README integration. Для локального torchrun тестовый
контейнер использует hostname=localhost с отключённой внешней сетью.

`production-core` собирается от закреплённого CUDA 12.4 runtime и копирует
только установленные `/opt/envs/core`, `/opt/envs/gen3c`, `/opt/envs/vggt`,
нужные Gen3C/DINOv2 runtime sources и лицензии. Один wheel проекта установлен
во всех трёх prefix. Builder stages, Miniforge root, wheelhouse и полные
upstream checkout в финальный образ не наследуются; внутри скопированных
prefix остаются все файлы, которые требуют закреплённые runtime-зависимости.
Образ работает как пользователь `distil3d`, имеет фиксированные `/data`,
`/models`, `/prepared`, `/runs`, `/cache`, пустой entrypoint и безопасную
справку в качестве команды по умолчанию. `production-moge` наследует этот
результат и добавляет закреплённые пакеты только в `/opt/envs/gen3c`, не
меняя состав `core` и `vggt`.

`./distil3d build` материализует фиксированный tag `core`, а
`--variant moge` — согласованную последовательность `core` затем `moge`.
Обычная сборка честно записывает Git revision, dirty-state, общий lock
aggregate, variant и закреплённые upstream revisions в labels. Режим
`--candidate-output` разрешён только для чистого неизменившегося checkout и
нового каталога вне него: после CPU gate он запускает проверки prefix,
сохраняет оба exact image ID одним `docker save` и создаёт соседний
`candidate.json`.

Строгая внешняя schema candidate принадлежит
`novel_view.runtime.candidate`; `novel_view.cli.candidate` является узким
внутренним writer/reader для host launcher. Запись содержит только schema 1,
чистую 40-символьную source revision, `linux/amd64`, канонический aggregate
семи production locks и ровно два фиксированных image tag/ID. Candidate ID
выводится из первых 12 символов source/core/moge identities. Неизвестные,
отсутствующие или несовпадающие внешние поля отклоняются один раз; внутренние
dataclass повторно не проверяются. Reader печатает только известные
`key=value`, а writer прямо создаёт JSON с режимом 0644. Научные workflow и
обычный lifecycle не читают candidate. Host-side `accept` использует этот
reader для bootstrap загруженного bundle и затем сверяет точные image
ID/labels. Узкий `novel_view.cli.acceptance` внутри выбранного `core` строго
читает уже созданные v1/v2 checkpoint/selection records и печатает только
фиксированные `key=value`; Bash не разбирает JSON. Evidence layout и Docker
остаются только в host launcher.
Три соседних режима `waymo-candidate`, `waymo-depth-selection` и `waymo-ddw`
читают существующие исторические outcomes, сверяют выбранный clip/полную
восьмёрку reports/точный gate и сохранённый worker exit. Они возвращают
научный verdict и допуск отдельно; обычный `run` этот reader не вызывает.

Пакет `novel_view.diagnostics` является production-владельцем
явно вызываемых диагностик: CP4/A100/Apex/NCCL в `training.py`, Gen3C
model-ready controller и его приватного worker, Waymo v2 ingress, Waymo
raster STOP и установленный образ в `image.py`. Последняя без GPU, моделей
и данных выполняет `pip check`, проверяет нужные импорты и их происхождение,
non-root пользователя и отсутствие инициализации CUDA отдельно для prefix.
Сам `diagnostics/__init__.py` содержит только паспорт и не импортирует NumPy,
PyArrow, Torch или model stack. Обычные CLI, plan, run и workflow diagnostics
не импортируют. Старые запускаемые файлы под `tests/`, Gen3C
compatibility/runtime-check workers и отдельный training shell удалены:
model-ready использует текущие Stage 6 artifacts, а CP4-диагностика —
`runtime.presets.TRAINING_CP4`. Тестовые фабрики остаются только в
`tests/support/`; production их не импортирует.

`diagnostics/job.py` вызывается только внутренней CLI-командой `doctor`:
строго разбирает job и существующие spec/selection/records, открывает
точно выбранные файлы и вызывает нужный prefix через обычный subprocess.
Он использует ContainerRoots, но не создаёт RuntimeContext/attempt.
Приватный `_job_worker.py` импортирует только нужные backend, открывает
официальные T5/VAE/Grounding assets без загрузки моделей, проверяет CUDA
и NCCL с фактическим числом ranks. Training сохраняет отдельную узкую
CP4-диагностику; MIG адресуется по UUID фактической CUDA-карты, а не
container-local индексу. Ни один из этих путей не пишет в `/runs` и не
вызывается обычным workflow. Реальные устройства проверяются в Stage 11.
Для исторического depth comparison явно проверяются MoGe и VGGT в своих
prefix; CPU gate читает полный ordered набор восьми candidate reports.
Четыре DDW doctor используют только MoGe/Cache4D/warp, выбранный clip и
готовый depth gate, без требования Gen3C checkpoint или training CP4.

Холодный пакет `novel_view.geometry` зависит только от NumPy и стандартной
библиотеки. Он владеет rigid W2C/C2W, центрами и осями камер, `wxyz`
quaternion, численно устойчивыми relative W2C, кратчайшей SO(3)-
интерполяцией, reconstruction `xyzw` camera codec, resize-to-cover/crop и
проекцией на ориентированную полилинию, а также dataset-neutral
orientation-first Sim(3) camera packs и лёгким численным контрактом
`PosedDepthSequence`. Пакет не делает re-export и не
импортирует dataset, model, workflow или файловую политику. nuPlan wrapper,
EUVS/Gen3C, Gaussian, VGGT source и Waymo-потребители уже используют
соответствующую чистую математику этого владельца;
прежние `generation/gen3c/camera_math.py`, `raster/geometry.py` и
корневой `polyline.py` удалены.

Холодный пакет `novel_view.inputs` владеет concrete внешним чтением без
общего reader interface. `inputs.types` содержит маленький `FrameRef`;
`inputs.nuplan` — read-only SQLite batching, restricted decoder calibration
blobs, EPSG-aware camera-wrapper и единый OpenCV raster owner; `inputs.euvs`
— строгий ordered selection без host paths и physical index/pair/RGB;
`inputs.gaussian.spec` — tagged selection для full sequence, independent
clips, dense independent, dense overlap21 и dense handoff;
`inputs.gaussian.reader` — единственный decoder точного
`gaussian_depth_export/v1` JSON и неизменяемые `GaussianFrame`/
`GaussianExport`. Эти selection parser сохраняют заданный порядок concrete
ID, не сканируют данные и не загружают model/dataset stack. Gaussian-reader
сохраняет порядок producer, соединяет
относительные frame paths только с `info_file.parent` и не открывает payload,
не обходит каталоги и не применяет resolve/root-containment. Импорт паспортов
`novel_view.inputs` и его подпакетов не загружает внешние данные.

Новый лёгкий пакет `novel_view.preparation.waymo_ddw` начинает единственный
producer `ddw_preparation/v1`. `spec.py` и `selection.py` содержат strict
внешние значения Waymo v2, закреплённые MoGe/Gen3C model identities и
ordered устойчивые `sample_id` с segment/start/magnitude/sign. `source.py`
превращает один sample в точный 121-frame запрос существующему Waymo v2
reader, сохраняет audited physical `WaymoFrameKey` и отдаёт одноразовый
поток frame bundles без предварительной материализации. `raster.py` одним
проходом строит только FRONT alpha-zero remap, центральный 20:11 crop,
half-pixel `K_canvas`, measured W2C, uint8 RGB и sparse camera-Z из всех
LiDAR returns. Every-fifth evaluation labels назначаются до фильтрации
rectification support; фильтр сохраняет labels и пересчитывает только frame
offsets. Исторический трёхкамерный depth-comparison находится в соседнем
`preparation.waymo_depth` и не входит в поддерживаемый R4c producer. Парсеры
не открывают Parquet, model assets или machine roots;
холодный импорт source не загружает PyArrow/OpenCV/Torch. Исполняемый
`depth.py` передаёт ordered RGB и measured horizontal FOV существующему
standalone `models/moge` backend, затем на каждом frame вычисляет exact
positive relative-L1 weighted-median scale только по
`evaluation_holdout=False`. Dense результат остаётся camera-Z измеренной
FRONT camera с исходными K/W2C; VGGT, depth gates и model-predicted camera в
этот путь не входят. Model/process ошибки не маскируются новым wrapper.
`target_path.py` отдельно вычисляет точную 121-frame cosine-траекторию и
меняет только X-трансляцию копии measured W2C. Он не получает sample
identity или K, поэтому intrinsics остаются ответственностью measured FRONT
camera pack. `warp.py` содержит один one-source/one-target process seam:
uint8 FRONT RGB переводится в float32 TCHW `[-1,1]` по одному кадру прямо в
scratch NPY, а normalized результат первого прохода может быть передан без
8-bit round-trip. Private `_warp_worker.py` впервые импортирует Torch и
image-pinned Gen3C utility, сохраняет literal sanitization → reliable mask →
unprojection → splat и формирует hole values RGB `-1`/depth `0`. Utility
импортируется из установленного в image package без отдельного пути к
checkout. Parent возвращает фактические RGB/depth/known и telemetry без SHA,
provenance,
filesystem preflight или отдельного worker-output validator. `condition.py`
явно вызывает его дважды: A→virtual B, затем фактические
`I_B/D_B/known_B` становятся normalized source/depth-validity прохода B→A′.
Virtual source получает ones rectification и virtual W2C, а обе K и final
target W2C остаются measured FRONT A. Итог хранит только A′ RGB/known,
максимальные CUDA peaks и сумму времени; повторного audit render нет.
Первая часть `bake.py` переводит final normalized TCHW в uint8 THWC точной
покадровой формулой `rint(clip(rgb + 1, 0, 2) * 127.5)` и пакует плоский
known по кадру с `bitorder=little`. Обратное чтение учитывает только первые
`H×W` бит. Эти NPY — временный batch transport, а не постоянный prepared
формат; каталог не сканируется и предварительно не проверяется.
`artifacts.py` владеет четырьмя постоянными `.pt` форматами: BF16 base
latents, BF16 DDW pose latent, dataset-wide BF16 empty prompt и sparse
measured FRONT-A LiDAR. Последний сохраняет K, frame offsets, canvas XY,
positive camera-Z и исходный boolean evaluation holdout; dense MoGe и
synthetic B/A′ depth отсутствуют. Прямой writer использует `torch.save` без
temporary/replace/no-overwrite, а reader один раз проверяет exact fields,
format, sample/variant, CPU dtype/shape/contiguity/finiteness и предметные
LiDAR-инварианты. Torch импортируется только внутри фактического вызова.
Один вызов `bake_prompt()` запускает ровно один `/opt/envs/gen3c` process и
передаёт ему только text-encoder identity и final `empty-prompt.pt` path;
`run_v1()` вызывает его один раз на dataset.
Private `_prompt_worker.py` импортирует установленный image-pinned Gen3C
пакет без `upstream_root`/`sys.path` scan, кодирует `['']` с длиной 512,
проверяет first-token-only mask и нулевой padded embedding tail, затем
пишет contiguous CPU BF16 payload напрямую. Device/model ошибки выходят как
process failure без retry; parent import остаётся Torch/Cosmos-free.
`stage_bake_input()` пишет один measured target, окончательный DDW condition
и sparse measured LiDAR в batch scratch; в памяти следующей стадии остаётся
только `VaeBakeTask` с путями. Workflow получает группы до восьми tasks из
`iter_vae_batches()` и передаёт их `bake_items()`, который запускает один
private `_vae_worker.py` без повторной проверки размера группы. Worker в Gen3C prefix сначала
публикует sparse `lidar-depth.pt`, затем один раз загружает installed
tokenizer/VAE, делает BF16-first clean/source encode из 121 кадров и frame 0,
а после освобождения target — float32-first condition/known pose encode.
Core parent поэтому не импортирует Torch. Exact CPU BF16 base/pose payloads
пишутся напрямую; process error прекращает batch без retry или rollback
частичных файлов.
Condition reader доверяет этому временному writer; `_read_target` сохраняет
проверку uint8/THWC, поскольку его также вызывает model-ready диагностика
с самостоятельным внешним RGB-путём.
`record.py` владеет единственным постоянным списком подготовленного набора:
`novel-view/waymo-ddw-prepared/v1`. Он сохраняет exact workflow/input/model
identities, selection order, 121 timestamp и фиксированные относительные
пути `items/<sample_id>/{base,pose,lidar-depth}.pt`. Reader строго проверяет
сам внешний JSON, но не сканирует каталог, не открывает artifacts и не
классифицирует reuse/missing; training epochs/resume в record отсутствуют.
`workflows/ddw_preparation.py` теперь явно связывает весь порядок selection →
Waymo FRONT/LiDAR → MoGe → cosine path → A→B→A′ → prompt/VAE bake. Runner
выбирает literal `moge` и `inference_cp1`; output детерминированно находится
в `/prepared/waymo-ddw/<job.name>/`, процессные логи принадлежат текущему
attempt, а `prepared.json` пишется последним без staging или rollback.

Отдельный лёгкий пакет `novel_view.training.gen3c` владеет поддерживаемым
R4c baseline. `data.py` содержит строгий training/validation split по
ordered `sample_id` и закрытую 85-строчную карту старых `global_fit_index`.
`lora/spec.py` разбирает неизменный baseline v1 и хранит его фиксированные
параметры, а `lora/depth_spec.py` — только параметры экспериментального v2.
`workflows/gen3c_training.py` разрешает обе версии для `plan` как
`core`/`training_cp4` и исполняет их через разные точные
`lora/_worker_v1.py` и `lora/_worker_v2.py`. Общая узкая workflow-функция
собирает только совпадающие пути, окружение и параметры процесса. V1 не
импортирует depth-модуль; парсеры не открывают prepared artifacts, LiDAR,
модель или каталоги. V2-only `lora/depth.py` переиспользует baseline item,
присоединяет точный `lidar_depth` locator и без сканирования открывает
строгий Stage 6 artifact. Единственный reducer исключает heldout/slot 0,
берёт FP64 median log camera-Z в каждой latent cell и делит values/mask
только upstream `split_inputs_cp(seq_dim=2)`. Тот же конкретный модуль
содержит `Conv3d(16,32,1) → SiLU → Conv3d(32,1,1)` observer: source rank
подгоняет его только на training clean latents, validation сравнивает с
pooled training-median constant, после успешного signal gate четыре FP32
тензора замораживаются и раздаются всем ranks, а внешний CPU/CUDA RNG
восстанавливается. Соседний `lora/depth_objective.py` импортирует только
этот v2-контракт и baseline `EdmForwardResult`: predicted readout остаётся в
графе, clean readout отделён, mask пересекается с generation region, а
локальный backward получает `gradient_average_size * E_r / N`. Отдельный
detached report равен `sum(E_r)/N`; global zero cells останавливают шаг.
`lora/depth_method.py` явно соединяет observer fit/restore, один прежний EDM
forward, новый depth loss, validation, epoch checkpoints и узкий общий LoRA
update без изменения `run_training_step_v1()`.
`lora/depth_checkpoint.py` переиспользует ровно тело v1 state, добавляет
tagged objective, четыре contiguous CPU FP32 observer tensor, fit
identity/result и lineage fresh attempt. Его restore строит exact frozen
observer и не вызывает fit; untagged v2 не принимается. Лёгкий
`lora/depth_spec.py` является общим владельцем scientific identity для PT и
JSON, поэтому binary и readable writers не импортируют друг друга.
`lora/depth_record.py` строго читает полный v2 companion JSON и проверяет
согласованность его split с общим A/B identity. Общий
`checkpoint_selection.py` теперь имеет независимые v1/v2 функции: v2
сравнивает только objective и observer lineage, затем выбирает минимум
`(validation_total_loss, completed_step)` без PT, scan или path checks.
`training/gen3c/data.py` соединяет
ordered split с `PreparedRecord`, проверяет только split/segment identity,
сохраняет lazy base/pose/prompt locators и строит literal PCG64 order с
normalized next-item cursor. Worker-side iterator читает artifacts только
на source rank, раздаёт возможную data error до tensor collective и затем
broadcast-ит ровно clean/source/pose/prompt без LiDAR и повторной проверки
готовых tensors. `training/gen3c/lora/objective.py` стал единственным новым
владельцем baseline conditioner/noise/EDM: каждый rank делает четыре draw до
CP broadcast, split идёт только по `seq_dim=2`, condition не отбрасывается,
а Kendall mean сохраняет logvar на всём tensor. Результат возвращает
CP-local prediction/clean target/generation mask для отдельного v2, но не
импортирует его. `training/gen3c/topology.py` поднимает и один раз сохраняет
фактические CP4 и gradient-average groups без GPU/hardware preflight.
Зависимость baseline остаётся односторонней: `method → step → objective`;
v2 направлен как `depth_method → depth_objective → objective/depth` и назад
не импортируется.
`lora/method.py` переиспользует model-side exact 28-block LoRA layout,
включает CP → reentrant activation recompute → DDP, создаёт прежний
FusedAdam с constant scheduler только над LoRA и задаёт concrete epoch loop
fresh/resume. `lora/step.py` владеет одним explicit update и отдельным
validation measurement. `lora/checkpoint.py`
является единственным владельцем tagged v1 PT state и узкого read-only
untagged adapter: stable LoRA keys, FP32 master/moments, scheduler,
next-item cursor и RNG по global rank. Соседний `lora/record.py` независимо
владеет строгим читаемым JSON; writers не импортируют друг друга и пишут
напрямую без staging/no-overwrite. После каждой полной эпохи rank 0 пишет PT
и соседний JSON, а validation возвращает training RNG и mode до следующего
update. `training/gen3c/checkpoint_selection.py` не читает stdout, PT или
каталоги. Он является также строгим reader собственного переносимого v1/v2
JSON: проверяет точные поля, runs-relative locators, принадлежность выбранной
записи явному набору, metric и v2 objective/lineage. Это граница сохранённой
записи, а не scan или checkpoint compatibility gate.
`workflows/gen3c_checkpoint_selection.py` подключает обе версии
как `gen3c_checkpoint_selection/v1` и `/v2` на `core`/`cpu_test`.

Лёгкий пакет `novel_view.evaluation.ddw` владеет одним matched
`ddw_evaluation/v1`. `spec.py` строго связывает один PreparedRecord,
validation split, явные v1/v2 checkpoint JSON и общий step; соседние PT
проверяются только на соответствие этому step, а v2 также на objective,
observer fit и lineage. `samples.py` сохраняет порядок validation selection
и напрямую читает соответствующий raw Waymo FRONT target с его измеренными
K/W2C и sparse LiDAR. `_generation_worker.py` одним CP4 запуском получает
base/v1/v2 latents при одинаковых seed и schedule: base отключает LoRA, а
каждый tuned вариант отдельно заменяет исходный adapter без накопления весов.
`_decode_worker.py` одной загрузкой VAE декодирует три результата, затем одна
загрузка MoGe даёт относительную глубину для каждого варианта. `metrics.py`
подгоняет масштаб только по nonheldout LiDAR кадров 1–120, считает только
heldout depth и полный 121-frame RGB, а также создаёт один четырёхпанельный
MP4 target/base/v1/v2 с LiDAR. `execute.py` держит latents, decoded arrays и
depth только во временной папке одного sample, поэтому они удаляются перед
следующим item; постоянными остаются строгий итоговый JSON и MP4 каждого
item. `record.py` проверяет scientific identities и пересчитанные агрегаты,
но не открывает артефакты и не вводит filesystem preflight, discovery,
latest, staging либо старый request record. Явный порядок подключён через
`workflows/ddw_evaluation.run_v1()`; job с конкретными run paths остаётся в
игнорируемом `jobs/local/`.

## Границы системы

Git предназначен только для кода, небольших конфигураций, синтетических
тестовых данных и документации. Внешними ресурсами являются:

- исходные EUVS/nuPlan и Waymo под подключённым `/data`;
- веса моделей под `/models`;
- переиспользуемая подготовка под `/prepared`;
- результаты попыток под `/runs` и временный кеш под `/cache`.

Машинные корни задаются игнорируемым `profiles/local/<name>/.env` и
подключаются к этим фиксированным контейнерным путям. Установленные
окружения и нужный upstream-код входят в production-образ; сторонний
checkout не требуется от оператора для обычного запуска.

## Конвейер

Целевой пользовательский конвейер состоит из восьми последовательных этапов:

```text
extract
  → alignment
  → source_rgb
  → source_geometry
  → generate
  → target_rgb
  → masks
  → metrics
```

- `extract` выбирает строки source и target, читает связанные nuPlan-метаданные
  и строит физические камеры.
- `alignment` использует готовые центры камер для пространственного
  сопоставления проездов; timestamps не синхронизируются.
- `source_rgb` приводит source-кадры и intrinsics к единому растру.
- `source_geometry` запускает выбранный модельный модуль и возвращает
  геометрию, согласованную с метрическими source-камерами.
- `generate` подготавливает вход Gen3C и создаёт target-предсказания.
- `target_rgb` впервые открывает target RGB как GT.
- `masks` строит статическую область оценки.
- `metrics` вычисляет LPIPS, DINO, PSNR и SSIM.

Это описание научного порядка, а не исполняемый список `stages`:
прежний `pipeline.py` удалён. Поддерживаемые сквозные
композиции находятся в `novel_view.workflows`; научная математика остаётся в
тематических модулях, а не внутри файлов-обвязок.

## Конфигурации

Старые `novel_view.config.legacy` и машинный `runtime.local.yaml` удалены
из рабочего интерфейса. Пакет config читает только job/recipe формы.
Прежние EUVS `configs/experiments/*.yaml` перенесены без изменения пар в
корневые `selections/euvs/`; `jobs/legacy/euvs/` связывает четыре scopes с
существующими source-views и generation v1/v2 recipes.

Прежняя схема больше не обслуживает Gen3C training/evaluation. Поддерживаемый
путь получает машинные корни из локального profile, а научные значения — из
versioned `jobs/`, `recipes/` и `selections/`. Старые `check/start/status`,
adoption, prepare-missing и низкоуровневые Gen3C команды удалены вместе с
операторскими request/progress форматами; `./distil3d` остаётся единственным
пользовательским жизненным циклом.

Папка `configs/` удалена. Исторический полный Waymo split без изменения
сохранён в ignored `selections/local/waymo-segments-v1.yaml`; его читает
существующий `waymo_keysets/v1`. Прежние depth/DDW gate и assignment
перенесены к конкретным владельцам. Реальный DDW85 split и одноразовая карта старых
индексов находятся в исключённом из Git `selections/local/`; публичный Git
хранит только малый синтетический пример схемы.

Поддерживаемый R4c-контракт находится в `models/gen3c/spec.py`: 121 кадр,
canvas 704×1280, `fps=10` и точные формы base/pose/prompt. Подготовка пишет
versioned artifacts и один `PreparedRecord`; `training/gen3c` соединяет его
с ordered split, но baseline v1 читает только base/pose/prompt и не
импортирует v2 depth.

`training/gen3c/lora` разделяет точный Kendall EDM forward, LoRA method,
update, checkpoint и читаемый record. V1 сохраняет прежние 28 adapters,
FusedAdam, scheduler, CP4 и continuous/resume порядок. Экспериментальный v2
направленно зависит от v1 forward, добавляет frozen latent-depth observer и
глобально нормированный sparse LiDAR loss, но не меняет baseline. PT и JSON
форматы v1/v2 различаются; checkpoint selection читает только явно названные
records и не сканирует каталоги.

`evaluation/ddw` выполняет один matched base/v1/v2 порядок на одинаковом
фактическом шаге и seed. Nonheldout LiDAR используется только для масштаба,
heldout — только для depth score; постоянными результатами являются JSON и
MP4. Реальная CP4/A100/model-ready проверка остаётся явной диагностикой и
финальной кампанией Stage 11, а не скрытым условием обычного запуска.
Холодный пакет `novel_view.inputs.waymo` задаёт физический frame key, пять
камер, пять LiDAR, общий calibration context и один полный frame bundle без
ролей source/target. Он разделён на scalar types, camera, LiDAR и frame и не
переэкспортирует их из package root. Frame timestamp обозначает начало первого TOP
LiDAR scan; отдельная frame vehicle pose относится примерно к середине
frame, а не к этому key time. Слой также отделяет покамерные
времена/nominal pose и нативные Waymo/OpenCV
camera axes. Physical dataclass доверяют массивам, которые decoder создаёт
owned и read-only на внешней границе. Соседние `index`, `decode`,
`lidar_decode` и `reader` выполняют key-only аудит, выбирают row group и
потоково связывают точные camera/LiDAR decoder-ы в одноразовые frame bundle,
не удерживая всё окно. PyArrow и OpenCV загружаются только при явном чтении,
а прежний `novel_view.waymo` удалён после переключения всех потребителей.
Отдельная data-команда проверяет закреплённое окно
`exposed_debug` из 121 frame, временную семантику и численный предел RSS;
выбор рецепта и R&D-обвязка находятся за границей R4a.
Исторический recipe-neutral слой R4b теперь находится в
`novel_view.preparation.waymo_depth`. `waymo_keysets/v1` фиксирует один
центральный 121-frame key на каждый явно заданный segment ID до RGB, depth,
coverage и выбора рецепта. Три ordered keyset-файла пишутся прямо в текущую
attempt, читаются только по явному пути и не образуют registry или файловый
discovery. Соседний трёхкамерный слой строит для FRONT, FRONT_LEFT и
FRONT_RIGHT alpha-zero pinhole canvas 704×1280, прямую OpenCV remap-карту и
перевод sparse LiDAR camera projections. Он сохраняет прежние measured
K/W2C, порядок camera/frame/LiDAR/return/native-index и общий каждый пятый
heldout sample. Backend-neutral metric-depth records того же пакета принимают
только camera-Z метры на этих измеренных камерах. `clip.py` за один проход
строит shared LiDAR evidence и три временных RGB-потока. `moge.py` сохраняет
measured K/W2C и подбирает независимый положительный relative-L1 scale
каждого кадра по non-heldout точкам. `vggt.py` использует предсказанные
камеры и глубину, одним Sim(3) переносит их в измеренную систему координат и
не подгоняется по LiDAR.

`comparison.py` буквально выполняет MoGe, сразу пишет его отдельный outcome,
затем выполняет VGGT и пишет второй outcome. Только невозможность получить
научно пригодный результат модели превращается в candidate failure;
неожиданная ошибка процесса, файлового ввода-вывода или кода прерывает
attempt. `selection.py` хранит две явные версии решения над восемью ordered
clips: v1 применяет прежний минимум heldout-точек к каждому кандидату, а v2
сначала фиксирует один согласованный backend-neutral count и больше не
использует его как кандидатный gate. `candidate_record.py` сохраняет прежний
candidate v2. `selection_record.py` пишет текущий selection v4 с версией
workflow и исходом `selected` либо `scientific_stop`. При чтении старого
selection v2/v3 только его прежняя оболочка переводится отдельным
`legacy_selection_record.py`; научные поля разбираются один раз общим
читателем. Открытия соседних файлов, обращения к Git и вычисления SHA нет.

Поддерживаемый FRONT-only `preparation.waymo_ddw` переиспользует совпавшие
raster builder, RGB remap и range-image→world. Его projection без legacy
epsilon, bilinear-support фильтр и holdout остаются отдельными: исторический
трёхкамерный порядок ими не подменяется. Исторические DDW fit/audit,
canary/gate принадлежат
`preparation.waymo_ddw.legacy`, а выбор MoGe/VGGT
уже принадлежит историческому `preparation.waymo_depth`.
Поддерживаемый R4c-путь больше не использует этот старый слой: Stage 6
создаёт один ordered `PreparedRecord`, Stage 7 читает split по `sample_id`,
а `_worker_v1.py` и `_worker_v2.py` пишут раздельные versioned PT/JSON.
Model-side `models.gen3c.lora_weights` отвечает только за точную замену и
включение LoRA в generation/evaluation. Старые working manifest,
prepared-index, adoption, operator request/progress, отдельные generation/
decode/comparison workers и низкоуровневые команды удалены.

Явная model-ready диагностика по-прежнему последовательно запускает prompt и
VAE на одной выбранной GPU, но теперь использует Stage 6 artifacts и
устойчивые `sample_id/magnitude/sign`. Она не создаёт production artifacts;
полный A100/CP4 результат принадлежит Stage 11.
Общий LiDAR-evidence слой перед model adapters восстанавливает только
выбранные range-image pixels в глобальных координатах, учитывает TOP
per-pixel pose, переносит предоставленные Waymo camera projections на тот же
canvas и один раз фиксирует общий каждый пятый held-out sample. MoGe и VGGT
не реализуют эту физическую математику повторно. Конкретные model adapters
живут в `preparation.waymo_depth.{moge,vggt}` и вызывают source-neutral
`models.moge` и `models.vggt` через их существующие process seams. Тяжёлые
модели загружаются только в workers; NumPy-оркестратор остаётся лёгким.

Исторические DDW canary, probe и survey находятся в
`preparation/waymo_ddw/legacy/`. `spec.py` однократно проверяет внешнюю
форму и научную ось. `execution.py` читает явный depth-selection locator,
требует выбранный MoGe, затем строит прежний clip и получает metric FRONT
depth. `source.py` открывает уже существующий uint8 THWC RGB mmap и
заимствует depth/valid/measured K/W2C; отдельного normalized scratch нет.
Машинные пути, Git/SHA gates, directory discovery и старый preregistration
reader в этот слой не перенесены. Закрытые clip keys находятся в local
selection, ось — в recipe, depth-selection locator — в job.

`path.py` добавляет clip/variant identity к общей cosine-траектории Stage 6.
`render.py` делает два вызова того же one-source `run_forward_warp`:
`A→B→A′`, передавая во второй проход фактические normalized RGB/depth/known
первого. Собственный legacy worker и неиспользовавшаяся two-source ветка
удалены. `local.py` сводит окончательный condition к маскам и headroom,
освобождает плотные результаты до `reference.py`. Последний использует
существующий global Cache4D для всех 121 кадров, без нового renderer.

`masks.py` считает шесть прежних descriptors; `gate.py`/`v2_gate.py` и
`calibration.py`/`v2_calibration.py` сохраняют отдельные научные версии.
`selection.py` оставляет правило all-clips/both-signs без ранжирования.
`headroom.py` измеряет RGB-порчу только на known restoration pixels кадров
1–120 с прежними порогами и float32 tolerance. Это научное условие, что
вход требует восстановления, а не проверка свободной памяти.

`canary.py` выполняет исходную ось d1..4; failed headroom остаётся ошибкой
до reference. В `v2_canary.py` canary v2 останавливается на первом отказе
оси d2..4. Probe проходит всю ось d3/4, survey — d1..4; reference
пропускается лишь для не прошедших вариантов. Три явных workflow
`waymo_ddw_{canary,probe,survey}.py` подключены к runner; canary имеет
отдельные `run_v1()`/`run_v2()`. Все четыре формы используют moge/CP1.

`record.py` напрямую пишет один `result.json` соответствующей family `/v2`.
V2 canary STOP и непрошедшие probe/survey сохраняют `scientific_stop` и
возвращают код 2; обычная ошибка модели/процесса/I/O не становится научным
отказом. Старые writer-only `/v1` формы без живых читателей удалены, а не
объявлены совместимыми с новым JSON.
Узкий `read_legacy_ddw_outcome()` этого же владельца читает именно четыре
сохраняемых `/v2` формы для явной приёмки: полную соответствующую ось либо
допустимый canary-v2 STOP и прежние headroom verdict. Он не проверяет заново
массивы, пути, наличие файлов или всю телеметрию и не встроен в workflow.

`assignment.py` сохраняет прежнее BLAKE2s rank/round-robin назначение по
полному keyset, затем `waymo_ddw_fit/v1` выбирает explicit indices. Каждый
item последовательно получает свой FRONT/MoGe input и логи, local/headroom,
reference и прежний fit gate. Только accepted повторно рендерится,
сравнивается с измерением и кодируется в uint8 RGB/packed-known. Данные
прошлого item освобождаются до подготовки следующего.

`fit_record.py` пишет clip/v2: payload относительно папки item record.
`waymo_ddw_fit_collection/v1` читает только явно перечисленные clip/v2,
сводит по полной assignment axis, сохраняет accepted/rejected и missing
indices. Каждый payload связан с исходным item_record; коллекция его
не открывает, не копирует и не переинтерпретирует относительно своей папки.

`audit.py` допускает только rejected с полным evidence и passed headroom
через отдельный AUDIT_MASK_GATE. `waymo_ddw_audit/v1` применяет его до
MoGe, повторяет render/сравнение и пишет condition, preview и visual-audit/v2
в отдельную attempt; accepted set неизменен. `legacy_fit_record.py` читает
только прежний rejected clip/v1 для этого аудита; старые campaign-relative
accepted payload не объявляются новыми. Python-пакет `finetuning`, старые
fit/queue/audit операторы и заменённые тесты удалены. Закрытая assignment
сохранена только в ignored `selections/local/stage11/legacy/`.

Общая внешняя форма читается один раз через `config.job`; выборки и
научные поля — их предметными владельцами. Повторные filesystem/type gates
старой конфигурации не перенесены. `novel-view` и `python -m novel_view`
до завершения перехода показывают только справку/версию и направляют к
`./distil3d`; старые `stages` и `check-config` не исполняются.

## Операционные сценарии

Прежние `scripts/r4c_ddw_pilot/` и последний EUVS shell-forwarder удалены.
Папки `scripts/` больше нет: пользователь запускает корневой `./distil3d`
с явным job. `jobs/legacy/` содержит 12 EUVS и 6 Scene9 форм, которые
переиспользуют готовые recipes и существующие версии workflow. Файловый
обмен с 3dgs описан в корневом `contracts/3dgs-diffusion/`; производитель
3dgs, единственный fixture и write-only `prepare.json` не изменены.

`workflows.euvs_source_views` — первый научный route общего runner:
`euvs_source_views/v1` выбирает `core` и `inference_cp1`, дедуплицирует exact
ordered source tuples и последовательно вызывает один source-neutral VGGT
seam на launcher-assigned GPU. Model-owned record пишется напрямую в
`attempt/source-views/<first-source-token>/`; внутренних GPU scheduler/locks,
directory scans, reuse и `check-only` нет. Исторические source-views jobs
находятся в `jobs/legacy/euvs/<scope>/source-views/`.

`workflows.euvs_generation` реализует явные `euvs_generation/v1–v2`. Они
читают одну ordered selection и exact `source_views_attempt`, открывают одну resident
Gen3C session для launcher-assigned CP1 либо CP2 и исполняет пары буквально в
selection order. `v1` использует авторегрессионный seed, `v2` — source reseed
для каждого окна; остальной план и session loop общие. Результат каждой пары находится в
`attempt/pairs/<pair-name>/{generated_rgb.npy,run.json}`. Старые single и
campaign Python owners удалены вместе с внутренними GPU-группами, очередью,
lock-файлами, scan/reuse/resume и `check-only`.

Общий `generation/gen3c/recipe.py` один раз разбирает одинаковую часть
generation-настроек EUVS/Gaussian: checkpoint/model identity, seed, steps и
необязательную LoRA. Он зависит только от стандартной библиотеки и возвращает
один `Gen3cGenerationRecipe`, который внешние specs содержат композиционно.
`model_request.py` получает его напрямую и разрешает paths относительно
`models_root`; тот же путь использует явный job doctor. Внешние YAML и records
не меняются, выбор научной window policy остаётся в workflow.

`workflows.gaussian_generation` реализует `gaussian_generation/v1–v5`:
ordinary full sequence, exact selection независимых native clips, dense
independent, dense overlap21 и pure-CPU PNG handoff. `v1–v3` используют
literal `core`, `v4` — literal `moge`, `v5` — literal `core`/`cpu_test`;
model topology остаётся launcher-assigned CP1/CP2. `v2` выполняет
clips в selection order, `v3` начинает каждый dense chunk с fresh source, а
`v4` дополнительно передаёт previous trailing RGB/depth/valid ровно
следующему чанку. Одна resident session переиспользует Gen3C, а в v4 также
MoGe. Научные планы принадлежат `generation.gaussian.{full,clips,dense}`,
records v1–v3 — `generation.gaussian.record`, exact record-to-PNG join —
`generation.gaussian.handoff`. Registry или скрытого режима нет.

`workflows.waymo_depth_comparison` реализует один исторический
`waymo_depth_comparison/v1`: читает один explicit 121-frame clip, строит
общую трёхкамерную LiDAR-разметку, последовательно запускает MoGe и VGGT и
сохраняет candidate v2 record. `workflows.waymo_depth_selection` реализует
две отдельные версии решения над восемью явно перечисленными candidate
records. V1 сохраняет прежний heldout-count gate, v2 — исправленную
backend-neutral интерпретацию. Эти workflow нужны для воспроизводимости
старого выбора глубины и не входят в поддерживаемую R4c подготовку.

## Индекс EUVS

`novel_view.inputs.euvs.index` читает `frames.csv` из `dataset_root` и разрешает
упорядоченный список `image_token` в роль-независимые `FrameRef`. Результат
сохраняет порядок запроса и содержит готовые абсолютные пути к выбранным
JPEG и nuPlan DB.

Столбец `location` хранит физический alias, использованный в пути
материализованного кадра. Обязательный `logical_locations` содержит
разделённые `|` локации EUVS, которым семантически принадлежит кадр. Обычно
это одно и то же значение; общие source-проезды locations 32 и 33 имеют
`32|33`, оставаясь единственной строкой с глобально уникальным
`image_token`.

Индекс проверяет принадлежность ожидаемой логической location, а также
точные traversal и channel, но не знает про source/target,
YAML-конфигурацию и геометрию камер. Дополнительные столбцы CSV разрешены.
Индекс не выполняет filesystem preflight, не открывает JPEG и не выполняет
запросы к SQLite: он доверяет подготовленной внешней раскладке и возвращает
пути выбранных строк.

`novel_view.inputs.nuplan.db` пакетно читает сырые `ego_pose` и калибровки камер
выбранных кадров, открывая каждую nuPlan SQLite один раз за операцию и
только для чтения. Калибровка сохраняет исходные camera-to-ego вектор
переноса, кватернион в порядке `wxyz`, матрицу внутренних параметров,
коэффициенты дисторсии, модель и размер изображения. Этот слой сохраняет
исходный порядок кадров, но не строит матрицы и не выполняет временное
выравнивание.

`novel_view.inputs.nuplan.camera` соединяет один `RawEgoPose` с одной
`RawCameraCalibration`, проверяет dataset-смысл EPSG и строит
роль-независимую физическую позу камеры через чистую
`novel_view.geometry.camera`.
Соглашение `T_A_B` означает преобразование из системы `B` в систему `A` и
действует на вектор-столбец:

```text
T_global_camera = T_global_ego @ T_ego_camera
T_camera_global = inverse_rigid(T_global_camera)
```

Канонический контракт хранит неизменяемую `float64`-матрицу
`global_to_camera`, EPSG и предоставляет производные центр и оси камеры в
глобальной системе. Оси камеры следуют OpenCV: `+x` направлена вправо,
`+y` вниз, `+z` вперёд.

Wrapper проверяет единый EPSG и передаёт упорядоченные матрицы чистой
relative-W2C функции. Она вычитает глобальные центры до вращения, поэтому не
отменяет друг из друга однородные переносы UTM-масштаба. Результат сохраняет
порядок и возвращается wrapper-ом как неизменяемые `float64` OpenCV W2C.
Чистая математика не знает nuPlan/EPSG, а wrapper не знает source/target,
intrinsics после обработки растра и alignment.

`novel_view.inputs.euvs.pair` собирает одну полную физическую пару в памяти.
Reader принимает явные name, location, channel, source/target traversal и
ordered tokens, не импортируя legacy config. Для
каждого кадра он позиционно соединяет `FrameRef`, `RawEgoPose`,
`RawCameraCalibration` и `GlobalCameraGeometry`. Source и target разрешаются
отдельно, а метаданные читаются существующими пакетными читателями для их
объединённой последовательности. Порядок явной selection сохраняется
без сортировки и разворота.

Каждая последовательность сохраняет traversal и устойчивый ordered
`sequence_id`, а пара — name/location/channel и обе последовательности.
Временные метки изображений и поз строго возрастают отдельно внутри source
и target. Они не обязаны совпадать друг с другом, а времена разных
асинхронных проездов не сравниваются. Все камеры одной пары должны находиться
в единой системе EPSG. Этот слой не читает RGB, не выполняет alignment и не
применяет ограничения временной оси Gen3C.

`novel_view.geometry.polyline` содержит роль-независимую планарную
математику. Функция
проецирует произвольные точки на ближайшие зажатые сегменты ориентированной
полилинии и возвращает координаты проекций, индексы и доли сегментов,
накопленное расстояние вдоль source, обычное расстояние и знаковое поперечное
отклонение. Положительный знак означает левую сторону относительно
направления выбранного сегмента.

Отдельная роль-независимая функция выбирает ближайшую вершину только среди
двух концов уже найденного сегмента; точный midpoint выбирает меньший индекс.
Она возвращает изменяемый owned-массив, чтобы предметный слой мог явно
восстановить свои exact choices до публикации неизменяемого результата.
Внутренние массивы считаются типизированными и корректными; модуль не
повторяет матрицу shape/dtype/finiteness-проверок. Явная ошибка остаётся
только для математически неопределённой полилинии с недостатком вершин или
нулевым сегментом.

Модуль работает в единицах входных координат и не знает про EUVS, роли
source/target, EPSG, время и камеры. Он не ищет новую ветвь траектории, не
проверяет монотонность, неоднозначность или допустимые расстояния и не
исправляет самопересечения. Эти решения принадлежат предметному alignment.

`novel_view.generation.euvs.plan` связывает физическую EUVS-пару с чистой
планарной математикой. Он в исходном порядке проецирует глобальные `XY`
центров target-камер на полилинию центров source-камер и сохраняет знаковый
продольный выход target до начала либо после конца source. Затем для каждого
target выбирается ближайший конец уже найденного source-сегмента; точный
midpoint выбирает меньшую позицию. Это не глобальный повторный поиск.

Знаковый остаток считается относительно незажатого
`target_source_progress`, который уже включает `source_domain_excess`.
Поэтому выход до начала или после конца source не теряется. Many-to-one,
пропуски source и регрессии не исправляются и не отклоняются. Назначение
крайнего кадра также не означает автоматического принятия endpoint.

Тот же файл один раз связывает ordered `sequence_id` source RGB и
`PosedDepthSequence`, строит measured target `K/W2C` на общей сетке и
компонует generic Gen3C timeline и conditioning. Он не читает target RGB,
не запускает модель, не задаёт пороги качества и не знает job/runtime/GPU
или сохраняемых records.

`novel_view.geometry.raster` содержит роль-независимую линейную геометрию
растра. `CoverCropTransform` вычисляет минимальный целочисленный
resize-to-cover, детерминированный центральный crop и однородную матрицу,
которая переносит пиксельные координаты и уже выпрямленную pinhole-матрицу
`K` на выходную сетку.

Контракт использует непрерывные координаты границ изображения: растр
`H × W` занимает прямоугольник `[0, W] × [0, H]`. Места выборок конкретного
алгоритма интерполяции в этот слой не входят. Целочисленное округление может
сделать фактические горизонтальный и вертикальный масштабы немного разными;
`K` преобразуется именно фактическими масштабами.

Модуль не знает про EUVS, source/target, JPEG, дисторсию, OpenCV, модельный
модуль геометрии или Gen3C. Размеры и pinhole `K` приходят от
типизированного внутреннего владельца и не проходят повторную validation-
матрицу; вычисленные малые массивы возвращаются read-only.

`novel_view.inputs.nuplan.raster` соединяет эту линейную геометрию с native
nuPlan calibration. Первая явно названная политика использует native `K`
как матрицу выпрямленной камеры и вычисляет:

```text
K_output = CoverCropTransform.pixel_transform @ K_native
```

Лёгкий `build_nuplan_output_intrinsics` публикует только этот неизменяемый
`K_output`. Полный `build_nuplan_raster_plan` использует тот же приватный
валидированный расчёт и затем добавляет distortion maps и маску. Поэтому
camera-only потребитель не строит тяжёлые OpenCV-карты, но не получает
вторую реализацию пиксельной математики.

`cv2.initUndistortRectifyMap` получает сразу `K_output` и конечный размер.
Поэтому план описывает один обратный `remap` из distorted native-растра в
конечную pinhole-сетку без промежуточного native-size undistort, отдельного
resize и скрытой полупиксельной поправки.

`NuPlanRasterPlan` содержит только размеры, итоговый `float64 K`, две
`float32` source-coordinate карты и boolean `valid_mask`. Маска означает,
что `INTER_LINEAR`-выборка единичного исходного растра не получила вклад
нулевого `BORDER_CONSTANT`; она не подменяется прямоугольным
`validPixROI` или nearest-neighbour маской.

Адаптер требует строго нулевой skew исходной calibration, потому что OpenCV
не использует этот компонент source `K`. Hardware label, camera pose,
source/target, JPEG, RGB-порядок каналов, модельный модуль геометрии, Gen3C
и файловые результаты в raster plan не входят.

Тот же `novel_view.inputs.nuplan.raster` исполняет готовый план для одного
закодированного изображения (в nuPlan — JPEG). Оно декодируется в
сохранённой ориентации как `uint8 BGR`; после проверки точного
native-размера выполняется один `cv2.remap` с `INTER_LINEAR` и нулевым
`BORDER_CONSTANT`, а затем каналы переставляются в RGB. Результат содержит
неизменяемый C-contiguous массив и ссылку на тот же `NuPlanRasterPlan`,
поэтому итоговые пиксели хранятся вместе с соответствующими `K` и
`valid_mask`.

Этот слой не знает про source/target, порядок EUVS, `FrameRef`, кеширование,
конкретный модельный модуль или Gen3C и не нормализует пиксели в `float32`.
Следующий адаптер последовательности не должен копировать большие карты в
каждый кадр. Более широкий uniform-FOV также остаётся возможной последующей
политикой, а не параметром текущего YAML. SQLite, camera composition и
raster I/O имеют одного concrete input-владельца, а чистая pixel math
остаётся в `novel_view.geometry.raster`.

`novel_view.inputs.euvs.rgb` применяет однокадровый исполнитель к полной
`EuvsFrameSequence`. Результат сохраняет точный входной объект и
позиционную связь каждого `EuvsFrameInput` с `RasterizedRgb`; кадры не
сортируются, не разворачиваются, не фильтруются и не дедуплицируются.
Кадровый результат сохраняет `image_token`, а последовательность — тот же
ordered `sequence_id` и traversal, что и физический input.
Пакетный результат является временным объектом внутри одного Python-процесса
для коротких текущих проездов, а не файловым или межпроцессным протоколом
между `novel-core` и модельными окружениями. Однокадровый исполнитель
остаётся границей для будущей последовательной или блочной обработки.

В рамках одного вызова планы кешируются по физическому ключу
`(db_path, camera_token)`. Повторный ключ обязан иметь точно ту же сырую
калибровку; несколько камер внутри последовательности поддерживаются.
Локальный словарь кеша заканчивает существование после операции, но каждый
построенный plan продолжает жить в результатах кадров, которые совместно
ссылаются на него. Глобального реестра и скрытого межпарного удержания карт
нет.

Общая операция роль-независима и позднее может применяться к target для
метрик. Отдельная source-обёртка передаёт ей только `pair.source`, поэтому
этап `source_rgb` не открывает target JPEG. Этот слой не зависит от
alignment, не выбирает conditioning-кадры, не нормализует RGB под модель и
не создаёт файловых результатов.

`novel_view.geometry.depth` задаёт узкую dataset-neutral границу после
модельного адаптера. `PosedDepthSequence` хранит ordered `sequence_id`,
метрическую camera-Z depth, явную маску геометрической валидности, pinhole
`K`, OpenCV W2C и необязательную confidence для каждого кадра. Индекс во
всех массивах означает ту же позицию в `sequence_id`; RGB-объект внутрь
контракта не вкладывается.

Контейнер намеренно лёгкий: он не копирует тяжёлые массивы и не повторяет
матрицу проверок форм, dtype и физических инвариантов. Конкретные DA3/VGGT
адаптеры отвечают за source-only принадлежность, порядок, совместное
происхождение значений и публикацию owned/read-only массивов. Если rotation
пришла из `float32`-модели, адаптер явно проецирует её на SO(3), измеряет
величину исправления и только затем строит результат; контейнер не исправляет
камеру скрыто.

Все плотные поля находятся на общей сетке source RGB. Модельный адаптер
обязан явно перенести выбранную геометрию с собственной сетки на эту сетку;
его исходная depth, предсказанные камеры, преобразование пикселей и
диагностики могут сохраняться в отдельном полном результате. При соблюдении
этой границы общий последующий код не получает смесь depth одной камеры с
`K/W2C` другой.

`reference_to_camera[i]` переводит точки из системы первой физической
source-камеры в систему камеры, которой принадлежат `depth_z_m[i]` и
`intrinsics[i]`. Матрицы используют OpenCV-оси и метры. Первая матрица не
обязана быть единичной: после общего Sim(3)-выравнивания предсказанная
камера может иметь измеренный остаток относительно nuPlan. Позднее
seed-local преобразование обязано совместно перенести source и target, а не
независимо занулять первую source-позу.

`geometry_valid_mask` не смешивается с растровой валидностью: адаптер
формирует её явно относительно выбранного source RGB. В текущих адаптерах
невалидная каноническая depth равна нулю, а валидная конечна и строго
положительна.
Полностью невалидный кадр сохраняется как явное измерение, а решение
отклонять последовательность или не использовать этот кадр относится к
последующей политике качества.
Необязательная `backend_confidence` остаётся непрозрачным конечным числом
без общего направления, порога или вероятностного смысла; вне
геометрической маски она канонически равна нулю. `K` имеет стандартную
нулевую skew, которую поддерживают текущие OpenCV- и Gen3C-пути.

Этот модуль не запускает модель, не выбирает backend, не читает target RGB,
не выполняет Sim(3), LiDAR-уточнение или Gen3C conditioning и не задаёт
файловый либо межпроцессный формат. Любой новый backend становится
сменяемым через построение того же цельного результата, а не через общий
ABC, реестр или набор независимо подменяемых полей.

`novel_view.generation.euvs.plan.EuvsGenerationInput` хранит
`EuvsPairInput`, отдельный source `EuvsRasterizedSequence` и
`PosedDepthSequence`. На этой единственной границе один раз сопоставляются
три ordered `sequence_id` и общая `image_size_hw`; совпадение экземпляров
Python не требуется.

Для каждого `pair.target.frames[j]` контракт хранит measured nuPlan `K` на
фактической выходной размерности source geometry и measured OpenCV W2C из
системы первой физической source-камеры в target camera `j`. Target W2C не
зависит от первой predicted PDS-камеры, которая после общего Sim(3) может
быть не единичной. Поздний seed-local слой обязан одним преобразованием
перенести source и target вместе.

Контракт не читает target RGB, не хранит remap-карты и не принимает пару по
качеству. `EuvsGenerationPlan` рядом хранит exact mapping, timeline,
conditioning и лёгкий Gen3C request для одной пары.

`novel_view.generation.gen3c.timeline` является первой модельно-специфичной
границей после нейтрального входа. Прямой baseline резервирует global slot 0
для source condition, помещает первый exact target в slot 1 и квантует
остальные относительные target image timestamps на закреплённую сетку 24 fps
точной целочисленной арифметикой. Source и target absolute timestamps не
сравниваются; порядок пары не сортируется, не разворачивается и не чинится.

`Gen3cTimeline` не импортирует EUVS: он хранит только forward-карту
target→output slot и target-ordered source indices, переданные предметной
композицией. Полная длина и полуоткрытые окна выводятся из последнего
target-якоря по контракту pinned Gen3C:

```text
window size = 121
overlap = 1
step = 120
model frames = 120*k + 1, k >= 1
```

Окна описывают один общий persistent-запуск и не являются независимыми
вызовами модели. Timeline не разворачивает mapping в плотный per-slot source
массив, не выбирает conditioning для промежуточных и padding-слотов, не
строит query `K/W2C`, не выполняет seed-local rebase и не знает про RGB,
depth, masks, PyTorch, checkpoint или Cache4D. Эти решения остаются
последующими отдельными границами; совместимый дообученный checkpoint обязан
получать ту же временную ось, что и базовая модель.

`novel_view.generation.gen3c.windows` является единственным владельцем
исполняемых решений поверх этой оси. Frozen `Gen3cWindowCall` хранит start,
stop, реальный либо предыдущий seed, physical source index, согласованную
замену первой W2C/K-строки для `source-reseed/v1` и local slice, который
убирает повторный seam. Модуль не импортирует Cache4D, Torch, pipeline,
Gaussian или workflow; старый составной worker только исполняет готовые
решения в прежнем порядке.

`novel_view.generation.gen3c.conditioning` добавляет две последовательные
лёгкие границы. `Gen3cDirectQueryTrajectory` выбирает source anchor из
первого exact target mapping и выполняет один общий rebase всех source и
measured target W2C. Он хранит только `N+M` камер, а не полный временной ряд.

`Gen3cProjectedSingleSourcePlan` фиксирует явно названный baseline выбора
одного source-кадра. При материализации канонического окна query-центры
интерполируются общей `novel_view.geometry.trajectory`: центры линейно,
C2W-вращения — по кратчайшей геодезической SO(3), а `fx/fy/cx/cy` — линейно
между exact target anchors. После последнего target камера и `K`
удерживаются. Ровно противоположные вращения отклоняются как неоднозначные,
а exact anchors копируются без интерполяции.

Для промежуточного слота query-центр возвращается из anchor-системы в
систему первой физической source-камеры и затем в measured global `XY` без
добавления UTM-origin. После существующей проекции на source-полилинию
выбирается ближайший endpoint уже выбранного сегмента. Slot 0, exact target
slots и padding явно восстанавливают anchor, исходный target mapping и
последний source соответственно. Никакой монотонности, repair, порогов или
скрытого разворота нет.

Результат содержит только 121 query W2C, 121 `K` и 121 source index для
одного глобального окна Cache4D, без скрытой identity-метки. EUVS-adapter
использует этот индекс одновременно для source RGB и соответствующей строки
depth/K/W2C; их ordered identity заранее связана предметным планом.

Холодный `novel_view.models` содержит только уже используемые конкретные
model capabilities и ничего не реэкспортирует. Первая такая граница —
`models.gen3c.cache4d.request`: `Gen3cSourceRows` является только статическим
интерфейсом `len/read`, а `Gen3cConditioning` хранит source rows, K/W2C,
глобальные индексы, diagnostic slots и optional второй context layer. Здесь
нет dataset-типов, generation order, runtime-validator или Torch.

`novel_view.models.gen3c.cache4d.protocol` является единственным владельцем
базовых имён NPY. Он создаёт три memmap и читает EUVS, Gaussian либо Waymo
source строго по одной строке; dense переиспользует Gaussian rows, а его
overlap context реализует тот же узкий интерфейс. Camera/index arrays
сохраняются один раз; тот же reader используют составной Gen3C worker и
standalone Cache4D worker. Научные объекты больше не получают каталог и не
содержат `write_exchange`.

`novel_view.generation.gen3c.request` оставляет прежние raster/timeline
метаданные только для текущего parent API и оборачивает общий conditioning
одним optional набором context-result slots. Соседний `protocol` является
единственным владельцем generation markers, output/context-result и
телеметрических NPY-имён и их записи. `euvs_input`
строит это описание поверх существующего projected-nearest плана без
изменения его численной политики. Backend не знает, из EUVS или другого
источника пришли согласованные RGB/depth/mask/K/W2C и глобальные раскладки.

`novel_view.models.gen3c.cache4d` является первой тяжёлой модельной
границей. `runtime.py` собирает source evidence, блочно вызывает официальный
`unproject_points`, создаёт один Cache4D и предоставляет один raw render.
Составной Gen3C worker импортирует runtime напрямую, поэтому cache и warp не
покидают его процесс. Для standalone Waymo reference соседние
`backend.py`/`_worker.py` используют тот же runtime через временный NPY.

Worker выбирает одну source-строку для каждого global slot, исключает из
support валидную depth свыше официального предела 100 м и блочно вызывает
официальный `unproject_points`. После этого создаётся один общий Cache4D на
всю последовательность; окна `[120*i, 120*i+121)` рендерятся с правильным
`start_frame_idx`. Baseline фиксирует официальный dynamic-порог `0.05` и
отключённый foreground masking. Полные render-тензоры не покидают worker:
наружу возвращаются coverage, far-depth доля, overlap-ошибка и несколько
кадров. Multi-source и bridge остаются следующими слоями; diffusion вынесен
в отдельную границу ниже.
Диагностический результат намеренно не является conditioning-протоколом.

`novel_view.models.gen3c.session.Gen3cModelSession` является единственным
владельцем живущего Gen3C model-state. Лёгкие `spec.py` и `request.py`
фиксируют model facts, checkpoint/LoRA, sampling и CP topology; session
локально импортирует official stack, строит один pipeline, загружает
base/full/LoRA веса, нормализует seed и выполняет один window call. После
construction он сохраняет RNG и перед каждым новым составным запросом
восстанавливает эту границу ровно один раз. Dataset, Cache4D, window policy,
stitching, markers и публикация результата остаются generation-side.

`novel_view.generation.gen3c.session.Gen3cGenerationSession` является
parent-границей составной генерации. Она получает статически типизированный
`Gen3cConditioningInput`, именованный model artifact, sampling и process
resources, затем через `runtime.distributed` поднимает один закреплённый
torchrun process. Model pipeline остаётся resident между последовательными
запросами; `generation.gen3c.execute.generate_gen3c_sequence` является
одноразовой обёрткой над тем же контрактом.
Base и полный fine-tuned checkpoint отличаются только
`Gen3cModelSpec`; архитектурное имя остаётся официальным
`Gen3C-Cosmos-7B`. LoRA выбирается явно: controller хранит provenance для
run record, а worker получает только model-ready checkpoint и strength;
training topology в модельном процессе не проверяется.

Diffusion и Cache4D живут в одном worker: полный cache и warp не становятся
IPC или постоянным форматом. Cache4D, входные memory map и output принадлежат
одному запросу и освобождаются до его terminal marker; сама модель остаётся
resident до закрытия сессии. Политика `autoregressive/v1` начинает первое
окно настоящим source slot 0, а последующие — последним сгенерированным
кадром предыдущего окна. Экспериментальная `source-reseed/v1` оставляет
первое окно неизменным, а каждое следующее начинает настоящими RGB, W2C и K
одной source-строки, уже выбранной для глобального начала окна. Cache4D,
глобальные query slots, hold-last tail и последовательное RNG остаются
общими, чтобы режим менял только источник оконного seed. Rank 0
записывает первый блок целиком и `video[1:]` каждого следующего, поэтому
lossless `uint8 [T,H,W,3]` содержит каждую глобальную строку ровно один раз.
Rank 0 пишет файл непосредственно в caller-owned путь. Transient
request-каталог содержит только conditioning NPY, markers, diagnostics и
телеметрию; staging output, hard link, no-overwrite и повторная проверка
worker output удалены.

Только rank 0 наблюдает файловые команды сессии. В CP2 он синхронно рассылает
малую числовую пару `(opcode, request_index)` через уже существующую
NCCL-группу; предметные массивы по-прежнему передаются NPY, а JSON, pickle и
отдельный сервис не добавляются. Один запрос выполняется за раз,
конкурентный вызов отклоняется до изменения request state, а `close`
дожидается активного запроса. Любая ошибка rank или ранний выход процесса
делает сессию broken, после чего
родитель завершает всю OS process group. Перед каждым запросом worker
восстанавливает одно post-construction RNG-состояние, не сбрасывая RNG между
окнами запроса. Поэтому одинаковый вход в одной сессии воспроизводим и
сопоставим с одноразовой обёрткой.

Rank 0 публикует terminal marker только после flush и закрытия output,
закрытия всех input memory map, освобождения request-local Cache4D,
сохранения телеметрии и barrier всех rank. Телеметрия хранит current
allocated/reserved до запроса, request peak и current после очистки отдельно:
один peak не используется как доказательство отсутствия накопления памяти.
Для каждого результата совместимые `per_rank_peak_cuda_*` остаются полным
пиком одноразового пути — максимумом startup и именно текущего request —
поэтому их смысл для существующих run-record не меняется. Это не
накопительный максимум всех предыдущих запросов resident-процесса.

Численное выравнивание от одного до двадцати overlap-context кадров находится в
`novel_view.generation.gen3c.context_depth`. Оно принимает только готовые raw
MoGe depth/valid, Cache4D reference depth/valid и query W2C/K, выполняет
прежние metric fit, local filter и двунаправленное соседнее warp-согласование.
Выбор slots остаётся у dense overlap science, raw MoGe и Cache4D — у своих
model/process владельцев, а запись NPY пока остаётся в составном worker.

Для одного запуска поддерживаются CP1 и CP2. `torchrun`, видимые GPU,
`WORLD_SIZE` и `LOCAL_WORLD_SIZE` обязаны описывать одну и ту же явную
топологию. CP3 отклоняется до весов, поскольку temporal latent 16 не делится
на три. Context parallel разделяет активации, но каждый rank загружает полный
checkpoint и строит свой Cache4D. Массовое распределение независимых пар,
MP4, resume и метрики этому backend не принадлежат.

`novel_view.inputs.gaussian.reader` владеет одним внешним
`gaussian_depth_export/v1`: metadata читается в producer order, а payload
открывается ровно по одной запрошенной RGB/depth/mask строке. Относительные
пути считаются от `info_file.parent`; полного filesystem preflight нет.

Ordinary full sequence уже находится в `generation.gaussian.full`. Он
выбирает ближайший native кадр на каждый непрерывный слот 24 fps, одной
`CoverCropTransform` приводит RGB, camera-Z depth, mask и `K` к `704×1280`,
а source и target W2C одним rigid rebase выражает относительно первой source-
камеры. Target slots идут `1..N`; source/query tail удерживает последнюю
строку до целого окна Gen3C. Модуль строит только построчный источник и
camera/index conditioning без job, runtime, NPY-имён или тяжёлого
model runtime.

`generation.gaussian.record` является конкретным владельцем неизменённых
Gaussian records v1–v3. Loader строго различает full/clip, dense-independent
и dense-overlap21 schemas, а writer пишет выбранный `run.json` напрямую без
staging/atomic/no-overwrite. Canonical full, clips и dense-independent уже
пишут его напрямую; dense-overlap также напрямую создаёт record v3.
Builders создают прежние dataclass без повторного внешнего reader; три
маленькие функции собирают общие generation/output/dense-input поля.

`workflows.gaussian_generation.run_v1()` связывает exact producer JSON и
`full_sequence` selection, создаёт один concrete base/full model request,
открывает одну launcher-assigned CP1/CP2 `Gen3cGenerationSession` и прямо
пишет `generated_rgb.npy`, target-only `targets.mp4` и record v1 в attempt
root. Каталоговых scan, GPU scheduler, reuse/resume, `check-only`, output
validator и публикационной обвязки нет. Historical record v1 не умеет
сохранять LoRA identity, поэтому workflow не принимает LoRA скрыто.

`generation.gaussian.clips` строит полные 120-target окна, выравнивает
последнее по правому краю и возвращает только 1-based IDs из selection в её
порядке. `workflows.gaussian_generation.run_v2()` записывает каждый ID в
`clips/clip-NNN/` (имя каталога 0-based), заново строит source-local
conditioning и сохраняет исторический v1 record с полным clip-интервалом.
Одна model session переиспользуется только как дорогой runtime; generated
history между clips не передаётся. Старые clip flags, all-clips campaign,
preflight output checks и clip owner удалены.

`generation.gaussian.dense` теперь владеет dense camera table,
shortest-SO(3) trajectory, independent chunks и их Gen3C conditioning. Он
получает exact external transforms/intrinsics, source pose range из selection
и factor/capacity из recipe; конкретные числа сцены в production-коде
не находятся. Row index является stable dense camera ID, W2C уже включает
virtual-camera shift, а последний чанк держит последнюю target camera только
в padding slots. Каждый чанк читает только собственный набор физических
source rows и делает отдельный anchor-local rebase.
Неиспользуемые `bake_camera_ids` и `production_source_pose_indices` удалены;
builder и `write_standard_camera_table` вызываются напрямую, без второго writer.
Поле `dense.bake_stride` в recipe v3/v4 сохраняется для совместимости схемы,
но PNG-выборка задаётся отдельными selection и `bake_dense_stride` workflow v5.

`workflows.gaussian_generation.run_v3()` строит таблицу до модельной сессии,
пишет её в `dense-camera-table/`, затем через одну resident session создаёт
`chunks/chunk-NNN/{generated_rgb.npy,targets.mp4,run.json}`. Record v2
сохраняет exact dense IDs, owned output slots, physical source poses и LoRA
identity. В новом пути нет старых queue/GPU groups, scan/reuse, `max-new`,
`check-only`, проверки существующей таблицы или отдельного CLI.

`workflows.gaussian_generation.run_v4()` использует ту же exact camera table,
но планирует 100 owned и до 20 trailing targets. Каждый запрос сохраняет
fresh physical source; generated RGB, выровненные metric depth и valid mask
trailing slots становятся единственным previous context следующего чанка.
Один resident process удерживает Gen3C и MoGe, а direct record v3 сохраняет
owned/context slots, LoRA identity, context diagnostics и seam MSE/PSNR. В
новом пути нет prefix resume, scan/reuse, `max-new`, `check-existing`,
`check-only` или отдельного CLI; image variant — literal `moge`.

`generation.gaussian.handoff` получает две exact runs-relative attempts,
строит `chunk-000..N/run.json` из сохранённого `chunk.count`, связывает
ordered dense IDs только с production slots и сверяет, что одинаковые ID в
двух методах относятся к одной physical camera trajectory. Pure-CPU
`workflows.gaussian_generation.run_v5()` пишет общую
byte-exact `dense-camera-table/`, две `method/selected/` выдачи native PNG и прямые
`prepare.json` joins. Старые glob/scan, `check-only`, runtime README,
staging/atomic и весь `gaussian_scene_gen3c.dense_chunks` удалены.

`novel_view.generation.euvs.record` задаёт отдельный лёгкий предметный контракт
постоянной identity. Он сохраняет ordered pair tokens, exact target slots и
target-level source indices, именованную conditioning policy, runs-relative
geometry, models-relative checkpoint, sampling, фактический CP1/CP2 и
`uint8 RGB` descriptor. Полный camera plan, K/W2C, GPU ids, runtime
telemetry, абсолютные пути,
SHA и artifact registry не дублируются. Модуль не импортирует workflow и не
знает про CUDA, конкретный conditioning-plan или model runner; workflow
проецирует resolved-контракты в стабильные поля записи. Строгий JSON-loader
читает v1–v5, причём v1 разрешается только явным legacy opt-in. Исторический
v4 сохраняет LoRA identity без реально применённой strength; новые
LoRA-запуски получают v5 со strength, не меняя старую схему. Writer пишет
JSON прямо в выбранный caller-ом путь и не проверяет RGB, filesystem или
resume-совместимость.
Reader сохраняет точные поля версии, legacy opt-in и обязательность LoRA
identity/strength; повторный разбор значений перед неизменными `__post_init__`
удалён. Типы tuple/float и предметные ограничения сохраняются у этих классов.

`novel_view.evaluation.euvs.samples` открывает ровно переданный generation
`run.json`, берёт sibling output из `record.output.file`, выбирает записанные
`target_output_index` и растрирует только target-последовательность. Результат
сохраняет точный target/output/source order в собственных малых
`EuvsSample/EuvsSamples`; через `EuvsRasterizedFrame` доступны token,
настоящий RGB, raster-valid mask, K и измеренная камера без дублирования.

`novel_view.evaluation.euvs.masks` владеет EUVS native-grid mask contract и
прямым cache `/cache/euvs/masks/<recipe-id>/native/<token>.png`. `True` и PNG
`255` означают потенциально движущийся объект. Последовательность сохраняет
ordered token identity; повторный отсутствующий token передаётся модели один
раз и возвращается во всех исходных позициях. Только `FileNotFoundError`
считается cache miss. На внешней PNG-границе проверяются native size и
polarity `0/255`; sidecar, inventory, SHA, scan и filesystem preflight
отсутствуют. Nearest-neighbour переносит mask на существующий
`NuPlanRasterPlan`, не смешивая её с `plan.valid_mask`.

`novel_view.models.grounded_sam2` владеет raw baseline: Grounding DINO tiny
создаёт boxes, SAM2.1 Hiera Large независимо сегментирует кадр, а instance
masks объединяются OR. Один составной worker сохраняет offline load,
BF16/TF32, native target size и exact thresholds/options; fixed Gen3C Python
запускается через no-timeout process seam. `evaluation.euvs.masks` передаёт
ordered native RGB, model-параметры и постоянный log path, а само сохраняет EUVS
recipe/token order, cache hit/miss, duplicate collapse и polarity.
Prediction RGB и target/source-роль raw model adapter-у недоступны.

`novel_view.evaluation.euvs.support_projection` задаёт чистый предметный
протокол
`four-neighbour-zbuffer/v1`. Для одной строки source metric camera-Z depth
переводит integer source pixel centres через OpenCV `K/W2C` на continuous
target grid. Точка допускается только при finite positive target camera-Z и
центре внутри target raster. Она вносится в уникальные `floor/ceil`-соседи;
integer coordinate обновляет один пиксель. По каждому target pixel выбирается
минимальная target camera-Z, а exact-depth labels объединяются OR. Вся
valid source-поверхность участвует в z-buffer, поэтому static foreground
может закрыть dynamic background. Выходы раздельно хранят coverage и
видимую boolean source-метку; dilation, epsilon и confidence threshold
отсутствуют.

`novel_view.evaluation.euvs.support_views` является малой чистой функцией и
публикует две области:

```text
target_static_support = raster_valid & ~target_dynamic
source_aware_static_support =
    target_static_support & ~projected_source_dynamic
```

Projection coverage намеренно не является её входом, поэтому disocclusion
остаётся частью оценки. Prediction RGB не участвует. Тот же модуль связывает
записанный source index, saved VGGT PDS и измеренную target W2C с этими
примитивами; прежние общие evaluation wrappers удалены.

`novel_view.metrics.image` владеет dataset-neutral численным протоколом
`audited-static/v1`: exact uint8 PSNR, population Gaussian SSIM по полным
поддержанным окнам 11x11, pixel/patch support weights и явные состояния
finite/perfect/undefined. `novel_view.metrics.aggregate` владеет
equal-camera macro и одинаковым весом полных pair/location rows.
`novel_view.evaluation.euvs.execute` является EUVS-композицией: принимает
exact `EuvsSupport` и возвращает per-frame две параллельные дорожки
`target_static` и `source_aware_static`.
LPIPS 0.1.4 AlexNet spatial и DINOv2 ViT-B/14 изолированы в частном worker
окружения `/opt/envs/gen3c` через `runtime.process.run_process`. Точные model load и raw
forward/features принадлежат `novel_view.models.lpips` и
`novel_view.models.dinov2`, но остаются in-process частью того же worker.
Worker владеет deterministic Torch flags, raw cosine и единым временным
результатом; support aggregation вызывает dataset-neutral `metrics`, а
RGB/support NPY не становятся постоянным форматом.
`workflows.euvs_evaluation` выделяет каждой паре собственную папку логов
текущего attempt и разные `source-masks.log`, `target-masks.log`, `metrics.log`.
Полный stdout/stderr остаётся после удаления scratch и при ошибке процесса;
вычислительного deadline нет.

DINOv2 обрабатывает полный FOV после bicubic resize в 462x840 и возвращает
1980 patch tokens без CLS. Boolean support интегрируется по исходным pixel
cells в patch preimages, поэтому effective mass может быть дробной. LPIPS и
DINOv2 являются support-weighted readout, но не строгими masked-метриками:
receptive field и global attention видят исключённый контекст. Результат
сохраняет pixel/window/patch masses; `evaluation.euvs.benchmark` напрямую
вызывает `metrics.aggregate.summarize_cameras` для equal-camera сводки пары.
Только perfect PSNR имеет состояние `positive_infinity`;
пустая effective область — `undefined(empty_support)`, а отсутствие полного
SSIM-окна — `undefined(no_full_windows)`; публичные числовые поля не содержат
NaN/Inf. Вычислительный слой не создаёт файлов и не зависит от Gen3C
checkpoint.

`novel_view.evaluation.euvs.record` владеет существующим строгим
`novel-view/euvs-metric-result/v1`. Он сохраняет runs-relative generation
locator, versioned protocols, mask recipe и две ordered metric-дорожки с
effective masses. Tokens, slots, mapping, geometry, model и sampling остаются
единственным источником истины в связанном `run.json`; RGB, masks, tensors,
SHA и абсолютные пути не копируются. Reader не ищет соседние artifacts, а
writer пишет прямо в выбранный workflow путь.

`novel_view.workflows.euvs_evaluation` реализует один
`euvs_evaluation/v1`. Strict job указывает selection и либо точную generation
attempt, либо точный legacy-список records. Attempt reference детерминированно
даёт `pairs/<pair-name>/run.json`; directory discovery отсутствует. Workflow
последовательно выполняет samples → masks → support → metrics и пишет только
`attempt/metrics/<pair-name>.json`. Первая настоящая ошибка завершает attempt;
continue, reuse, resume и `check-only` отсутствуют. Старые одиночные,
campaign и benchmark Python-entrypoints удалены. Отдельный comparison над
двумя готовыми evaluation attempts реализован независимо от этого пути.

`novel_view.evaluation.euvs.benchmark` и
`novel_view.workflows.euvs_comparison` владеют `euvs_comparison/v1`.
Strict job задаёт одну selection и две runs-relative evaluation attempts;
workflow строит только точные `metrics/<pair-name>.json` и связанные через
них generation records. Он не импортирует evaluation execution, не запускает
модели и использует `core` с нулевым `cpu_test` preset. Pair сравнивается
только при одинаковых conditioning slots/mapping, geometry, sampling, CP,
output grid, metric/support/projection protocols, mask recipe и effective
support masses. Missing и incompatible остаются явными строками и не
попадают в scalar aggregate. Equal-camera pair means сводятся с равным весом
pair внутри location и равным весом location глобально; LPIPS improvement
имеет обратный знак. `frames.csv`, `pairs.csv` и `summary.csv` пишутся прямо
в текущую attempt без discovery, скрытого evaluation, staging, atomic rename
или no-overwrite.

Научный scope остаётся обычным `ExperimentConfig`, а не отдельным реестром.
Текущие конфигурации производной Gen3C-on-EUVS кампании образуют
непересекающееся разбиение 80 формальных source `1..5` → target `6`
комбинаций: это `16` логических locations × `5` source traversals текущего
материализованного nuPlan-поднабора, а не официальный denominator или PairSet
EUVS Setting 1. Исторические проектные классы A/B/C дают `primary35` и
`extended16`; `raw29` является механическим дополнением inventory, а не
научно отобранным классом. `primary35` содержит строгие полные пары основной
метрики, `extended16` — дополнительные B/C-пары с шестью явно обрезанными
target-интервалами, а `raw29` намеренно сохраняет полный target остальных
комбинаций. Benchmark не объединяет эти области сам: каждая передаётся и
публикуется отдельным запуском; `raw29` не агрегируется с двумя другими.

`novel_view.workflows.euvs_generation` является поддерживаемой композицией
ordered selection. `generation.euvs.plan` строит CPU-план каждой пары,
`generation.gen3c.model_request` задаёт одну base/full/LoRA identity, а одна
resident session выполняет все запросы attempt. Pair, model, sampling,
execution topology и output descriptor сохраняются предметным
`generation.euvs.record`; каталог и имя файла не заменяют эту identity.

Исторический `balanced-896` результат первой пары предшествует постоянному
контракту source-геометрии: его summary содержит число, крайние токены, режим
и сетки, но не полный упорядоченный список source-токенов. Поэтому сценарий
помечает его `legacy-attested`, требует явного `--allow-legacy-geometry` и не
считает универсальным форматом продолжения. Новый summary может содержать
полный `source_image_tokens`, который проверяется дословно; отдельный manifest
или SHA для этого не создаётся.

Настоящий запуск создаёт direct pair-каталоги и временный scratch под
фиксированным cache root. Model worker пишет выбранный caller-ом
`generated_rgb.npy`, после чего workflow напрямую записывает `run.json`.
Предварительного output validator, staging, atomic/no-overwrite и скрытого
completion/reuse-протокола нет. Preview, общий manifest/registry и второй
diffusion API не добавляются.

`novel_view.generation.euvs.source_views` владеет измеренным EUVS-входом.
Небольшой `SourceGeometryInput` сохраняет точный `EuvsRasterizedSequence`,
его измеренные pinhole `K` и OpenCV W2C, перебазированные из глобального UTM
в систему первой source-камеры:

Его builder для численной устойчивости переиспользует общую
операцию `nuplan.camera`, а не строит собственную относительную позу прямым
перемножением однородных матриц с UTM-переносами порядка миллионов метров:

```text
R_reference_to_camera[i] = R_global_to_camera[i] @ R_global_to_camera[0].T
t_reference_to_camera[i] =
    R_global_to_camera[i] @ (C_global[0] - C_global[i])
```

Переносы остаются метрическими, первая матрица точно единична, порядок
совпадает с source RGB. Frozen DA3 получает этот measured input; VGGT
получает только source-neutral RGB request, а его отдельная математика —
явные массивы. Повторной внутренней type-проверки camera-pack нет;
ошибка предметной camera-операции приходит от её существующего владельца.

`novel_view.runtime.executables` хранит буквальные `/opt/envs/core`,
`/opt/envs/gen3c` и `/opt/envs/vggt` Python locators без scan, fallback или
registry. Конкретный backend сам выбирает один из них.

`novel_view.runtime.process` содержит общий no-timeout запуск одного
внутреннего worker. Он использует parent guard, отдельную OS process group,
caller-chosen log и очищенное от пользовательских Python-путей и известных
Hugging Face-токенов окружение. Обычный `wait()` не имеет computation
deadline; grace применяется только при явном `terminate`.
`novel_view.runtime.distributed` строит только закреплённую
single-node torchrun-команду поверх no-timeout handle; новый Gen3C parent
использует её без ручного Python path и общего computation deadline.

Предметные массивы записывает соответствующий пакет:
Исследовательский `research.da3_nested.io` один раз материализует упорядоченный source
RGB во временный NPY. Каталог обмена удаляется после операции. Это транспорт,
а не постоянный формат артефакта или общий модельный API.
Чтение NumPy и закрытие memmap принадлежат конкретному serializer; общего
`model_process` и отдельной process-resource оболочки больше нет. EUVS и
Cache4D получают постоянный log path от своего consumer, не кладут лог в
scratch и не добавляют проверок файловой системы. Frozen DA3 сохраняет
явный Python своего исследовательского окружения.

`novel_view.diagnostics.gen3c` запускает текущие prompt/VAE model-ready
проверки через тот же изолированный процессный слой и Stage 6 artifact
formats. `novel_view.diagnostics.training` отдельно проверяет только явный
профиль A100/CP4, MIG, Apex/`amp_C`, NCCL и BF16 collectives. Ни одна из этих
диагностик не вызывается `plan` или обычным workflow; реальный прогон
отложен до Stage 11. Соседняя `novel_view.diagnostics.image` вызывается
только явным candidate-build и проверяет установленные prefix без GPU,
моделей и данных.

`diffusion/code/research/da3_nested/adapter.py` сохраняет замороженный
pose-conditioned DA3-прототип вне устанавливаемого wheel. Его соседний
`_worker.py` запускается по явному пути; зависимости направлены только
из research в `novel_view`, не обратно.
Используется checkpoint
`depth-anything/DA3NESTED-GIANT-LARGE-1.1` в отдельном `novel-da3`.
Официальному API передаются только source RGB, локальные измеренные W2C и
`K`; `align_to_input_ext_scale=True` обязан вернуть depth в масштабе входной
метрической траектории. Результат структурно проверяет, что возвращённые
conditioned W2C равны входным, а `K` отличается только известным
upper-bound resize и округлением к patch 14. Эти камеры не называются
предсказанными.

DA3 работает на своей сетке. Для source `704×1280` реально проверены
`process_resolution=504`, `896` и `1280`, которым соответствуют
`280×504`, `490×896` и `700×1274`. Разрешение передаётся явно при каждом
вызове: адаптер не скрывает научный выбор за значением по умолчанию. Сырые
depth, confidence и необязательная sky mask сохраняют модельную сетку.
Адаптер возвращает их на исходную source-сетку ближайшим соседом, как
официальный путь оценки DA3, затем применяет явную растровую и
sky-валидность. Интерполяция не заявляет, что низкоразрешённая модель
восстановила новые детали.

DA3-адаптер не добавляет свободный запуск без камер, независимо
предсказанные камеры, выбор между measured/predicted camera pack, LiDAR,
пороги confidence, скрытый OOM fallback или разбивку последовательности.
Production-вызова этого адаптера нет: исследовательский API и его
численные тесты оставлены frozen до конкретного versioned workflow
либо отдельного решения об удалении. Поэтому `novel_view.models.da3` не
создан и не считается частью фактической модельной архитектуры.

`novel_view.models.vggt` владеет одним source-neutral raw worker/protocol для
EUVS и Waymo. Оба потребителя передают только ordered RGB, получают одинаковые
пять модельных массивов и отдельные peak RAM/VRAM. Worker запускается через
фиксированный `/opt/envs/vggt/bin/python` и общий no-timeout process handle;
пути private worker, `runtime.python.geometry` и общий request deadline
потребители больше не знают. EUVS передаёт raw prediction дальше без
обратной привязки к `SourceGeometryInput`.

Измеренные nuPlan K/W2C сохраняются в общем входе,
однако официальный VGGT-Omega не принимает camera conditioning, поэтому
worker получает только точный ordered source RGB. Сырой результат содержит
camera-Z depth в неизвестном масштабе VGGT-Omega, исходный confidence,
9D pose encoding и декодированные официальной функцией предсказанные OpenCV
W2C/K. Pose хранит
`[tx,ty,tz,qx,qy,qz,qw,fov_h,fov_w]`, где quaternion использует scalar-last
`xyzw`, в отличие от `wxyz` в nuPlan.
Model W2C использует внутреннюю world-систему VGGT, обученную относительно
первой входной камеры, но prediction не обязан давать для неё точную
identity. Его переносы и `depth_model_units` принадлежат одному неизвестному
gauge и обязаны масштабироваться вместе; ни одну из этих частей нельзя
независимо смешивать с measured nuPlan cameras. `depth_confidence` сохраняет
сырой некалиброванный score `1 + exp(logit)`, а не вероятность или маску.

Три режима меняют только подготовку входной сетки одного backend:

- официальный `balanced-512`: `704×1280 → 384×688`;
- экспериментальный `balanced-896`: `704×1280 → 672×1216`;
- экспериментальный прямой вход `704×1280`.

Balanced-режимы повторяют официальный Pillow bicubic и `ToTensor`; скрытой
нормализации RGB нет. Линейная `source_to_model_pixels` использует принятое
в проекте соглашение координат по границам пикселей без неявного сдвига
`±0.5`, а конкретные sample locations bicubic остаются свойством
исполнителя Pillow. Экстремальный aspect ratio, для которого официальный
loader сделал бы неявный центральный crop, пока отклоняется.

PyTorch и официальный пакет не импортируются в `novel-core`.
`models.vggt.protocol` один раз потоково материализует упорядоченный
`uint8 RGB [N,H,W,3]` как временный NPY, а raw backend читает семь отдельных
численных NPY без pickle. Эти NPY не
являются артефактами запуска или публичным форматом; частичный результат не
публикуется. Worker выполняет один совместный forward всей текущей
source-последовательности, поэтому не вводит внешний windowing.

`models.vggt.record` владеет постоянным source-neutral результатом: пять
прежних NPY (`depth`, `confidence`, `pose`, predicted W2C/K) и
`summary.json`. Writer получает raw prediction, ordered opaque tokens и
source size, создаёт caller-owned каталог и пишет туда напрямую. Новые
записи всегда содержат полный ordered tuple; reader выводит provenance из
его наличия и продолжает читать старые count/edge-token summaries как
`legacy-attested`. Он не импортирует EUVS и не решает, соответствует ли
запись выбранной сцене. Эта policy находится в
`generation.euvs.source_views` и не возвращается в model package. Модуль
также дедуплицирует exact ordered source tuples, читает/rasterize только
source RGB, строит source-neutral request и связывает готовый record с
физической EUVS source-последовательностью. Здесь же определён
`ResolvedEuvsVggtGeometry`; evaluation support использует этот owner.
Reader generation records v1–v5 возвращает существующие
`EuvsPairSelection/EuvsFrameSelection` без legacy config, сохраняя файловые
поля, literal direction и прежний opt-in. Старый compatibility import
`workflows.euvs_vggt` удалён после переключения последнего test caller.

Сырой backend не читает target RGB, не назначает confidence threshold, не
выравнивает модельные камеры с nuPlan, не переводит depth в метры и не
строит `PosedDepthSequence`. Он намеренно заканчивается до alignment;
следующую модельно-специфичную границу выполняет отдельный модуль ниже.

Dataset-neutral математика camera-pack живёт в
`novel_view.geometry.sim3`: она принимает только полный
predicted `float32 [N,3,4]` и measured `float64 [N,4,4]` OpenCV W2C,
проецирует модельные вращения на SO(3) и решает один orientation-first
Sim(3). Порогов качества и dataset-контейнеров в ней нет; тот же результат
может безопасно использовать Waymo measured-camera adapter.

`novel_view.models.vggt.alignment` передаёт этой математике явные
`prediction.model_w2c` и measured W2C. Он не принимает EUVS-объект;
величина поправки вращений остаётся частью цельного
`CameraPackAlignment` и сохраняется как диагностика.

Один Sim(3) переводит модельную world-систему `M` в систему первой
физической source-камеры `R`:

```text
X_R = s Q X_M + d,  s > 0,  Q ∈ SO(3)
```

Fit является лексикографическим orientation-first, а не совместной
оптимизацией метров и градусов. Сначала для каждой пары OpenCV W2C
строится относительный model-to-reference rotation и находится его
равновесное chordal L2-среднее. Неединственное среднее отклоняется как
математически неопределённая операция. Затем при фиксированном `Q` одним
условным МНК по всем camera centers находятся положительный `s` и перенос
`d`.

Aligned predicted W2C строятся через преобразованные центры, что явно
сохраняет связь камер и depth:

```text
C_R = s Q C_M + d
R_w2c_R = R_w2c_M Qᵀ
t_w2c_R = -R_w2c_R C_R
depth_z_m = s * depth_model_units
```

`K` при world Sim(3) не меняется. Сам alignment не масштабирует dense
depth и не переносит её на source raster: он возвращает один неизменяемый
`CameraPackAlignment` с масштабом, согласованными predicted W2C и
диагностическими center/rotation residuals. Следующий адаптер принимает
этот результат целиком.

На этом слое нет порогов качества, RANSAC/IRLS, inlier mask, LiDAR,
per-frame scale, замены predicted cameras на nuPlan или выбора рабочей
геометрии. Stationary-последовательность, неположительный scale и
неединственное среднее rotation отклоняются как структурно
неопределённые случаи, а не как политика качества сцены.

`novel_view.models.vggt.posed_depth` завершает модельно-специфичный переход
в общий `PosedDepthSequence`. Он принимает source-neutral raw prediction,
цельный `CameraPackAlignment`, source-validity и ordered sequence ID; EUVS
типы в модельный пакет не входят.

Depth переводится в метры одним scale всей последовательности. Для
balanced-режимов depth и исходный confidence переносятся с model-grid на
source-grid билинейно; direct `704×1280` не выполняет resize.
Предсказанные intrinsics переносятся по принятой пиксельной геометрии:

```text
K_source = inverse(source_to_model_pixels) @ K_model
```

`geometry_valid_mask` остаётся source raster validity, дополненной только
структурной проверкой конечной положительной depth. Confidence сохраняется
как неприведённое backend-поле без порога. Вне маски depth и confidence
равны нулю.

Результат является одной цельной predicted-camera-гипотезой, а не выбором
окончательной рабочей геометрии. Простая замена `K/W2C` на measured
nuPlan-камеры запрещена: она меняет луч и camera-Z той же точки.
Физически корректный measured-camera-вариант требует отдельного
`unproject → Sim(3) → reproject → z-buffer` этапа.

## Запуски

Текущие `euvs_generation/v1–v2` получают все машинные пути из фиксированного
`RuntimeContext`. Worker пишет lossless NPY прямо в
`attempt/pairs/<pair-name>/generated_rgb.npy`, а workflow затем напрямую
пишет `run.json`. Кеши и временный обмен находятся под `/cache`.

Этот R&D-контур доверяет настроенным локальным корням и обычной политике прав
рабочей машины. Он не проверяет inode, symlink races, UID, sticky-bit или
точный mode файлов. Научная идентичность результата сохраняется в строгом
предметном record.

Общего `state.json`, manifest или SHA нет. Pair paths выводятся только из
явной selection и exact source-view attempt. Directory scan, completed-output
reuse, resume и совместимость checkpoint этому workflow не принадлежат;
повтор выполняется новой attempt.

## Окружения

Production-образы разделяют несовместимые части на три фиксированных prefix:

- `/opt/envs/core` — лёгкая orchestration, readers и workflows;
- `/opt/envs/gen3c` — Gen3C, evaluation stack и в варианте `moge` MoGe;
- `/opt/envs/vggt` — отдельный VGGT-Omega stack.

Один и тот же wheel `novel-view-pipeline` устанавливается во все три prefix
на этапе сборки. `novel_view.runtime.executables` содержит их буквальные
Python locators без scan, fallback или registry. `production-core` включает
все три prefix, а `production-moge` добавляет пакеты только в `gen3c`;
тестовый `cpu-test` остаётся отдельным target только с `core` и тестами.

Отдельное исследовательское DA3-окружение не определяет состав production
images; старые runtime YAML больше не читаются. Готовность GPU, весов и данных не выводится
из наличия prefix и остаётся явным `doctor` либо финальной Stage 11
кампанией.

## Реализованные границы

Поддерживаемые команды и процессы перечислены в ближайших README пакетов,
`workflows`, корневых `jobs` и `tests`; этот файл фиксирует их связи и владельцев,
но не дублирует покомандный статус реализации. Основной EUVS-путь разделяет
чтение данных, геометрию, Gen3C, постоянный `run.json`, оценку и сравнительный
 benchmark. Поддерживаемые Waymo/DDW preparation, R4c LoRA training и matched
 evaluation разделены между `preparation`, `training` и `evaluation`;
 исторические сценарии также получили владельцев в `preparation`, а
 переходный `finetuning` удалён. Gaussian workflow находится внутри diffusion
 и обменивается результатами с 3dgs только через файловый contract.

Тяжёлые модели работают в изолированных окружениях через общий runtime.
Научная идентичность хранится в типизированных записях соответствующего
предметного слоя; файловые каталоги, блокировки и локальные пути не заменяют
её и не становятся общим registry. Измеренный FRONT-A LiDAR уже входит в
Waymo/DDW preparation, экспериментальное R4c v2 и matched evaluation;
их конкретные контракты описаны выше. Новые multi-source порядки или иные
преобразования depth не включаются скрыто в существующие версии workflow.
