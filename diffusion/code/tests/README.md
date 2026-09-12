# Проверки

Исторический DDW сохраняет `test_waymo_ddw_legacy_{science,primitives}.py`:
независимые маски и пороги, uint8/headroom parity, два warp и global
reference, освобождение плотных результатов. `test_ddw_fit_science.py`
объединяет assignment/fit и отдельный rejected-only audit с настоящим MP4;
`integration/test_ddw_legacy_orders.py` закрепляет четыре разных
workflow/record/STOP-порядка.
Они не заменяют численные тесты поддерживаемого Stage 6 или native Gen3C.

`unit/` содержит быстрые модульные и небольшие сквозные проверки контрактов
пакета, включая компактное численное доказательство общей camera/trajectory,
raster cover/crop и polyline-математики, nuPlan W2C/OpenCV-обвязки,
concrete FRONT alpha-zero/crop/half-pixel K, measured Waymo W2C и sparse
LiDAR camera-Z с неизменным every-fifth evaluation split после фильтрации,
а также exact MoGe relative-L1 weighted median, holdout-изоляцию, measured
camera identity и прямое распространение model/process ошибки,
точную 121-frame cosine DDW-траекторию для обоих знаков, zero magnitude и
неизменные measured rotation/исходный W2C,
а также точную связь A→B→A′: actual outward RGB/depth/known, ones virtual
rectification, measured return target и отсутствие retry после process
failure,
точный clipped/banker's-round перевод normalized condition в uint8 THWC и
little-endian packed-known с отбрасыванием padding bit,
EUVS `frames.csv` routing, сохранение ordered tokens/traversal, независимый
рост image/pose timestamps и роль-независимую сборку RGB-последовательностей,
общий camera-pack геометрических backend и лёгкий dataset-neutral
`PosedDepthSequence` с ordered ID без вложенного RGB, DA3 Nested и сырой
VGGT-Omega-адаптер на общем межпроцессном слое без импорта PyTorch, а также
orientation-first Sim(3)-согласование predicted и measured камер и перевод
цельной VGGT-гипотезы в метрический `PosedDepthSequence`. Здесь же проверяется
чистая временная раскладка exact EUVS targets на совместимые окна pinned
Gen3C и компактная оконная материализация query W2C/K и projected-nearest
single-source indices. Отдельный малый value-test закрепляет оба seed-режима,
единственный seam и прежний metric/neighbour context-depth kernel без модели
и внешних данных. Соседний dataset-neutral metric-test закрепляет
finite/perfect/undefined PSNR/SSIM, effective pixel/patch mass и
equal-camera → pair → location reductions. Один Cache4D serializer отдельно
проверяется с context и без него: точные массивы/индексы и однопроходное
чтение RGB. Это не четыре фиктивных reader-сценария по одному названию.
Отдельные случаи доказывают связь независимо
созданных RGB/depth/pair/mask объектов по ordered tokens и отклонение другого
порядка либо другой source-сетки. Здесь же проверяются строгий Gen3C run-record JSON,
переносимые locators и точная target binding.
EUVS samples проверяются на точный переданный record path, output slots,
target-порядок и target-only rasterization без чтения source RGB. Один
компактный support-тест закрепляет direct token PNG cache, polarity `0/255`,
duplicate collapse, nearest remap, four-neighbour z-buffer, обе support-
дорожки и сохранённый disocclusion. Настоящий evaluation chain связывает
JSON/output slots, target-only RGB, сохранённый VGGT и две маски. Родительская
Gen3C-сессия имеет отдельный CPU proof. `integration/test_model_api.py`
проверяет Cache4D/warp, официальный VGGT loader и используемые сигнатуры
SAM2/MoGe/LPIPS/DINOv2 без весов; команды — в README integration.
Новый R4c objective proof закрепляет буквальные четыре noise draw до CP
broadcast, native conditioner constants, CP2/CP4 value+gradient parity,
неотбрасываемый condition и differentiable recompute seam без v2/depth.
Соседний компактный proof захватывает фактические CP и gradient-average
groups, exact 28-block rank-8/scale-1 LoRA control, FusedAdam-параметры и
порядок одного update. Validation измеряется отдельно без parameter update;
hardware/filesystem gates и общий trainer в этом тесте отсутствуют.
Ещё один CPU-proof сравнивает два непрерывных update с
`update → checkpoint → resume → update`: совпадают loss, LoRA, FP32
FusedAdam, scheduler и RNG, а legacy cursor переводится только через
85-строчную карту. PT и соседний JSON проверяются отдельно; реальные веса,
CUDA и файловый preflight не запускаются. Concrete loop proof закрепляет
одинаковый epoch-порядок fresh/resume и PT/JSON только на границе эпохи; v2
fresh выполняет fit до model load, а resume восстанавливает observer/lineage
без повторного fit. Параметризованный fake runner-case запускает только
точный `_worker_v1.py` либо `_worker_v2.py`, а CPU selection-case принимает
ранний шаг и сохраняет первый record при полном равенстве без открытия PT.
V2-only depth proof отдельно проверяет точную свёртку sparse
FRONT-A LiDAR в latent log-depth grid, непересекаемый holdout и разрезание
values/mask только через upstream CP primitive. Он же доказывает точные 577
FP32 параметров observer, детерминированный training-only fit, отдельный
validation-median signal gate, восстановление RNG и раздачу четырёх
замороженных tensor; v1 этот модуль не импортирует.
`test_r4c_depth_loss.py` отдельно проверяет stop-gradient clean branch,
градиент predicted branch, нулевую ошибку при равенстве и точное глобальное
среднее/градиент `P*E_r/N` для CP2/CP4, включая пустые local shards.
Он же проверяет совместный EDM+depth update до gradient clip и то, что v2
validation усредняет уже целые item-метрики между samples.
Matched DDW evaluation proof связывает ранние v1/v2 records одного step,
проверяет полную замену adapter, four-neighbour heldout depth, all-121 RGB,
строгий итоговый JSON, четырёхпанельный MP4 и сквозной fake workflow без
`request.json`. Модели, CUDA и raw Waymo в нём заменены только на дорогих
границах.

В Stage 9.1 CPU-проверки из бывшего `model/`, включая все шесть численных
R4c файлов, перенесены в `unit/` и `integration/`. Настоящие no-weight API
проверки установленных Gen3C/VGGT/SAM2/MoGe/metrics находятся в `integration/` с явным
marker `native_api`; `./distil3d test all` исключает только этот marker,
не всю модельную область. Их отдельный выпускной прогон требует принятых
production-prefix. В 9.14 последний экспериментальный DA3 API перенесён в
`integration/test_da3_native_api.py` с тем же marker, и `model/` удалён.
Это отдельная исследовательская проверка вне обещанной поддержки
`core+moge`, а не одиннадцатый обязательный выпускной API-случай.

Прежние операторы `data/` перенесены к предметным владельцам в 9.5–9.9;
папка и временный импорт через conftest удалены. Production не импортирует
tests. Generated-Parquet и настоящий SQLite/CSV/RGB seam выполняются в
обычном CPU-образе с PyArrow/OpenCV; реальные данные не используются.

`support/` содержит только небольшие фабрики и CPU-двойники, которые
переиспользуют несколько автоматических проверок. Production-код эту папку
не импортирует.

`contract/` доказывает, что пакет загружен из установленного wheel, production
не импортирует `tests` или `research`, новый plan CLI не импортирует model
stack, `novel_view.geometry` зависит только от NumPy/stdlib и импортируется
без OpenCV/PyArrow/Torch, а внешняя job/recipe-оболочка, golden plan и
operational `attempt.json` имеют заявленную форму. Здесь же закреплены
read-only reader,
единственный Python writer и отсутствие job/workflow imports из `runs`.
Отдельный R4c contract сохраняет declared training/validation order по
`sample_id`, отклоняет неоднозначный внешний split и строго разбирает
ровно 85 строк закрытой карты старых fit indices без чтения данных. Он же
закрепляет `PreparedRecord` join, segment-disjoint split, буквальный PCG64
epoch order и canonical next-item cursor без изменения global NumPy RNG.
Отдельный Gaussian producer-v1 contract проверяет полный JSON, исходный
порядок кадров и относительные пути без открытия отсутствующих payload-файлов.
Соседний selection-contract закрепляет строгие EUVS/Gaussian YAML, пять
Gaussian kinds, точный порядок concrete ID и холодный импорт без model/data
стека.
Компактный Gaussian v1–v2 proof дополнительно создаёт настоящие
producer rows, проверяет raster/W2C/slots, right-aligned clip
boundaries, fresh source seeds, одну resident-session seam, runner routes и
строгие v1 records. Соседние compact proofs закрепляют dense v3–v4 records,
previous context/seam и pure-CPU v5 join из exact v2/v3 records в ordered
native PNG/`prepare.json`.

`integration/` содержит generated-Parquet end-to-end Waymo v2 reader case с
переставленными sensor rows, каноническим порядком и точными camera/LiDAR
payload. Соседний DDW source case доказывает точные 121 ordered keys только
явно заданного segment/start. Оба модуля имеют холодный импорт без
PyArrow/OpenCV/Torch/worker. Forward-warp cases отдельно закрепляют точную
покадровую нормализацию `0/127/128/255`, прямой NPY process lifecycle,
распространение process failure, literal depth sanitization/mask/hole
convention, chunk size 2 и необязательный pinned-utility CPU golden без
CUDA. Prepared-artifact proof выполняет прямой round-trip BF16
base/pose/prompt и sparse measured LiDAR, проверяет exact fields, неизменный
optimizer/evaluation split, отсутствие dense depth и swapped sample ID.
Два prompt-bake proof закрепляют один process-вызов, прямой читаемый
artifact, `encode_prompts([""], 512)`, first-token mask, zero padded tail,
CPU BF16/contiguity и отсутствие retry после process failure без CUDA/model
load.
Компактный VAE-bake proof численно различает BF16-first target и
float32-first condition, доказывает BTNCHW/BTN1HW layouts, ordered batch
`8 + 2`, один model load на worker, frame-0 source latent, exact base/pose
payloads, публикацию staged LiDAR в Gen3C prefix и остановку на первой
process/item ошибке без retry.
Prepared-record contract одним малым fixture доказывает selection order,
121 timestamp, exact v1 fields/model identity, три относительных artifact
locator и прямой JSON round-trip; filesystem artifacts не подготавливаются.
Полный дешёвый `ddw_preparation/v1` seam читает настоящие generated-Parquet
keys для tracked job, подменяет только дорогие raster/model границы,
проверяет `moge`/`inference_cp1`, attempt-owned log paths и запись
`prepared.json` строго после batch bake.
Соседний R4c baseline batch proof открывает только base/pose/prompt через
Stage 6 readers, передаёт clean/source/pose/prompt и доказывает source-rank
error handoff до tensor broadcast; LiDAR и depth-v2 в этот путь не входят.
Соседний DDW evaluation workflow case выполняет ordered validation sample,
сохраняет JSON+MP4 и доказывает холодный parent без Torch/OpenCV/Cosmos/MoGe.
Здесь же холодно запускаются два сохранённых
legacy entry point и новый внутренний plan из установленного wheel вне
checkout. Настоящие subprocess-
проверки runner/process/store сохраняют коды 0/23, изолируют две attempts и
передают SIGTERM leader/child/grandchild одной process group. Отдельный
resume-case требует точный `job/run/source-attempt`, восстанавливает
сохранённый resolved job без исходного YAML и контейнера-источника и дословно
переносит opaque checkpoint в новую attempt. Двухчастный run-ref отклоняется
до Docker, а повторный resume адресует уже выбранную attempt. Полный Docker-case
мягкой и принудительной остановки, исчезновения host PID и `remove` находится
в корневом `tests/launcher/test_lifecycle.sh` и автоматически входит в
`./distil3d test all` после pytest.

Contract-набор наблюдаемости дополнительно проверяет read-only снимок exact
attempt без записи и выбор последней записанной attempt для run-ref. Exact
отсутствие не создаёт каталог или файл; порядок выбирается по сохранённому
`started_at`, а Docker-состояние остаётся за корневыми launcher-тестами.

В 9.13 внешние attempt, EUVS/Gaussian и исторические fit records объединены
по владельцам в `contract/`. Неиспользуемые metrics/video doubles удалены;
batch-проверка R4c использует настоящие маленькие BF16 tensors, подменяя
только distributed операции. Читаемые исторические форматы и постоянные
JSON fixtures не изменялись.

Тестовые данные должны быть маленькими и синтетическими; датасеты и
вычисленные результаты в Git не добавляются.
