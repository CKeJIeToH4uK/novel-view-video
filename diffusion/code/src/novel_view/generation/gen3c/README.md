# Адаптер Gen3C

Подпакет переводит подготовленные численные входы в соглашения закреплённого
Gen3C и изолирует его несовместимое модельное окружение.

- `request.py` задаёт статическую границу научного источника и тонкий
  generation request без runtime-validator; `protocol.py` добавляет к общему
  Cache4D serializer только markers, output и context-result имена.
- `recipe.py` один раз разбирает общие внешние Gen3C/LoRA параметры EUVS и
  Gaussian в два небольших dataclass; зависит только от стандартной библиотеки.
- `model_request.py` из готового recipe без открытия файлов собирает конкретные base/full/LoRA
  model и sampling records. Эта функция выделена после второго реального
  потребителя — EUVS и Gaussian — и не является registry или универсальной
  фабрикой моделей.
- `resources.py` один раз описывает фиксированные container cache/offline
  переменные и launcher-assigned CP для этих двух callers. Он не проверяет
  каталоги, GPU или веса и не выбирает execution preset.
- `euvs_input.py` отдаёт уже связанные предметным планом source RGB/depth
  строки и строит лёгкие conditioning-массивы без знания имён NPY и без
  изменения временной, camera или source-selection политики.

- `timeline.py` размещает переданные exact target timestamps на 24-fps сетке
  и описывает окна длины 121 с перекрытием в один кадр; EUVS-типы он не
  импортирует.
- `conditioning.py` совместно переносит source и measured target камеры в
  систему source anchor, через `novel_view.geometry.trajectory`
  интерполирует direct query W2C/K и материализует projected-nearest
  single-source indices ровно для одного окна.
- `windows.py` хранит чистый порядок model-вызовов: global start/stop,
  источник seed, согласованную замену первой W2C/K-строки и единственное
  удаление seam-кадра. Он не запускает Cache4D или Gen3C.
- `context_depth.py` после всех окон выравнивает готовую raw MoGe depth по
  metric Cache4D depth и оставляет локально и между соседними камерами
  согласованные значения; Gaussian-типы и файловый протокол ему неизвестны.
  Raw per-slot вызов и единственный resident load принадлежат
  `models.moge`; составной worker вызывает их прямо в памяти без нового
  процесса или NPY.
- `session.py` владеет parent lifecycle одной context-parallel группы,
  последовательными request markers, состояниями `running/broken/closed` и
  телеметрией; `execute.py` выполняет через ту же границу один запрос.
- Предметные постоянные EUVS records находятся в соседнем
  `generation/euvs/record.py`, а Gaussian records — в
  `generation/gaussian/record.py`; этот model-specific подпакет владеет
  только временным протоколом и исполнением Gen3C.
- `_worker.py` является единственной составной точкой torchrun: он соединяет
  сырой Cache4D runtime, порядок окон и optional context-depth проход. Сама
  resident-модель уже принадлежит
  `models.gen3c.session.Gen3cModelSession`: там живут official pipeline,
  base/full/LoRA construction, CP topology, one-window inference и
  request-level RNG. По умолчанию окна связаны авторегрессионно;
  экспериментальный `source-reseed/v1` начинает каждое следующее окно с
  настоящей согласованной source-строки.
  Cache4D импортируется из закреплённого установленного prefix; явный путь к
  старому checkout остаётся только совместимостью для ещё не перенесённых
  callers.

Глобальный план остаётся пропорционален `N+M`. Exact target slots дословно
сохраняют существующий mapping, padding удерживает последнюю камеру и source,
а соседние окна разделяют одну и ту же строку и один общий Cache4D.

Полные RGB/depth/mask-тензоры существуют только во временном каталоге и в
модельном процессе. Публичный результат содержит coverage по всем слотам,
долю отброшенной глубины свыше официальной границы 100 м, контроль overlap и
не более семи RGB/mask-кадров для просмотра. Coverage и far-depth считаются
от полного растра, overlap RGB измеряется в диапазоне `[-1, 1]`, а время и
CUDA peak охватывают сборку тяжёлых входов, unprojection, создание cache и
все рендеры.

Компактный Cache4D-результат остаётся самостоятельной диагностикой и не
становится входом diffusion. Diffusion-worker заново собирает Cache4D и сразу
потребляет полные render-тензоры в том же процессе; cache и warp на диск не
сериализуются.

Базовый NPY-формат имеет одного владельца в
`novel_view.models.gen3c.cache4d.protocol`. Он по одной строке пишет source
RGB/depth/valid, сохраняет camera/index arrays и optional второй context
layer. EUVS, Gaussian, dense и Waymo объекты только строят
`Gen3cConditioning` и не знают файловых имён.

`Gen3cGenerationSession` получает именованный model artifact и научные
sampling-параметры один раз, а каждый `generate` — отдельный точный
`Gen3cConditioningInput` и путь результата. Запросы выполняются строго
последовательно; конкурентный `generate` отклоняется, а `close` дожидается
активного запроса. Ошибка делает сессию непригодной для продолжения. Перед
каждым запросом восстанавливается одинаковое post-construction RNG-состояние,
но внутри окон одного запроса RNG не сбрасывается. Совместимый
`generate_gen3c_sequence` создаёт такую сессию для одного запроса.
Фиксированные свойства закреплённой модели —
`704×1280`, 24 fps и окна 121 с шагом 120 — не становятся настраиваемым
профилем. `autoregressive/v1` передаёт следующему окну последний
сгенерированный кадр предыдущего. `source-reseed/v1` вместо этого выбирает
по глобальному начальному слоту одну source-строку и совместно использует её
RGB, W2C и K как локальную строку 0. В обоих режимах сохраняются один общий
Cache4D, прежняя глобальная ось, padding и последовательное RNG-поведение.
Rank 0 пишет overlap только один раз прямо в выбранный caller-ом
`uint8 NPY [T,H,W,3]`; staging, hard link и повторный output validator не
используются. Результат в памяти сохраняет numeric input, `model_id`,
sampling-параметры и телеметрию для существующих run records.

LoRA provenance дополнительно несёт literal working manifest, evidence,
число эпох и реально применяемую strength. Generation только переносит эту
уже выбранную training identity и model-ready checkpoint; её создание и
выбор принадлежат training-этапу. Worker restricted-load-ит только вложенный
`adapter`, не читает training topology, optimizer, scheduler, progress или
RNG. Для CP2 используется уже созданная EUVS-группа. Отдельный adapter-файл
или merged 7B checkpoint не создаётся.

Полные входы каждого запроса по-прежнему передаются временными NPY. Только
rank 0 наблюдает файловые маркеры; при CP2 он рассылает остальным rank малую
числовую команду через существующую NCCL-группу. Готовность публикуется после
закрытия output/input memory map, очистки request-local Cache4D и общей
синхронизации. Телеметрия различает текущую CUDA-память до запроса,
request-пик и остаток после очистки. Для каждого результата старые
`per_rank_peak_cuda_*` сохраняют смысл полного пика одноразового пути как
максимум startup и именно текущего request; это не накопительный максимум
всей истории resident-процесса. Автоматический перезапуск worker сюда не
входит.

`euvs_generation/v1–v2` создают в каталоге пары `generated_rgb.npy` и
`run.json`. Запись содержит ordered source/target tokens, exact target slots,
target-level source indices, переносимые locators geometry/checkpoint и
фактический NPY descriptor. Исторический LoRA v4 хранит identity без
strength; новый v5 дополнительно сохраняет реально применённую strength.
SHA-проверок в R&D-контуре нет.
Она не дублирует K/W2C, полный 241-slot plan, GPU ids, телеметрию или файловый
inventory. Старый NPY без sidecar не считается нативно завершённым
запуском. Geometry provenance описывает способ подтверждения ordered source
tokens и не привязан к имени конкретного backend.

Обычные новые native-записи используют v2 и содержат только
`execution.context_parallel_size` со значением 1 или 2. Строгий loader также
читает v1 как `execution=None`, не угадывая topology по имени файла. Такие
старые записи подходят одиночной оценке, но не exact CP campaign/replay.
`source-reseed/v1` использует v3, а новые LoRA-записи — v5.
Source-reseed использует v3 и дополнительно сохраняет
`sampling.window_seed_policy`; старые v1/v2 при чтении однозначно означают
`autoregressive/v1`. Поэтому два оконных режима не могут случайно считаться
одним и тем же завершённым запуском.

Context parallel поддерживается только для одного или двух rank. Размер три
отклоняется до загрузки весов: временной latent закреплённой модели имеет длину
16 и не делится на три. Каждый rank всё равно владеет полными весами и своим
Cache4D; разделяются последовательностные активации, а не checkpoint.

Рабочий scratch создаётся под выбранным caller-ом cache root; в нём живут
только request NPY, markers и worker log. RGB и optional context depth
пишутся сразу по выбранным workflow путям, а `run.json` создаётся после
успешного terminal marker. Код доверяет локальному
R&D-каталогу и не навязывает UID, sticky-bit или точный mode: нужные права и
группы задаются окружением рабочей машины.

Текущий projected-nearest single-source, `filter_points_threshold=0.05` и
`foreground_masking=False` — явный baseline. Target RGB не
читается. Multi-source, bridge и скрытое принятие качества геометрии не
поддерживаются. `euvs_generation/v1–v2` используют одну долгоживущую сессию
на всю selection-order attempt и освобождают принадлежащие запросу Cache4D
и входы между парами. Внутренних очередей, сканирования готовых файлов,
возобновления и автоматического перезапуска после ошибки сессии нет.
