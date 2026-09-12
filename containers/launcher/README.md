# Внутренние функции host launcher

Папка содержит небольшие Bash-функции корневой команды `./distil3d`.
Пользователь не запускает эти файлы напрямую.

## Текущие файлы

- `profile.sh` буквально читает шесть ключей `.env` ровно по одному,
  разбирает упорядоченный список GPU без пустых элементов/повторов и выбирает
  первые 0/1/2/4. Он формирует единые параметры сети, устройств, IPC и memlock
  для actual create; Python получает готовые значения для отображения плана;
- `doctor.sh` реализует явные `doctor host` и `doctor job`. Последняя
  выбирает образ и окончательный план, затем запускает один `--rm`
  контейнер с общими восемью mounts, UID/GID и GPU resources. Она не
  создаёт attempt, не вызывает host doctor и сохраняет код диагностики;
- тот же `doctor.sh` даёт два явных исторических маршрута
  `doctor waymo-ingress|waymo-raster --profile <profile> -- <arguments>`.
  Они запускают существующую диагностику frozen Waymo-окна в установленном
  `core` без GPU, job и attempt, с теми же mounts и UID/GID. Аргументы после
  `--` передаются буквально; Waymo root задаётся под `/data`, а закрытый
  split — под `/project-config/selections/local/`. Нового preflight обычных
  workflows эти команды не добавляют;
- `images.sh` сохраняет отдельный Linux/amd64 `cpu-test` только для тестов и
  синтетического lifecycle. CPU pytest-контейнер использует `--network none`
  и `--hostname localhost`: локальный torchrun разрешает собственное имя
  без внешней сети. Для внешней команды `build` он собирает
  фиксированный `production-core` либо последовательность
  `production-core → production-moge` без profile, передавая вычисленные Git
  и lock metadata через build arguments. Общую сборку одного project wheel,
  его установку в три production prefix и финальные слои выполняет
  Dockerfile. При явном `--candidate-output` этот файл требует чистый
  неизменившийся checkout и новый каталог вне него, запускает CPU gate и
  без-GPU диагностику установленных prefix, сохраняет ровно два образа в
  `images.tar`, а внутреннему Python writer передаёт канонический список семи
  locks и два exact image ID для `candidate.json`. Встроенный reader печатает
  только известные `key=value`; Bash не разбирает JSON и не использует
  `eval`. Обычный selector работает в уже собранном `core`, а окончательный
  `plan` — в выбранном `core|moge` по точному image ID. Конфигурационные
  контейнеры монтируют только `jobs/recipes/selections` read-only.
  Для них базовый CUDA entrypoint
  отключён, чтобы stdout принадлежал только selector либо JSON plan.
  Полученный plan уже показывает полный внутренний argv Python runner, но не
  создаёт attempt и не запускает этот argv. Метки читаются по выбранному ID;
  изменение tag или несовпадение окончательных variant/preset завершает
  команду до создания attempt. Скрытой сборки при обычных командах нет;
- `attempts.sh` создаёт run/attempt ID, восемь mounts, labels, UID/GID и один
  сохраняемый контейнер через `create → start`. Foreground следует за логами,
  затем принимает exit code только из terminal `exited|dead`; обрыв logs при
  живом container даёт служебный код 1 и recovery ref, но не посылает сигнал.
  `--detach` пропускает только просмотр. Ctrl-C нейтрально завершает follower
  с кодом 130, но не утверждает состояние и не останавливает контейнер. Resume
  читает сохранённый resolved job через `core`, затем строит окончательный
  план в выбранном образе с одним read-only `/runs`. Исходный YAML не нужен,
  checkpoint передаётся без проверки. `run` также строит окончательный план
  до создания attempt. Явный `stop` разрешает exact attempt, вызывает
  `docker stop --signal SIGTERM --time <grace>`, затем печатает фактические
  state/exit/OOM из повторного inspect. Код 137 не считается OOM без
  отдельного `OOMKilled=true`;
- `observability.sh` разрешает run-ref либо exact attempt-ref по Docker
  labels и read-only Python-снимку `/runs` внутри `core`. `status` показывает переносимое
  состояние рядом с Docker state/exit/OOM и ничего не записывает; `logs`
  читает либо следует за stdout/stderr точного контейнера. Несколько активных
  attempts дают код 2 и exact candidates, container без record требует exact
  ref. `remove` удаляет только завершённый container по exact ID и оставляет
  attempt-root и record;
- `accept.sh` содержит единственную публично вызываемую
  `accept_a100_4()` и владеет bootstrap/закрытием отчёта. Команда создаёт
  новый report, загружает только соседний
  `images.tar`, разбирает `candidate.json` установленным `core` без `eval`,
  сверяет clean host revision, точные IDs/labels и non-root user обоих
  образов. После parse каталог один раз получает candidate ID; в него
  попадают только candidate, разрешённые host/image/archive facts и строгий
  `summary.tsv`, но не profile, секреты или сам архив. Здесь же один явный
  native TE recompute запускается на первой GPU профиля в Gen3C prefix
  точного core-образа, с read-only tests и записываемым JUnit. Отдельного
  job/workflow или выдуманного training record у него нет;
- `accept_foundation.sh` создаёт campaign-local копии фиксированных форм,
  подставляет только известные внешние роли Stage 11, проверяет четыре
  различные A100 без MIG, образы и конкретные job diagnostics, а также
  сохраняет окончательные планы до каждого запуска. Шесть исторических
  Waymo-форм копируются из `jobs/legacy/waymo/`, закрытые selections берутся
  из `selections/local/stage11/legacy/`;
- `accept_attempt.sh` ведёт ровно одну активную попытку, отдельный просмотр
  её логов, terminal inspect, точную остановку при сигнале и ограниченные
  записи evidence. Host не называет Python-owned `attempt.json`: небольшой
  `attempt.tsv` приходит от существующего reader, а selection читается
  установленным Python-владельцем схемы. Для шести исторических Waymo
  случаев тот же установленный Python читает предметный outcome и
  сохранённый worker exit; shell отдельно сопоставляет actual container
  exit, состояние и OOM. Ожидаемый scientific STOP с кодом 2 принимается
  только для canary v2, probe и survey с пригодным record: attempt остаётся
  `failed`, а технический результат приёмки — `succeeded`;
- `accept_campaign.sh` содержит буквальный порядок EUVS, Gaussian, отдельной
  Waymo preparation одного 121-кадрового sample, TE recompute, R4c LoRA v1,
  экспериментального v2 и matched evaluation. После исходных 19 случаев
  отдельно выполняются historical depth comparison, CPU depth-selection
  по полному операторскому набору восьми reports и четыре DDW-порядка.
  Только выбранный MoGe допускает DDW; gate STOP или выбранный VGGT дают
  явный dependency skip. Candidate comparison не подменяет вход gate.
  DDW doctor вызывается после привязки конкретного gate attempt. Всего
  26 случаев и три служебные строки отчёта. TE должен пройти до обоих
  fresh training; ошибка пропускает их, но не независимые сценарии.
  Четыре DDW не зависят от успеха
  друг друга. В нём нет общего DAG, scheduler, retry или поиска последнего
  запуска;
- `tests.sh` реализует `test unit|contract|integration|all` внутри
  обновлённого `cpu-test`, без host Python и скрытого `doctor`. Явный выбор
  `not native_api` оставляет весь CPU-набор и отделяет проверки установленных
  модельных API; их release-проверка выполняется на production-prefix отдельно,
  без скрытой production-сборки в test all. `integration` включает также
  корневые Python-проверки запускателя, поэтому объединение отдельных групп
  совпадает с Python-набором `all`. Только `all`
  после успешного pytest запускает на host настоящий
  `tests/launcher/test_lifecycle.sh`; остальные группы не запускают этот сценарий.

Machine data/model/prepared/runs/cache roots при `plan` передаются Python как
строки для будущей mount-таблицы и не подключаются к Docker. Selector
возвращает только `job_name`, `image_variant` и `preset`; shell не выполняет
его вывод и не читает JSON. `cpu-test` выбирает только `_lifecycle_smoke`;
остальные jobs выбирают `core|moge`. Обычные команды используют явно
собранные образы и не требуют candidate.json. GPU-attempt получает quoted
`device=...`, локальные индексы `CUDA_VISIBLE_DEVICES=0..N-1`, `--ipc host`
и `memlock=-1:-1`. CPU и конфигурационные контейнеры сохраняют только
`--network none`, даже если профиль перечисляет GPU. Устройства не
опрашиваются при планировании. Явный `doctor job` получает те же параметры
уже для настоящей диагностики. `accept` использует профиль только после
bootstrap и никогда не копирует его в возвращаемый отчёт.

Корневой `distil3d` владеет только разбором внешней команды. Каждый новый
файл здесь добавляется вместе с одной реально работающей командой; общий
`utils.sh`, научная логика и разбор JSON сюда не помещаются. Общие подмены
Docker/Git для наблюдаемых тестов лежат в корневом `tests/support/` и не
используются production launcher.
