# Контейнеры diffusion

Папка владеет одним многостадийным Dockerfile для CUDA 12.4 / Ubuntu 22.04
и точными Linux/amd64-зависимостями. Пользователь запускает его только через
корневой `./distil3d`; отдельного Compose, host Conda или `PYTHONPATH` нет.

Первая production-пара — `core+moge`. `production-core` содержит
три независимых prefix:

- `/opt/envs/core` — лёгкий CLI, конфигурация и readers;
- `/opt/envs/gen3c` — Gen3C, evaluation и CUDA/PyTorch workers;
- `/opt/envs/vggt` — VGGT-Omega.

`production-moge` наследует `core` и добавляет 29-wheel delta в уже
существующий `/opt/envs/gen3c`; его wheel-builder наследует готовый
`gen3c-deps` и не копирует этот большой prefix в отдельную CUDA-devel
стадию. `/opt/envs/moge` не создаётся. DA3 остаётся
`research-only`: его исторические locks сохранены, но target и обязательная
строка candidate отсутствуют.

Один wheel из `diffusion/code` собирается общей `project-wheel-builder`
стадией и напрямую копируется для установки во все три prefix.
Его `LICENSE`/`THIRD_PARTY_NOTICES.md` устанавливаются вместе с metadata.
В `licenses/` лежат точные дополнительные тексты ANTLR, Decord, Triton,
ModelOpt, GPLv3 и GCC runtime exception 3.1; production-core копирует их в
`/opt/licenses/distil3d`, а moge наследует этот каталог.
`copy_conda_metadata.py` вызывается только при сборке и сохраняет
`info/index.json`, доступные `about.json`, `licenses/` и `recipe/`
фактически установленных Conda-пакетов в `/opt/licenses/conda/<prefix>/`.
Кеш пакетов и сам помощник подключаются временно и в runtime не попадают.
Рецепты содержат исходные тексты сборочных команд/патчей, но не подменяют
полные соответствующие исходники. Полнота бинарных уведомлений и право
передачи архива ещё не подтверждены; подробности — в корневом docs/licensing.md.
В runtime также остаются ровно два package tree с лицензиями, без `.git`
и тестов: Gen3C и DINOv2. Checkouts Apex, Transformer Engine, SAM2,
VGGT-Omega и MoGe нужны только builder stages. Данные, модели, checkpoints, wheelhouse,
системный toolchain devel-образа и тесты в production-образ не входят.
Исключение среди весов — маленькие линейные `.pth`, включённые самим LPIPS;
внешние pretrained models по-прежнему подключаются отдельно.
Готовый exact Conda-prefix переносится целиком: его lock-owned
compiler/CUDA-devel пакеты намеренно не вырезаются после replay.

## Текущие targets

Завершённый `8.3` реализует общую сборку
project wheel, независимые builders и три production prefix: `core`,
`gen3c` и `vggt`. `cpu-test` использует те же core-зависимости и тот же
project wheel, но устанавливает их в отдельной тестовой ветви сборки.
В завершённом `8.4` добавлены и проверены на Linux/amd64
`production-core` и производный `production-moge`.
Они имеют шесть OCI labels, пользователя `distil3d`, каталоги `/data`,
`/models`, `/prepared`, `/runs`, `/cache` и core CLI по умолчанию.
MoGe устанавливается через временные BuildKit mounts без wheelhouse в
финальных слоях. Публичные `build`, candidate и archive реализованы в `8.5`;
полный clean Linux/no-GPU барьер и archive roundtrip завершены в `8.14`.
После решения пользователя от 5 сентября 2026 года ручные SHA артефактов,
экспорт project wheel на host и нормализация времени ради одинаковых байтов
удалены. Версии, source revisions, base digests и функциональные параметры
сохранены. Повторная серверная сборка трёх prefix, cold imports,
`pip check`, MoGe dry-run и полный `./distil3d test all` прошли 5 сентября.
Отдельный builder рядом с публичными Gen3C dependencies
исправляет только внутренние `WHEEL` tags и пересозданный `RECORD` для Decord
и TE-cu12: официальный payload сохраняется, а исправленные
wheels заменяют его в builder wheelhouse до финального offline replay.

Точный порядок семи production locks, общий `lock_revision`, источники,
команды source builds и граница каждого `COPY` описаны в
`requirements/README.md`. Docker-проверки выполняются только на разрешённом
Linux-сервере. GPU, модели, данные и CP1/CP2/CP4 остаются Stage 11.

## Окружение CPU-проверок

Стадия `cpu-test-deps` содержит core-зависимости, Git и тестовые зависимости
из прежнего lock. Только после неё `cpu-test` устанавливает project wheel,
записывает метки ревизии и копирует тесты. Изменения кода, README или
ревизии поэтому не требуют повторной установки неизменных зависимостей.
Публичный target, состав библиотек и пользователь образа остаются прежними.

Только `cpu-test` ставит `/opt/envs/core/lib` перед унаследованными путями
`LD_LIBRARY_PATH`. Это выбирает уже установленную библиотеку C++ того же
prefix для SQLite/ICU, Torch и PyArrow: запуск отдельной группы больше не
должен зависеть от того, какая библиотека была импортирована первой.
Production targets и закреплённые зависимости этим не меняются; подгрузка
библиотек заранее и пропуски тестов не используются.
