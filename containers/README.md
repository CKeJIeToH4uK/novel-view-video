# Контейнеры

Папка хранит воспроизводимые основы образов проекта и внутренние средства
контейнерного запуска. Пользовательский интерфейс сюда не переносится:
внешней командой остаётся корневой `./distil3d`.

## Содержимое

- `diffusion/` — один многостадийный Dockerfile, машинный ordered inventory
  семи production locks и точные зависимости компонента `diffusion/code`;
- `launcher/` — внутренние Bash-функции корневого host launcher: profile,
  явный `doctor host|job|waymo-ingress|waymo-raster`, сборка production/test
  образов, команды `test`,
  config-only `plan`, научные и synthetic `run/resume`, read-only `status/logs` и явные
  `stop/remove`, а также одна фиксированная кампания `accept a100-4`.

Новый образ или вариант добавляется только после отдельного прототипа и с
обновлением корневого `ARCHITECTURE.md`.
