# Закрытый handoff для Stage 11

Этот список сохраняет состав прежнего четырёхкарточного контура, без
закрытых Waymo ID, серверных путей или результатов. С 12 сентября
целевая проверка переведена на одну A100;
[текущее состояние](../../docs/a100-operator.md). Код `a100-1` пока не реализован; `a100-4` не запускать
на одной карте. Реальная кампания требует отдельного разрешения.

Для однокарточной подготовки из этого списка нужны `waymo-one-item.yaml`,
его raw Parquet и веса Gen3C/tokenizer/VAE/T5/MoGe. DDW85 pack/split,
legacy checkpoint map и восемь validation segments для training/evaluation
не требуются новой кампании. Её реальные входы оператор готовит сам.
Разделы ниже остаются справочником для отдельного CP4 training, не новым
обязательством его запускать. Прежний R4c принят пользователем без нового
GPU-доказательства; экспериментальный v2 GPU/A/B отложен.

## Что передать

Tracked из одного проверенного source revision:

- Docker candidate, корневой `./distil3d`, profile example;
- `jobs/waymo/r4c-lora-edm/` и
  `jobs/waymo/r4c-lora-lidar-depth/` как training templates;
- `jobs/examples/r4c-select-v1/`, `r4c-select-v2/` и
  `r4c-ddw-evaluation/` как формы будущих run-specific jobs;
- recipes `gen3c-training/*` и `ddw-evaluation/r4c-lidar-depth.yaml`.

Закрытым bundle вне Git:

- `selections/local/stage11/waymo-one-item.yaml` для отдельной проверки
  `ddw_preparation/v1`;
- `selections/local/r4c-ddw85-split.yaml`;
- `selections/local/r4c-ddw85-legacy-map.json` только для однократного
  чтения старого untagged v1 checkpoint;
- готовый `waymo-ddw/r4c-ddw-prepared/prepared.json` и все 85 названных им
  base/pose/LiDAR artifacts;
- raw Waymo validation Parquet для восьми segment из закрытого split;
- Gen3C base checkpoint, Cosmos tokenizer/VAE, T5-11B encoder и MoGe-ViT-L
  checkpoint по относительным путям tracked recipes;
- исключённый из Git `profiles/local/external-a100-4/.env` с корнями узла.

Preparation одного sample из 121 кадра пишет отдельный
`/prepared/waymo-ddw/stage11-waymo-one-item/prepared.json`. Она не заменяет
готовый полный `/prepared/waymo-ddw/r4c-ddw-prepared/prepared.json`, который
остаётся входом v1/v2 training вместе со всеми 85 названными artifacts.

## Локальные jobs

`accept a100-4` сама создаёт один campaign-local набор под `jobs/local/`:
копирует tracked формы, переименовывает только preparation одного
121-кадрового sample в
`stage11-waymo-one-item`, меняет training split на
`selections/local/r4c-ddw85-split.yaml` и после producer runs буквально
подставляет точные attempt/checkpoint records в selection/evaluation.
Оператор не редактирует эти jobs вручную и не угадывает attempt ID.

## Порядок

1. `doctor` и `plan` для всех локальных jobs.
2. CP4 model-ready smoke.
3. V1 fresh, затем resume и v1 selection.
4. V2 fresh, затем resume и v2 selection.
5. Проверка равного фактического шага и matched base/v1/v2 evaluation.
6. Возврат ограниченного отчёта со ссылками на полные логи и attempts,
   без ручного изменения image и передачи тяжёлого дерева результатов;
   правила объёма — в [приёмочном handoff](../acceptance/STAGE11_HANDOFF.md).

Stage 7 доказал CPU/fake контракты. CP4 collectives, реальные training
fresh/resume и качество A/B новым кандидатом на GPU не проверены.
Решение пользователя не превращает эти отсутствующие результаты в `succeeded`.
