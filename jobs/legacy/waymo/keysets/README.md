# Исторические центральные ключи Waymo

`run.yaml` выполняет `waymo_keysets/v1` в образе `core`, режим `cpu_test`.
Оператор задаёт явный закрытый список в
`selections/local/waymo-segments-v1.yaml` и подключает Waymo v2 как набор
`waymo` через профиль машины. План задания не открывает список или данные.

Порядок: fit → dev → первые два debug ID; каждый сегмент даёт центральное
окно из 121 кадра с `start=(len(timeline)-121)//2`. Все три роли относятся
к official `training`; dev не превращается в official `validation`.
Индекс читает только ключи, RGB/depth не открываются. Уже прочитанные
fit timelines переиспользуются для debug.

В текущую attempt, под `keysets/`, напрямую записываются прежние файлы:
`fit-central-v1.json`, `dev-central-v1.json`, `vertical-debug2-v1.json`
со схемой `gen3c-waymo-keyset/v1`. Следующие задания получают точные пути
к этим файлам; поиска последнего результата нет.

Прежний полный `waymo-segment-split/v1` читается без конверсии, включая
описания dataset/ordering/assignment и дополнительные validation-роли.
В этом workflow используются только fit/dev/debug_subset. Минимальный
синтетический пример той же формы:

```yaml
schema_version: waymo-segment-split/v1
split_id: waymo-ddw-lora-v1
splits:
  fit:
    source_partition: training
    segment_ids: [example-fit-b, example-fit-a, example-fit-c]
  dev:
    source_partition: training
    segment_ids: [example-dev]
  debug_subset:
    source_partition: training
    subset_of: fit
    segment_ids: [example-fit-b, example-fit-a, example-fit-c]
```

Если указан `count`, он должен совпадать с длиной своего списка. Состав
сегментов задаётся явно; закрытые IDs, новые наборы данных и варианты
математики в эту папку не добавляются.
