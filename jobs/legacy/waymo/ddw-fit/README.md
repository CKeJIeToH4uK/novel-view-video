# Назначенные DDW-варианты

`run.yaml` запускает исторический `waymo_ddw_fit/v1` через `./distil3d`.
Скопируйте выборку в `selections/local/`: укажите готовый полный keyset под
`/runs` и нужные индексы. Варианты назначаются на полном keyset, затем
указанные строки выполняются последовательно. Смена поднабора не меняет
ранее назначенные варианты.

Каждый item пишет `items/clip-NNNN/clip.json` с accepted/rejected. Только
accepted получает повторный render и два condition NPY рядом с record.
Нужны raw Parquet выбранных clips и MoGe weights; generation checkpoint
не нужен. Не требуются старые очереди, worker IDs, SHA или depth-selection.
Ошибка модели прерывает запуск; готовые item records остаются. Повторный
поднабор выбирает оператор в новой job, автоматического retry/reuse нет.

Новый научный порядок оформляется отдельной версией, не скрытым флагом.
