# Waymo DDW selections

Файл версии 1 хранит ordered список конкретных Waymo clips: устойчивый
`sample_id`, исходные `partition`, `segment_id`, `start_frame_index` и
заданное DDW-смещение. Пути машины, готовые NPY/PT, timestamps, fit scores и
`global_fit_index` сюда не входят.

`example-front.yaml` — маленький синтетический пример preparation selection.
`example-r4c-split.yaml` показывает отдельное ordered-разделение уже
подготовленных `sample_id` на training и validation. Полная рабочая DDW85
selection, split и одноразовая карта старых fit indices передаются отдельно
и не отслеживаются Git.
