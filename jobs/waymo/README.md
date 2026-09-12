# Задачи Waymo

Папка содержит переносимые job для поддерживаемых Waymo-сценариев.
`r4c-ddw-prepared/` связывает публичный пример selection с
`ddw_preparation/v1`; `r4c-lora-edm/` запускает baseline
`gen3c_training/v1`, а `r4c-lora-lidar-depth/` запускает экспериментальный
`/v2` на том же tracked split. Оба маршрута реализованы и проверены на CPU;
реальные CP4/A100 и matched-step оценка качества остаются Stage 11.

`STAGE11_HANDOFF.md` перечисляет tracked и закрытую части будущей внешней
кампании. Он не содержит server paths или закрытых Waymo ID.

Новая подпапка появляется только вместе с работающим versioned workflow и
собственным `run.yaml`; пути конкретной машины сюда не добавляются.
