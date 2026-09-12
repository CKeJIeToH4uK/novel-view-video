# Подготовка одного Waymo DDW примера

`run.yaml` — маленький публичный пример полного `ddw_preparation/v1`. Он
использует `selections/waymo/example-front.yaml` и записывает результат в
`/prepared/waymo-ddw/r4c-ddw-prepared/` внутри контейнера.

Рабочая DDW85 selection остаётся отдельным закрытым входом; для неё создают
локальную job с тем же внешним контрактом, не меняя этот пример.
