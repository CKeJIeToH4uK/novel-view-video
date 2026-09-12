# Историческая подготовка глубины Waymo

Папка владеет прежним трёхкамерным путём глубины Waymo и центральными
keyset-файлами до чтения RGB:

- `keyset.py` и `spec.py` — ordered keysets, один explicit clip и восемь
  explicit candidate locators из внешних selection-файлов;
- `raster.py` — общий измеренный alpha-zero/20:11/half-pixel raster для
  FRONT, FRONT_LEFT и FRONT_RIGHT;
- `lidar_geometry.py` — преобразование Waymo range image в мировые точки;
- `depth_samples.py`, `depth_collection.py` и `depth.py` — трёхкамерная
  sparse LiDAR-разметка и backend-neutral metric depth;
- `clip.py` — один ограниченный проход, общий LiDAR и временные RGB трёх
  камер;
- `moge.py` и `vggt.py` — разные measured/predicted geometry adapters;
- `metrics.py`, `comparison.py` и `selection.py` — heldout-метрики,
  буквальный MoGe→VGGT и разные решения v1/v2;
- `candidate_record.py` — отдельные исходы методов и общий candidate v2;
- `selection_record.py` — текущий selection v4: выбранный метод либо
  явный `scientific_stop` с оценками;
- `legacy_selection_record.py` — только чтение старых selection v2/v3.
  Перевод старых полей вызывается по версии открытого JSON; новые записи
  этот модуль не используют. Исторические SHA-поля не пересчитываются.

Физические Parquet readers находятся в `inputs/waymo/`. Поддерживаемая
FRONT-only подготовка R4c находится рядом в `preparation/waymo_ddw/` и
переиспользует только совпадающую raster/LiDAR-математику. Трёхкамерное
сравнение остаётся историческим workflow и не становится новым training API.

Новый научный порядок добавляется отдельной версией workflow. В эту папку
не добавляются поиск файлов, предварительный обход данных, общие managers
или скрытые проверки машины.
