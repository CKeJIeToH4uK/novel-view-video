# Модульные проверки

Папка содержит быстрые CPU-проверки чистой математики и небольших предметных
контрактов. Поддерживаемые области сгруппированы по владельцам:

- численная геометрия, nuPlan raster и EUVS планирование; настоящие
  reader/generation/evaluation цепочки находятся в integration;
- лёгкие Gen3C/Cache4D protocol и process seams;
- численная R4c v1/v2 математика: Kendall EDM, LoRA update, depth reducer,
  fit/freeze observer и sparse CP gradient;
- Gen3C context-depth, подменённые LPIPS/DINOv2 границы, реальный SSIM
  oracle и CPU-проверка training-диагностики;
- Waymo входы, трёхкамерные depth v1/v2 и `preparation/waymo_ddw`;
- EUVS support/metrics и evaluation/comparison records;
- runtime; attempt и исторические fit records объединены с внешними
  форматами в `../contract/`, общий exclusive-create helper удалён.

`test_waymo_legacy_{depth_models,depth_selection,lidar_science,raster}.py`
разделяют численную трёхкамерную геометрию, глубину, выбор метода и
сохраняемую историческую раскладку. Настоящий порядок MoGe→VGGT и разные
canary/probe/survey находятся в integration. `test_ddw_fit_science.py` и
сохраняемые `test_waymo_ddw_*` проверяют отдельные исторические mask/fit/audit
сценарии; они не определяют R4c training/evaluation API.

Строгий внешний Waymo keyset проверяется вместе с настоящим job и его
порядком в `../integration/test_waymo_v2_keysets.py`; отдельного unit-теста
простого write/read больше нет.

`test_geometry_input.py` сохраняет численное EUVS rebasing на больших UTM
координатах. Свойства размера posed depth проверяются на настоящем
результате в `test_vggt_geometry.py`; отдельный `test_source_geometry.py`
с проверками простого хранения полей dataclass удалён.

В рабочем срезе Stage 9.1 все численные доказательства Stage 7 находятся в
`tests/unit`, `tests/contract` и `tests/integration` и входят в обычный
CPU-сбор; R4c не помечается `native_api`. Удалённый старый Gen3C test cluster
сюда не возвращается. Новая проверка должна закреплять значение или внешний
контракт, а не mock choreography, точный текст всех ошибок или лишний
filesystem preflight.
