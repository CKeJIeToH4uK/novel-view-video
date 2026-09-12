# Данные внешних contract-проверок

Папка хранит малые неизменяемые входы для соседнего contract-теста. Они
воспроизводят реальную внешнюю схему и не требуют датасетов, моделей или
payload-файлов.

- `gaussian_depth_export_v1.json` повторяет полную структуру неизменённого
  producer `3dgs/export/gaussian_depth.py`; frame-файлы намеренно отсутствуют,
  потому что reader декодирует только JSON и пути.
- `euvs_selection_v1.yaml` и `gaussian_selection_v1.yaml` содержат
  намеренно неотсортированные concrete ID и доказывают, что selection reader
  сохраняет исследовательский порядок.
- `euvs_generation_run_v1.json` … `euvs_generation_run_v5.json` — малые
  канонические записи EUVS generation. v4 сохраняет исторический формат без
  LoRA strength, а v5 фиксирует реально применённую strength.
- `gaussian_full_run_v1.json` и `gaussian_dense_run_v2.json` — минимальные
  канонические Gaussian records; handoff-тест переводит v2 в independent и
  overlap варианты.
- `gaussian_workflow_records.json` — семь полных результатов прежнего кода
  до 9A.3: full, два clips, два independent и два overlap. Сняты на
  синтетических входах `test_gaussian_generation_orders.py`; заменён только
  временный корень на `<test-root>`. Проверяют сборку JSON, не заменяют
  независимую математику и PNG-проверки этого теста.
- `waymo_ddw_prepared_v1.json` — канонический ordered `PreparedRecord` с
  двумя sample, 121 timestamp и относительными base/pose/LiDAR locators.

Новый fixture добавляется для внешнего формата или полного ожидаемого результата, остаётся
достаточно малым для обычного Git и не подменяет научные тесты из
`../../unit/` и `../../integration/`.
