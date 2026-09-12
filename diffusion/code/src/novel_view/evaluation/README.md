# Оценка результатов

Пакет связывает готовый результат generation с предметной оценкой. Общая
численная математика находится в `novel_view.metrics`, а здесь остаются
EUVS-вход, маски, support, композиция model-метрик и сохраняемый результат.

Подпакет `euvs/` содержит evaluation и comparison одного предметного формата:

- `spec.py` строго разбирает `euvs_evaluation/v1` и принимает либо точную
  generation attempt, либо явный список legacy records;
- `samples.py` открывает ровно указанный `run.json` и его записанный output,
  сохраняя target/output/source order из record;
- `masks.py` читает либо создаёт прямой PNG-cache
  `/cache/euvs/masks/<recipe-id>/native/<token>.png`;
- `support_projection.py` реализует `four-neighbour-zbuffer/v1`, а
  `support_views.py` строит target-static и source-aware дорожки;
- `execute.py` считает CPU PSNR/SSIM и один раз вызывает частный
  `_worker.py` для LPIPS/DINOv2;
- `record.py` строго читает и прямо пишет
  `novel-view/euvs-metric-result/v1`;
- `benchmark.py` читает две точные матрицы metric/run records, сохраняет
  paired compatibility и equal-camera/pair/location математику и прямо пишет
  три CSV без запуска evaluation. Сводку по камерам считает напрямую через
  `metrics.aggregate.summarize_cameras`; record хранит покадровые значения,
  без отдельного summary-класса и вычисляемых свойств.

`workflows.euvs_evaluation.run_v1()` последовательно исполняет selection и
пишет только `attempt/metrics/<pair-name>.json`. Он не сканирует каталоги,
не ищет готовый результат, не продолжает после ошибки и не имеет
`check-only`. Старая одиночная/campaign/benchmark Python-обвязка удалена;
`workflows.euvs_comparison.run_v1()` является отдельным CPU workflow над уже
готовыми evaluation attempts.

Подпакет `ddw/` содержит один matched `ddw_evaluation/v1`:

- `spec.py` связывает PreparedRecord, validation split и явно названные
  v1/v2 checkpoint JSON/PT на одном фактическом training step;
- `samples.py` сохраняет validation order и читает соответствующий raw
  Waymo FRONT target, measured cameras и sparse LiDAR;
- `sampling.py` одной CP4-сессией строит base/v1/v2 latents при одинаковых
  seed и schedule, полностью заменяя adapter между вариантами;
- `_decode_worker.py` одной загрузкой VAE и одной загрузкой MoGe получает
  RGB и relative depth всех трёх вариантов;
- `metrics.py` подгоняет масштаб по nonheldout LiDAR, оценивает heldout
  depth и RGB и пишет четырёхпанельное MP4;
- `execute.py` освобождает временные arrays после каждого sample, а
  `record.py` сохраняет один строгий итоговый JSON с item и aggregate rows.

`workflows.ddw_evaluation.run_v1()` исполняет этот порядок без поиска
`latest`, сканирования каталога, старого request record и final-only gate.
Постоянными результатами являются только JSON и MP4; latents, decoded RGB и
dense depth остаются временными.

При добавлении новой оценки сначала создавайте конкретный предметный модуль.
Не добавляйте общий evaluator, registry, filesystem preflight, staging,
atomic/no-overwrite или worker-output validator.
