# Проверки взаимодействия компонентов

Папка закрепляет реальные стыки readers, файлов, процессов и явных научных
порядков на малых синтетических входах. Дорогие модели и CUDA-границы в
обычном CPU-наборе подменяются. Generated-Parquet, serializers, значения
тензоров и решения проверяются непосредственно.

Основные группы:

- `test_nuplan_euvs_reader.py` — настоящий SQLite/CSV/RGB стык, порядок
  запроса, независимые часы, разные физические камеры и source-only чтение;
- `test_euvs_generation_orders.py` — source-only и оба Gen3C seed-порядка;
  `test_gaussian_generation_orders.py` — все пять реальных Gaussian
  workflows до records и PNG с подменой только дорогой генерации;
- `test_euvs_evaluation_chain.py` — чтение exact output/source slots,
  target-only RGB, saved VGGT, две support-маски и численные метрики;
- Waymo v2 reader и строгий training/validation keyset; `test_ddw_{warp_exchange,bake_artifacts,preparation_record}.py`
  связывают forward-warp, prompt/VAE, настоящие артефакты и generated-Parquet;
- `test_r4c_{batches,epochs,state,routes}.py` — оба versioned training
  порядка, checkpoint/resume и явный выбор; `test_ddw_evaluation.py` —
  matched LoRA/RNG, heldout/RGB и настоящий JSON/MP4;
- `test_ddw_legacy_orders.py`, `test_waymo_depth_comparison.py` и
  `test_waymo_diagnostic_science.py` — различные исторические порядки и
  явная операторская диагностика;
- Gen3C diffusion windows/session и model-ready диагностика, MoGe worker;
- process lifecycle и настоящий CPU torchrun на Linux;
- отдельные явные Waymo diagnostics и перенесённые исторические DDW
  сценарии с подменой только дорогих модельных границ. Прежний tests/data
  и его операторы удалены, исполняемый код принадлежит предметным пакетам.

Процессные проверки сохраняют отдельно явную остановку двух rank и смерть
их управляющего процесса. Runner проверяет две fresh attempts и отдельный
resume из полного saved job. SIGTERM всего дерева дополнительно проходит
в настоящем неизменном `tests/launcher/test_lifecycle.sh`; повторная
Python-проверка того же дерева удалена.

## API установленных модельных окружений

`test_model_api.py` содержит 10 именованных no-GPU `native_api` случаев
и одну отдельную проверку TE recompute, требующую CUDA. Обычный
CPU-набор исключает этот marker, не модельные вычисления вообще. Тяжёлые
imports выполняются внутри выбранных tests; ошибка импорта не становится
`importorskip`. Ни один R4c-файл не имеет этого marker.

| Суффикс nodeid после `test_model_api.py::` | Доказательство и prefix |
|---|---|
| `test_cache4d_buffers_and_chunked_unprojection` | Gen3C: настоящий Cache4D, две evidence layers, chunked/direct unprojection |
| `test_cache4d_identity_global_windows_and_empty_support` | Gen3C: identity pixels, global indices двух окон, empty support |
| `test_warp_identity_and_subpixel_splat` | Gen3C: настоящий identity и subpixel splat |
| `test_warp_collision_depth_weights_and_batch_maximum` | Gen3C: самостоятельные depth-weighted collision/batch-max формулы |
| `test_warp_sanitization_and_both_masks` | Gen3C: настоящее depth filtering и обе source masks |
| `test_vggt_official_loader` | Запущен из Gen3C pytest, сам вызывает `/opt/envs/vggt/bin/python`: две balanced сетки побитово совпадают с официальным loader |
| `test_lora_zero_delta_and_nonzero_forward` | Gen3C: настоящий маленький LoRALinearLayer; нулевая добавка и ненулевой forward без весов большой модели |
| `test_grounding_sam2_signatures` | Gen3C: настоящие GroundingDINO/SAM2 imports и аргументы |
| `test_lpips_dinov2_signatures` | Gen3C: настоящие LPIPS/DINOv2 imports и сигнатуры используемых вызовов |
| `test_moge_v1_signature` | Gen3C в образе moge: настоящий MoGe-v1 import и аргументы |

Последние три случая не загружают веса и не доказывают модельный forward
или качество. CPU-двойники этих моделей отдельно проверяют наши численные
преобразования. VGGT не требует установки pytest в свой production-prefix:
дочерний процесс исполняет только helper этого же учтённого файла.

Предварительный 9.11 прогон всех 9 случаев прошёл на сохранённом образе
Stage 8 (без GPU/сети/весов). Это не заменяет обязательный повтор 9.17 на
окончательном кандидате. Прежние смешанные SAM2/VGGT файлы удалены после
зелёных CPU replacements в unit; два CPU forward-warp случая сохранены.

### Обязательные команды 9.17

Выполняются **только на разрешённом Linux-сервере**, из корня его checkout;
на локальной машине Булата Docker не запускается. Образы предварительно
собраны из принятого кандидата. Подключаются только тесты и их настройки;
`novel_view` берётся из установленного wheel, исходники не монтируются.

```bash
docker run --rm --network none \
  --mount "type=bind,src=$(pwd)/diffusion/code/tests,dst=/tests,readonly" \
  --mount "type=bind,src=$(pwd)/diffusion/code/pyproject.toml,dst=/pyproject.toml,readonly" \
  --entrypoint /opt/envs/gen3c/bin/python distil3d:cuda124-sm80-core \
  -m pytest -c /pyproject.toml -p no:cacheprovider -q \
  /tests/integration/test_model_api.py::test_cache4d_buffers_and_chunked_unprojection \
  /tests/integration/test_model_api.py::test_cache4d_identity_global_windows_and_empty_support \
  /tests/integration/test_model_api.py::test_warp_identity_and_subpixel_splat \
  /tests/integration/test_model_api.py::test_warp_collision_depth_weights_and_batch_maximum \
  /tests/integration/test_model_api.py::test_warp_sanitization_and_both_masks \
  /tests/integration/test_model_api.py::test_vggt_official_loader \
  /tests/integration/test_model_api.py::test_lora_zero_delta_and_nonzero_forward \
  /tests/integration/test_model_api.py::test_grounding_sam2_signatures \
  /tests/integration/test_model_api.py::test_lpips_dinov2_signatures

docker run --rm --network none \
  --mount "type=bind,src=$(pwd)/diffusion/code/tests,dst=/tests,readonly" \
  --mount "type=bind,src=$(pwd)/diffusion/code/pyproject.toml,dst=/pyproject.toml,readonly" \
  --entrypoint /opt/envs/gen3c/bin/python distil3d:cuda124-sm80-moge \
  -m pytest -c /pyproject.toml -p no:cacheprovider -q \
  /tests/integration/test_model_api.py::test_moge_v1_signature
```

### Отдельная проверка TE recompute с CUDA

`test_model_api.py::test_te_recompute_preserves_result_and_gradients`
восстанавливает прежнюю численную проверку двух маленьких блоков через
настоящий production-вызов `_apply_activation_recompute`. Сохраняются
Dropout0.25/seed17/scale0.75, forward, градиенты входа, вложенного offset и
всех параметров, а также фактическое повторное вычисление при backward.
Веса большой модели не нужны. У закреплённого Transformer Engine 1.12.0
сохранение RNG требует CUDA даже при CPU tensors этого теста.

Эта проверка не входит в десять обязательных no-GPU команд 9.17. Общий
CPU-набор исключает её по marker `native_api`. Явный запуск выбранного
узла без CUDA естественно завершится ошибкой Transformer Engine, не
пропуском с успешным кодом pytest. Настоящий результат остаётся `pending`
до разрешённого окна Stage11.
Точная внутренняя команда в Gen3C prefix образа core с доступной CUDA:

```bash
/opt/envs/gen3c/bin/python -m pytest -c /pyproject.toml -p no:cacheprovider -q \
  /tests/integration/test_model_api.py::test_te_recompute_preserves_result_and_gradients
```

Тесты и pyproject подключаются read-only как в командах выше; `novel_view`
берётся из установленного wheel. Отдельные workflow, образ и кампания не
создаются. `./distil3d accept a100-4` автоматически выполняет этот случай
перед R4c training; обе версии обучения зависят от его успеха. Запускатель
и его отказ/остановка проверены с подменой Docker, но настоящий CUDA-результат
по-прежнему ожидает Stage 11.

### Необязательная исследовательская проверка DA3

`test_da3_native_api.py` содержит один отдельный `native_api` случай:
настоящий DA3 InputProcessor сравнивается с нашей сеткой и intrinsics для
разрешений 504, 896 и 1280. Его CPU-математика по-прежнему входит в
`../unit/test_da3_nested.py`; она не исключается из общего набора.

DA3 остаётся экспериментальным и не входит в десять обязательных случаев
`core+moge` для 9.17. Отдельную проверку запускают из `diffusion/code` в уже
подготовленном исследовательском окружении с установленными DA3 и
`novel_view`:

```bash
python -m pytest -q -m native_api \
  tests/integration/test_da3_native_api.py::test_sizes_and_intrinsics_match_all_planned_resolutions
```

Этот вызов не устанавливает зависимости и не создаёт новый образ.
Отсутствие DA3 завершает выбранную проверку ошибкой, а не пропуском.
Обычные `./distil3d test integration` и `./distil3d test all` исключают этот
marker при сборе тестов без импорта DA3. Группа integration также включает
корневые проверки запускателя; настоящий shell lifecycle выполняется
дополнительно только в all.

Новый файл здесь должен закреплять внешний результат взаимодействия двух
реальных владельцев. Чистую математику помещать в unit, внешнюю схему — в
contract. Подмена дорогого устройства не должна подменять проверяемую
формулу ожидаемым ответом.
