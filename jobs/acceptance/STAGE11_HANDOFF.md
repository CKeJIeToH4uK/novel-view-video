# Внешние входы EUVS, Gaussian и исторического Waymo для Stage 11

Этот список описывает роли данных без серверных путей, закрытых ID и весов.
Финальная кампания начинается только после отдельного разрешения
пользователя. Основная Waymo/R4c часть передаётся по
[`jobs/waymo/STAGE11_HANDOFF.md`](../waymo/STAGE11_HANDOFF.md).

С 12 сентября цель — одна A100 80 ГБ: 18 запусков и восемь явных
исключений; [текущее состояние режима](../../docs/a100-operator.md).
Код пока поддерживает только `a100-4`; упоминания этой команды и её 26
случаев ниже описывают прежний реализованный режим. CP2 и training v1/v2 с
зависимым DDW evaluation исключены из нового прогона. Перечисленные здесь
EUVS/Gaussian/legacy входы и TE-проверка остаются нужны однокарточному режиму.

## Кандидат и исходный код

`candidate.json` передаётся только вместе с соседним `images.tar` и чистым
checkout той revision, которая записана в поле `source_revision`. Нельзя
соединять старый архив с текущим `main`: если после его сборки изменился
исполняемый код, сначала собирается новый кандидат. Ручное редактирование
кода на операторском узле не требуется и сделает bootstrap невалидным.

## TE recompute перед обучением

В той же команде `accept a100-4`, после preparation и до R4c v1 fresh,
автоматически выполняется один
`test_model_api.py::test_te_recompute_preserves_result_and_gradients`.
Он проверяет настоящий TE checkpoint, результат и градиенты, включая
повторное вычисление. Используется Gen3C prefix точного core-образа и
первая GPU профиля; веса и датасет этому тесту не нужны. Tests и
`pyproject.toml` подключаются только для чтения, отдельную команду
оператор не вводит.

Ненулевой pytest exit или OOM означает ошибку; пропуск теста не считается
приёмкой. Ошибка пропускает обе версии обучения и их зависимые случаи,
но оставляет независимый исторический Waymo. Сигнал останавливает именно
текущий контейнер по его точной ссылке. В отчёте остаются JUnit, лог и
состояние контейнера; training record не создаётся (`recorded_state=absent`).
До настоящего GPU-прогона существующая строка AC остаётся `pending`.

Итого одна кампания содержит 26 случаев и 29 строк с bootstrap,
foundation и closure. Взаимный порядок исходных 19 случаев не изменён.

## EUVS

- `euvs_dataset`: совместимый EUVS/nuPlan root для reader `euvs_nuplan`;
- `euvs_one_pair_selection`: одна разрешённая pair в schema v1; одна и та же
  selection используется source views, generation и evaluation;
- `euvs_tuned_evaluation_reference`: готовый read-only результат
  `euvs_evaluation/v1` с теми же pair, sampling order и metric format, что у
  base evaluation текущего кандидата.

Tracked `selections/euvs/one-pair.yaml` проверяет форму, но перед кампанией
заменяется точной закрытой
`selections/local/stage11/euvs-one-pair.yaml` с реальными устойчивыми
токенами. Внешний tuned result размещается в runs-relative роли
`external-euvs-tuned-evaluation/reference/attempts/accepted`. После
source views точный attempt подставляется во все generation jobs; после CP1
autoregressive generation его точный attempt подставляется в evaluation;
после evaluation точный attempt и внешний tuned reference подставляются в
comparison. Если tuned reference не предоставлен, comparison получает
`skipped:missing-external-input`, а полная волна не считается принятой.

## Gaussian

- `gaussian_depth_export`: один разрешённый `gaussian_depth_export/v1` root
  с `export.json` и всеми названными им RGB/depth/mask файлами;
- `gaussian_dense_camera_table`: совместимые reconstruction transforms и
  intrinsics для того же scene;
- `gaussian_selections`: campaign-local full, one-clip, dense-independent,
  dense-overlap21 и dense-handoff selections для того же producer.

Команда ожидает `export.json`, `transforms.json` и `intrinsics.json` под
`/data/gaussian/stage11/`; названные export файлы остаются рядом согласно их
контракту. Закрытые выборки имеют точные роли:

- `selections/local/stage11/gaussian-full-sequence.yaml`;
- `selections/local/stage11/gaussian-independent-clip.yaml`;
- `selections/local/stage11/gaussian-dense-independent.yaml`;
- `selections/local/stage11/gaussian-dense-overlap21.yaml`;
- `selections/local/stage11/gaussian-dense-handoff.yaml`.

`replace-export` меняется только в campaign-local копиях jobs/selections.
После v3 и v4 их точные attempts подставляются в v5; поиск последнего run не
используется. Tracked selection используют открытый `scene_id: 0` только как
проверяемую форму и не утверждают существование реальной scene.

## Исторический Waymo: пять запусков и один выбор глубины

Это дополнение той же команды `accept a100-4`, не отдельная кампания.
После прежних 19 случаев, не меняя их взаимного порядка, выполняются
сравнение глубины, независимый CPU gate и четыре DDW-сценария; затем
закрывается общий отчёт. Все аппаратные результаты до реального Stage 11
остаются `pending`.

Существующие формы из `jobs/legacy/waymo/` копируются в
`jobs/local/<campaign>/`: `waymo-depth-candidates`,
`waymo-depth-selection-v2`, `waymo-ddw-canary-v1`, `waymo-ddw-canary-v2`,
`waymo-ddw-probe`, `waymo-ddw-survey`. Их recipes сохраняются без изменений;
новых дублирующих YAML под `jobs/acceptance/` нет.

### Что готовит оператор

Все имена ниже — роли файлов, а не закрытые идентификаторы. Пять YAML
передаются вне Git под `selections/local/stage11/legacy/`:

| Файл | Назначение |
| --- | --- |
| `waymo-depth-clip.yaml` | Один разрешённый исторический клип для MoGe→VGGT |
| `waymo-depth-choice-8.yaml` | Полный прежний упорядоченный набор восьми разных candidate reports |
| `waymo-ddw-exposed-debug.yaml` | Прежний exposed-debug клип engineering canary v1 |
| `waymo-ddw-canary.yaml` | Прежний canary-v2 клип; probe использует тот же вход, но другой порядок |
| `waymo-ddw-survey.yaml` | Явно выбранный исторический survey-клип |

Четыре clip-файла имеют форму
`selections/legacy/waymo/example-depth-clip.yaml`: точные `split_id`,
`official_partition`, `segment_id`, `start_frame_index` и все 121 реальный
timestamp. Синтетические значения примера не являются готовыми данными.
Предыдущий закрытый canary-v2 вход сохранён в
`selections/local/stage11/legacy/waymo_ddw_v2_preregistration.json`;
он помогает оператору восстановить YAML, но runtime его не читает.

Форма списка reports —
`selections/legacy/waymo/example-depth-choice-8.yaml`. Каждый из восьми
`candidate_results` указывает точный путь под `/runs` к разрешённому
`gen3c-waymo-depth-candidate/v2` JSON; сохраняются прежние восемь ключей,
их порядок и результаты обоих методов. Передаются сами JSON по названным
путям, а не только список. Новый одиночный comparison не заменяет этот
набор, не подставляется в него автоматически и не запускает восемь новых
опытов. Старый полный split, raw 718 items и DDW85 для этих шести задач не
нужны.

Raw Waymo Parquet выбранных клипов размещаются под `/data/waymo` по
исходной раскладке reader. Нужны `vehicle_pose`, `camera_image`, `lidar`,
`lidar_camera_projection`, `lidar_pose`, `camera_calibration` и
`lidar_calibration`. Reader сохраняет пять camera IDs, все пять LiDAR и
оба returns, включая TOP pixel-pose. Нельзя удалить остальные камеры из
Parquet только потому, что downstream использует три камеры или FRONT.

Веса подключаются по нынешним recipes:
`/models/moge-vitl/model.pt` и `/models/vggt-omega/model.pt`.
Пять аппаратных jobs используют `moge/inference_cp1`: одна A100 за раз,
не `training_cp4` и не `cpu_test`. Сравнение последовательно использует
`GEN3C_PYTHON` для MoGe и `VGGT_PYTHON` для VGGT. Четырём DDW нужен
установленный Gen3C/Cache4D для геометрии, но не generation checkpoint.
CPU selection работает в `core` с существующим preset `cpu_test`.

### Что именно должно выполниться

| Порядок | Полный объём и результат |
| --- | --- |
| `waymo_depth_comparison/v1` | MoGe→VGGT по FRONT, FRONT_LEFT, FRONT_RIGHT; каждая камера содержит 121 кадр. `result.json`: `gen3c-waymo-depth-candidate/v2`, отдельные исходы методов и camera metrics; завершённый record — код 0 |
| `waymo_depth_selection/v2` | Восемь reports → нормализованная общая LiDAR evidence → gate v2. `depth-selection.json`: `gen3c-waymo-depth-selection/v4`; `selected` и `scientific_stop` оба возвращают 0 |
| `waymo_ddw_canary/v1` | FRONT, 121 кадр, d1..4 × (-1,+1), восемь вариантов. `result.json`: `gen3c-waymo-ddw-engineering-canary/v2`, `passed`/код 0; отказ headroom остаётся ошибкой, структурированного STOP у v1 нет |
| `waymo_ddw_canary/v2` | FRONT, 121 кадр, d2..4 × (-1,+1), шесть вариантов. `gen3c-waymo-ddw-v2-canary/v2`: `passed`/0 либо первый `scientific_stop`/2, до reference неуспешного варианта |
| `waymo_ddw_probe/v1` | FRONT, 121 кадр, d3..4 × (-1,+1), все четыре результата. `gen3c-waymo-ddw-v3-probe/v2`: `passed`/0 либо `scientific_stop`/2 после полной оси |
| `waymo_ddw_survey/v1` | FRONT, 121 кадр, d1..4 × (-1,+1), все восемь результатов. `gen3c-waymo-ddw-survey/v2`: `passed`/0 либо `scientific_stop`/2 после полной оси |

Три камеры относятся именно к comparison. DDW использует FRONT из общего
трёхкамерного входа, не три одинаковых DDW-запуска. Каждый DDW-вариант
раздельно измеряет локальный A→B→A′ и глобальный Cache4D reference по всем
121 source/query кадрам. Probe и survey продолжают ось после failed
headroom, но пропускают reference такого варианта. Их нельзя объединять с
останавливающимся canary или сокращать оси ради успешного результата.

Только выбор `moge-v1-lidar-scale` допускает четыре DDW-задачи. `accept`
подставляет точный runs-relative путь текущего
`depth-selection.json` в их личные копии и только затем вызывает их
`plan` → `doctor` → `run`. `scientific_stop`, выбранный VGGT или ошибка gate
дают явный пропуск зависимых DDW, а не выбор последнего подходящего run.
Признание корректно записанного научного STOP не меняет настоящие
container exit/attempt state и не принимает произвольный код 2, OOM или
ошибку модели за успех.

### Объём передачи и возврат

Дополнительный вычислительный объём — пять отдельных аппаратных запусков
и один CPU gate. При прохождении всех вариантов это 3 MoGe + 3 VGGT
камерных расчёта сравнения, ещё 4 FRONT MoGe расчёта, до 26 DDW-вариантов
(52 локальных warp-прохода и 26 global references). Это не оценка времени:
реальная длительность и память фиксируются на узле. Raw клип для v2/probe
общий; совпавшие разрешённые исходные файлы не нужно передавать дважды.

Для ориентира только три временных uint8 RGB-потока
121 × 704 × 1280 × 3 занимают около 0,914 GiB на один клип. Это не полный
размер scratch: глубины, маски, model exchange, raw Parquet и веса
добавляются отдельно. Восемь входных reports — JSON, а не новые dense arrays.

Возвращаются точные небольшие records, планы, summary, состояние контейнера
и ссылки на полные логи. Для каждого из шести новых случаев копируется
его итоговый JSON как `records/<case>/record.json` в прежнем ограниченном
отчёте: до 4 MiB на JSON внутри общего лимита 25 MiB на случай. Полные
NPY, checkpoints, данные и веса туда не входят. Исходные
`result.json`/`depth-selection.json` и журналы остаются в соответствующей
attempt под `/runs`. Отрицательный научный verdict сохраняется явно.

## Что не передаётся через Git

Датасеты, producer export, camera tables, реальные selections, модели,
веса, checkpoints, профили узла и результаты запусков остаются вне Git и
образов. В tracked-файлах не должно появляться абсолютных host paths.
