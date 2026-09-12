# Проверки внешних и архитектурных контрактов

Папка содержит короткие проверки устойчивых границ установленного пакета:
форматов входов и записей, направления production-зависимостей, холодных
импортов и внешней job/plan-оболочки.

- `test_gaussian_export.py` закрепляет полный producer-v1 JSON, порядок
  кадров, все декодируемые поля и относительные пути без открытия frame
  payload.
- `test_job_planning.py` закрепляет внешнюю job/recipe оболочку,
  config-only plan и обязательный `inference_cp1` для EUVS source views.
- `test_input_selections.py` закрепляет внешние EUVS/Gaussian YAML,
  точный порядок pair/token/clip/dense ID и лёгкую границу импорта.
- Waymo DDW job/recipe/selection и `PreparedRecord` проверяются вместе
  с реальной подготовкой в `../integration/test_ddw_preparation_record.py`;
  прежний постоянный fixture остаётся здесь в `fixtures/`.
- `test_gen3c_training_specs.py` разрешает оба tracked R4c training job в
  `core`/`training_cp4`, проверяет раздельные строгие recipes и холодный v1
  planning path без depth/Torch/Cosmos.
- `test_r4c_data.py` закрепляет внешний split и legacy map,
  ordered `PreparedRecord` join, segment-disjoint identity, буквальный PCG64
  epoch order и normalized next-item cursor без открытия artifacts.
- Matched ранний v1/v2 step, heldout depth/RGB, итоговый record и
  четырёхпанельное MP4 объединены в `../integration/test_ddw_evaluation.py`.
- `test_euvs_records.py` закрепляет EUVS `run.json` v1–v5,
  явное legacy-чтение, научную identity и прямую write/read границу без
  предварительной проверки RGB или файловой системы.
- `test_ddw_fit_records.py` сохраняет текущий fit v2, отдельное чтение
  rejected v1 для audit и согласованность внешних научных полей. Порядок
  неполной collection и поля audit record проверяются вместе с настоящими
  workflow в `../unit/test_waymo_ddw_fit_workflows.py`.
- `test_operational_records.py` проверяет один настоящий writer/read
  attempt v1, неизменность записи при чтении и восстановление сохранённой job.
- `test_candidate_transfer.py` проверяет внешний candidate v1 через
  настоящий CLI, точные image identities и одну ошибку входного inventory.
- `test_legacy_acceptance.py` читает настоящие historical outcome records,
  отличает scientific STOP от процессной ошибки и допускает DDW только по
  полному ordered gate с выбранным MoGe. Проверяет явный doctor и prefix
  routing без загрузки моделей; это не замена приёмке на A100.
- `test_architecture_boundaries.py` сохраняет прежние границы imports/cold
  geometry/runs и дополнительно удерживает Stage 8 на host-стороне: Python
  не запускает Docker, scientific owners не читают candidate/profile,
  launcher не вызывает private worker, а Docker context исключает локальные
  и тяжёлые деревья. Вызов tests разрешён только явному host test-маршруту
  и одному точному TE node в accept; для production Python запрет полный.
- `fixtures/` хранит малые неизменяемые документы внешних форматов; у неё
  есть собственный паспорт и нет вычисленных данных.

Новый файл добавляется сюда только для публичной границы формата или
архитектуры. Научную математику, workflow и model-поведение следует проверять
в соответствующей группе, не перенося их в contract-набор.
