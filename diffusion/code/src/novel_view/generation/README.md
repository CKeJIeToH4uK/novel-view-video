# Подготовка генерации

Пакет связывает физические данные benchmark с уже готовой source-геометрией
и выполняет явные адаптеры конкретной генеративной модели.

- `euvs/source_views.py` владеет подготовкой переиспользуемых EUVS source
  views: сохраняет первое появление каждого exact ordered tuple, читает и
  растрирует только source RGB, строит source-neutral VGGT request и
  связывает готовый model-owned record по полной ordered identity. GPU,
  campaign paths, directory scan и NPY-формат здесь не определяются.
- `euvs/plan.py` один раз связывает physical pair, source RGB и bound
  `PosedDepthSequence` по ordered identity, сохраняет endpoint/regressive
  mapping и компонует generic timeline/conditioning в лёгкий Gen3C request.
  Job, runtime, records и model execution ему неизвестны.
- `euvs/record.py` читает и напрямую пишет предметный `run.json` v1–v5:
  ordered pair/tokens, conditioning slots/mapping, geometry, model/LoRA,
  sampling, CP и RGB descriptor. v1 требует явного legacy opt-in; v4
  сохраняет исторический формат без LoRA strength, а новые LoRA-записи v5
  включают её. Файл не открывает RGB и не делает resume/filesystem gates.
  При чтении форма версии и обязательность её полей остаются у reader;
  повторные проверки значений делегированы существующим `__post_init__`.
- `gaussian/full.py` владеет только ordinary full-sequence science:
  continuous 24-fps rows, один W2C rebase, target slots `1..N` и hold-last
  padding. `gaussian/clips.py` выбирает exact 1-based independent clip IDs,
  сохраняет selection order, 120-target boundaries и right-aligned последнее
  окно с fresh source sequence. `gaussian/dense.py` читает exact
  reconstruction camera table, строит shortest-SO(3) trajectory с внешними
  factor/stride, планирует independent 120-target и overlap21 chunks,
  создаёт fresh source-local conditioning и передаёт explicit previous
  RGB/depth/valid следующему overlap chunk. `gaussian/record.py` строго
  читает и напрямую пишет неизменённые Gaussian records v1–v3, включая
  context diagnostics и seam MSE/PSNR. Builders создают dataclass напрямую;
  внешний reader повторно не вызывается. `gaussian/handoff.py` по двум exact
  attempt-ссылкам строит ожидаемые chunk paths, связывает ordered dense IDs
  с production slots и напрямую пишет native PNG плюс `prepare.json` без
  scan, staging или runtime README.
  Таблицу строит `build_dense_camera_table`, записывает
  `write_standard_camera_table`; промежуточного writer нет. PNG-выборка
  принадлежит отдельному v5: selection с ID и `bake_dense_stride`.
  Старое `dense.bake_stride` в recipe v3/v4 пока принимается по прежней схеме,
  но не управляет PNG и не передаётся в вычисление траектории.
- `gen3c/` содержит модельно-специфичную временную адаптацию закреплённого
  Gen3C: exact target-кадры на сетке 24 fps, компактный общий rebase камер
  и материализацию одного окна query W2C/K и single-source indices. Этот же
  подпакет запускает официальный Cache4D в отдельном окружении, передавая
  уникальные source-массивы один раз и рендеря все окна общего расписания.
  Последняя граница перед worker принимает также явный численный вход другого
  источника, не превращая его в EUVS-контракт.
- `gen3c/model_request.py` без открытия файлов собирает один конкретный
  base/full/LoRA model request и sampling identity. Он общий ровно для двух
  фактических потребителей — EUVS и Gaussian — и не
  является registry или общей фабрикой моделей.

Target RGB и diffusion checkpoint не читаются. Тяжёлые тензоры остаются
временной внутренней деталью worker, а текущий projected-nearest выбор одного
source использует общие `geometry.trajectory` и `geometry.polyline` и явно
остаётся заменяемым baseline. Модельно-специфичные адаптеры
зависят от общего входа, а не наоборот.
