# Cache4D

Пакет владеет единым численным conditioning-форматом, общим для составной
Gen3C generation и standalone Waymo Cache4D.

- `request.py` — строки source evidence и лёгкие массивы conditioning;
- `protocol.py` — единственные имена входных и диагностических NPY,
  последовательная запись тяжёлых RGB/depth/valid;
- `runtime.py` — нормализация source, блочная unprojection, создание
  официального Cache4D и один сырой `render_cache`;
- `backend.py` и `_worker.py` — одноразовый standalone diagnostic через
  фиксированный Gen3C-интерпретатор без computation timeout.

Dataset-типы, generation markers, window policy, seed/stitching и Waymo
descriptors здесь не живут. Составной Gen3C worker импортирует только
`runtime.py`, но не standalone `_worker.py`.
