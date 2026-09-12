# Gen3C

Папка отделяет модельные возможности Gen3C от научного порядка generation.

- `cache4d/` — общий численный conditioning, тяжёлый model-runtime и
  самостоятельная диагностическая process-граница.
- `spec.py` сохраняет generation FPS24 и отдельно владеет точным R4c
  FPS10-контрактом растра, latent и conditioner shapes.
- `environment.py` отображает фиксированные container cache и temporary
  roots в точное окружение Gen3C без проверки машины и путей.
- `request.py` содержит только лёгкие model artifact, sampling и CP topology
  значения, неизменные на время одной resident-сессии.
- `session.py` владеет official pipeline, загрузкой base/full/LoRA весов,
  CP-группой, нормализацией одного seed, одним window-вызовом и границей RNG
  между запросами. Он не знает научный порядок окон или файловый protocol.
- `lora_weights.py` — модельная структура LoRA, restricted-load только
  вложенного `adapter`, полная замена stable LoRA tensors, временное
  включение/выключение и применение inference strength без чтения training
  topology, optimizer или RNG.

Окна, stitching, workflow, dataset-типы, Cache4D-композиция и output
publication сюда не добавляются. `__init__.py` остаётся холодным и ничего не
реэкспортирует.
