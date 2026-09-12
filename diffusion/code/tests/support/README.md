# Общие тестовые заготовки

Папка содержит только небольшие фабрики и двойники, которые переиспользуют
несколько автоматических проверок. Production-код не импортирует эту папку.

- `waymo_v2.py` создаёт маленький generated-Parquet набор Waymo v2 с явно
  заданными partition, segment и timestamps;
- `gen3c_training.py` содержит общие Torch CPU-двойники Gen3C training,
  маленькие prepared items и две раскладки старой индексной карты;
- `gen3c_training_records.py` содержит Torch-free фабрики читаемых
  checkpoint records для model и workflow проверок;
- `ddw_evaluation.py` содержит только параметры evaluation и маленькие
  prepared items; численные проверки и настоящий MP4 находятся в integration;
- `stage5.py` собирает общий маленький runtime и resolved job для нескольких
  workflow-проверок Stage 5;

Новый файл добавляется сюда только после появления второго реального
тестового потребителя. Операторские программы и production diagnostics здесь
не размещаются.
