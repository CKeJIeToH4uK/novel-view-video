# R4c LoRA EDM baseline

`run.yaml` связывает синтетический tracked split, подготовленный DDW record,
baseline recipe и `gen3c_training/v1`. Это переносимый пример job;
тот же route уже выполняет fresh/resume через `_worker_v1.py`. Реальный
DDW85 job с закрытой выборкой и точными путями хранится в `jobs/local/`.
