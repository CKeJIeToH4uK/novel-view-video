# R4c LoRA с LiDAR-depth

`run.yaml` связывает те же подготовленные данные и split с экспериментальным
`gen3c_training/v2`. Отличие от baseline находится в отдельном recipe;
реальный DDW85 job остаётся в `jobs/local/`. Запуск идёт через отдельный
`_worker_v2.py`: fresh attempt сначала подгоняет и замораживает observer, а
resume восстанавливает его из точного v2 checkpoint без повторного fit.

Гипотеза и правило сравнения с baseline записаны в
[`studies/r4c-lidar-depth`](../../../studies/r4c-lidar-depth/).
