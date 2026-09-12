# Рецепты DDW evaluation

`r4c-lidar-depth.yaml` фиксирует единственный поддерживаемый порядок matched-
сравнения: base, неизменный v1 checkpoint с Kendall EDM и экспериментальный
v2 checkpoint с Kendall EDM + LiDAR-depth вычисляются на одном
`completed_step` и seed. MoGe даёт относительную глубину; nonheldout LiDAR
подгоняет масштаб каждого основного кадра, а score считается только по
heldout LiDAR.

Сам job находится в игнорируемом `jobs/local/`, потому что пути к
PreparedRecord и checkpoint records называют конкретные запуски. Job должен
явно указать один `prepared_record`, один split и конкретные v1/v2 checkpoint
JSON; recipe не ищет `latest` или «лучший» checkpoint.

Добавляйте второй recipe только для действительно другого и отдельно
версионированного научного порядка. Не храните здесь пути сервера или
выборки данных.
