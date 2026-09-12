# Существующий обмен 3dgs и diffusion

## 3dgs → diffusion: готовый экспорт версии 1

Производитель — внешний Gaussian exporter, его код здесь не поставляется;
читатель — [inputs/gaussian/reader.py](../../diffusion/code/src/novel_view/inputs/gaussian/reader.py).
Компоненты не импортируют Python-код друг друга. `input.info_file` задания
diffusion указывает на существующее JSON-описание экспорта. Производитель
называет его `manifest.json`; читатель не требует именно такого имени.
Дополнительный manifest, каталог ресурсов, SHA или синхронизация не вводятся.

```text
export/
├── manifest.json
└── frames/pose_00002/
    ├── rgb.png
    ├── depth.npy
    └── mask.png
```

| Поле / файл | Смысл |
| --- | --- |
| `format`, `format_version` | Буквальные `gaussian_depth_export`, `1` |
| `inputs` | Scene ID и splats variant; старые абсолютные пути описывают происхождение, кадры читаются не по ним |
| `camera` | Имя камеры, исходные `[height,width]`, K, OpenCV axes, `world_to_camera`, pixel centers `0.5` |
| `render`, `depth`, `mask` | Прежние параметры RGB+ED, expected camera-Z в метрах, float32/NaN, маска uint8/255 |
| `target` | Имя сдвига и `shift_camera_xyz_m` в осях исходной камеры |
| `frames` | Упорядоченные pose ID, `timestamp_ns`, source/target W2C и относительные RGB/depth/mask пути |
| `summary` | Статистика уже выполненного экспорта, не инструкция читателю пересканировать файлы |
| `rgb.png` | Отрисованный 3DGS RGB, не исходный sensor frame |
| `depth.npy` | Float32 ожидаемая глубина по оси Z камеры в метрах; невалидное значение — NaN |
| `mask.png` | 255 означает пригодную глубину, 0 — непригодную |

Оси OpenCV: x вправо, y вниз, z вперёд. W2C переводит мировую точку в
координаты камеры. Для сдвига камеры `v` производитель пишет
`W2C_target = [I | -v] @ W2C_source`; K при чистом переносе не меняется.
`left_1.0` соответствует `[-1,0,0]` в осях камеры. Матрицы, timestamps
и `pose_index` остаются связанными одной строкой и порядком `frames`.

Пути кадра соединяются с родительской папкой переданного `info_file`;
старые `inputs.splats/transforms/intrinsics/environment` не используются
для открытия RGB/depth/mask. Поэтому готовый export переносится вместе со
своей папкой `frames`, без исправления старых абсолютных provenance полей.
Dense-задача получает нужные transforms/intrinsics отдельными явными
полями задания, а не пытается восстановить их из чужого пути.

Читатель сначала разбирает JSON и не открывает все payload. По запросу он
читает один triplet, переводит RGB из BGR OpenCV, получает valid из
`mask==255` и finite positive camera-Z; затем применяет существующий
cover/crop raster. Политики повторного запуска или публикации читатель
не добавляет. У прежнего 3dgs exporter есть собственная atomic-публикация;
она не является частью diffusion.

Единственный зафиксированный пример —
[gaussian_depth_export_v1.json](../../diffusion/code/tests/contract/fixtures/gaussian_depth_export_v1.json).
Он содержит scene 9 и две специально малые строки pose 2/5, а не полный
478-frame export. Проверка читателя находится в
[test_gaussian_export.py](../../diffusion/code/tests/contract/test_gaussian_export.py).
Новой копии fixture и нового schema framework нет.

## diffusion → 3dgs: подготовленные PNG

Существующий [generation/gaussian/handoff.py](../../diffusion/code/src/novel_view/generation/gaussian/handoff.py)
и `gaussian_generation/v5` читают явно выбранные independent/overlap
попытки, материализуют PNG для выбранных dense-camera ID и пишут прежний
`prepare.json`. Это выход diffusion, а не новый двусторонний API:
работающего 3dgs reader для этого файла в текущей волне нет.
Автоматический импорт, обратная зависимость и запекание не добавляются.

Новый раздел или поле добавляется только после изменения реального
производителя/потребителя и его минимального численного/PNG доказательства.
