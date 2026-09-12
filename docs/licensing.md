# Лицензии и границы распространения

Публикуются исходники собственного diffusion-кода, Dockerfile и списки
зависимостей. Готовые образы, внешние исходники моделей, веса, датасеты,
подготовленные данные и обученные checkpoints не включены.

Собственный код оформлен по [Apache-2.0](../LICENSE).
[Сторонние уведомления](../THIRD_PARTY_NOTICES.md) сохраняют атрибуцию и
исключения. Лицензия проекта не перелицензирует зависимости или данные.

Справочник ниже основан на аудите 9–12 сентября 2026 года и закреплённых
версиях [Dockerfile](../containers/diffusion/Dockerfile).
Это описание проверенных источников, не юридическое заключение и не
утверждение, что весь собранный бинарный образ можно свободно распространять.

## 1. Закреплённые внешние исходники

Во всех строках дата проверки — 09.09.2026. Полная ревизия содержится в
ссылке; сокращение рядом нужно только для чтения. Веса перечислены отдельно.

| Компонент и закреплённый источник | Лицензия кода | Что входит и что сохранить при передаче |
| --- | --- | --- |
| Gen3C `db2ffe12` — [LICENSE](https://raw.githubusercontent.com/nv-tlabs/GEN3C/db2ffe12ced12ddafcec5e0422ee46ce8520746b/LICENSE), [ATTRIBUTIONS](https://raw.githubusercontent.com/nv-tlabs/GEN3C/db2ffe12ced12ddafcec5e0422ee46ce8520746b/ATTRIBUTIONS.md) | Apache-2.0 основного кода; вложенные заимствования имеют свои уведомления | Установленный пакет Cosmos/Gen3C и нужные исходники в образе. Сохранить LICENSE, ATTRIBUTIONS и относящиеся уведомления; не объявлять всю вложенную кодовую базу единым нашим Apache-кодом |
| VGGT-Omega `39a0cb8a` — [LICENSE](https://raw.githubusercontent.com/facebookresearch/vggt-omega/39a0cb8af88554f15ddcb5354cd52bde588fa014/LICENSE) | FAIR Noncommercial Research License v1, 16.10.2024, с правилами допустимого использования | Пакет в отдельном `/opt/envs/vggt`; ограничение некоммерческого исследования относится уже к коду, а не только к весам. Текст присутствует в установленном пакете |
| MoGe `925b8ed8` — [LICENSE](https://raw.githubusercontent.com/microsoft/MoGe/925b8ed835a7a9cdb7578ba15c658a0afc969030/LICENSE) | MIT; встроенные части DINOv2 отдельно Apache-2.0 | Только вариант `moge`; сохранить собственные и вложенные уведомления |
| SAM2 `2b90b9f5` — [LICENSE](https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE), [LICENSE_cctorch](https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE_cctorch) | Apache-2.0 и BSD-3-Clause для cctorch | Установленный пакет; оба текста присутствуют. `SAM2_BUILD_CUDA=0` не отменяет уведомления входящего кода |
| DINOv2 `7764ea0f` — [LICENSE](https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE), [CellDINO code](https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE_CELL_DINO_CODE) | Стандартный DINOv2 — Apache-2.0; включённый CellDINO-код — CC BY-NC 4.0 | Docker копирует весь каталог `dinov2/`, включая CellDINO, и `LICENSE*`. Неиспользуемый некоммерческий код всё равно входит в распространяемый образ; стандартная метрика использует только ViT-B/14 |
| Apex `6424da3b` — [LICENSE](https://raw.githubusercontent.com/NVIDIA/apex/6424da3b4faa6c8f062da4a48c424fff3f02d42d/LICENSE) | BSD-3-Clause | Собранный CUDA wheel в Gen3C prefix; сохранить уведомление NVIDIA и условия |
| Transformer Engine `7a7225c4` — [LICENSE](https://raw.githubusercontent.com/NVIDIA/TransformerEngine/7a7225c403bc704264e7cf437369594aeb8b3ba3/LICENSE) | Apache-2.0 | Wheel 1.12.0 и common/CUDA часть; собственная лицензия не заменяет условия CUDA/cuDNN |
| TE → cudnn-frontend `936021bf` — [LICENSE](https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/936021bfed8c91dc416af1588b2c4eca631a9e45/LICENSE.txt) | MIT | Закреплённый вложенный источник сборки; учесть включённые части/уведомления в собранном продукте |
| TE → GoogleTest `f8d7d77c` — [LICENSE](https://raw.githubusercontent.com/google/googletest/f8d7d77c06936315286eb55f8de22cd23c188571/LICENSE) | BSD-3-Clause | Вложенный источник сборки, не самостоятельный runtime-пакет. Наличие checkout в builder не доказывает включение всего GoogleTest в финальный образ |
| utils3d `3fab839f` — [LICENSE](https://raw.githubusercontent.com/EasternJournalist/utils3d/3fab839f0be9931dac7c8488eb0e1600c236e183/LICENSE) | MIT | MoGe delta, сохранить уведомление |
| pipeline `866f059d` — [LICENSE](https://raw.githubusercontent.com/EasternJournalist/pipeline/866f059d2a05cde05e4a52211ec5051fd5f276d6/LICENSE) | MIT | MoGe delta, сохранить уведомление |
| DA3 `3d835ec1` — [LICENSE](https://raw.githubusercontent.com/ByteDance-Seed/Depth-Anything-3/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/LICENSE) | Apache-2.0 кода | Только исторический источник исследовательского адаптера. `research/da3_nested` не входит в wheel, официальный CLI или production-образ; прежние DA3 locks не входят в production inventory |

Минимальный установщик [Miniforge 26.5.3-0](https://raw.githubusercontent.com/conda-forge/miniforge/26.5.3-0/LICENSE)
имеет BSD-3-Clause; его лицензия не покрывает устанавливаемые пакеты.
Сам установщик остаётся в builder.

Три явно собираемых sdist из Dockerfile:
[iopath 0.1.10](https://pypi.org/project/iopath/0.1.10/) — MIT,
[asciitree 0.3.3](https://pypi.org/project/asciitree/0.3.3/) — MIT,
[antlr4-python3-runtime 4.9.3](https://raw.githubusercontent.com/antlr/antlr4/4.9.3/LICENSE.txt)
— BSD-3-Clause для ANTLR с отдельными вложенными уведомлениями в тексте.
У первых двух LICENSE прочитан из точного sdist и присутствует после
установки; у ANTLR собственного полного текста в установленном пакете не
найдено. Источники sdist уже указаны в Dockerfile, новый список загрузок не нужен.

## 2. Конкретные веса

Ссылки относятся к проверенным карточкам, а не удостоверяют файлы пользователя.
Пользователь самостоятельно получает модели и соблюдает условия их доступа.

| Используемый набор | Условия и источник | Размещение и решение |
| --- | --- | --- |
| Gen3C `nvidia/GEN3C-Cosmos-7B` | [Карточка `9bcfdb4f`](https://huggingface.co/nvidia/GEN3C-Cosmos-7B/blob/9bcfdb4f3924f41376daeadf6200826c12a3bf8e/README.md): NVIDIA Open Model License, не Apache кода | Внешние `gen3c/base` в inference recipe и `gen3c/official/checkpoints/Gen3C-Cosmos-7B/model.pt` в R4c; соответствие этих раскладок уточняет передача, не автоматическая синхронизация |
| Cosmos tokenizer CV8x8x8-720p | [Карточка `b6af4953`](https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p/blob/b6af495317c76f287a4131e9299936b1533f5f9f/README.md): NVIDIA Open Model License | Внешний `gen3c/official/checkpoints/Cosmos-Tokenize1-CV8x8x8-720p`, собственные условия и уведомления NVIDIA |
| T5-11B | [Карточка `90f37703`](https://huggingface.co/google-t5/t5-11b/blob/90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3/README.md): Apache-2.0 | Внешний `gen3c/official/checkpoints/google-t5/t5-11b`; сохранять относящиеся тексты, включая tokenizer |
| VGGT-Omega | [Публичная карточка](https://huggingface.co/facebook/VGGT-Omega), API сообщает ревизию `1041e80fc0e911235d3426b0a3d9a81075111a53`; доступ требует согласия и одобрения | Исходные `LICENSE.txt` и README этой ревизии вернули HTTP 401 без авторизации. Принятые условия конкретного получателя не получены; не подменять их лицензией кода. EUVS использует `vggt-omega/vggt_omega_1b_512.pt`, legacy recipe — `vggt-omega/model.pt` |
| MoGe ViT-L | [Принятая карточка `979e84da`](https://huggingface.co/Ruicheng/moge-vitl/blob/979e84da9415762c30e6c0cf8dc0962896c793df/README.md): Apache-2.0; [нынешняя `ad326bfb`](https://huggingface.co/Ruicheng/moge-vitl/blob/ad326bfb61facd6c52b5a825bc1e34d7c97d9672/README.md): MIT | Внешний `moge-vitl/model.pt`. Не заменять условия принятого источника нынешней карточкой; оба текста реально получены |
| SAM2.1 Hiera Large | [Карточка `665f8e2a`](https://huggingface.co/facebook/sam2.1-hiera-large/blob/665f8e2ad61cf5f53d65644ff27c8ee525124610/README.md): Apache-2.0 | Внешний `sam2/sam2.1_hiera_large.pt`, не часть установленного кода SAM2 |
| Grounding DINO Tiny | [Карточка `a2bb814d`](https://huggingface.co/IDEA-Research/grounding-dino-tiny/blob/a2bb814dd30d776dcf7e30523b00659f4f141c71/README.md): Apache-2.0 | Внешний `grounding-dino-tiny` с processor/tokenizer. Реальный загрузчик — Transformers 4.49.0; отдельный Git-checkout GroundingDINO Dockerfile не устанавливает |
| DINOv2 ViT-B/14 | [README точной версии исходников](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md) относит стандартные код/веса к Apache-2.0 | Оригинальный `dinov2_vitb14_pretrain.pth` из `dl.fbaipublicfiles.com`, внешний `dinov2/`. Не HF-конверсия `dinov2-base`; CellDINO/XRay-веса не используются и не скачивались |
| LPIPS linear weights | [LPIPS 0.1.4](https://pypi.org/project/lpips/0.1.4/), BSD-2-Clause в установленном LICENSE | Шесть маленьких `.pth` уже включены в wheel/образ, суммарно 46 356 байт. Метрика выбирает `v0.1/alex.pth`; это исключение из упрощённой фразы «в образе вообще нет весов» |
| AlexNet backbone для LPIPS | [Torchvision 0.21.0](https://github.com/pytorch/vision/blob/v0.21.0/README.md) отдельно предупреждает об условиях pretrained-моделей | Реальный loader указывает `alexnet-owt-7be5be79.pth` на `download.pytorch.org`; это отдельная загрузка/кеш, не линейные веса LPIPS. BSD кода недостаточно, чтобы объявить права на эти веса полностью проверенными |
| DA3NESTED-GIANT-LARGE-1.1 | [Карточка `b2359bdf`](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/blob/b2359bdf726fb44ef62acca04d629dcf158053e7/README.md): CC BY-NC 4.0 | Только экспериментальная возможность, не production/Stage 11 по умолчанию. Не обещать коммерческое использование весов на основании Apache кода |

## Режим Gen3C

Текущий [владелец сессии](../diffusion/code/src/novel_view/models/gen3c/session.py)
передаёт `disable_guardrail=True`. Фильтрация текста/видео и размытие лиц
в этот исследовательский путь не добавлены.

[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
и [README закреплённого Gen3C](https://github.com/nv-tlabs/GEN3C/blob/db2ffe12ced12ddafcec5e0422ee46ce8520746b/README.md#gpu-memory-requirements)
нужно рассматривать вместе: README рекомендует этот флаг для экономии памяти,
но наличие рекомендации не доказывает отдельное исключение из условий модели.
Соответствие конкретного применения этим условиям не подтверждено данным
репозиторием. Публикация исходников не означает разрешения любого использования весов.

## Данные и результаты

Waymo/DDW разработан с использованием Waymo Open Dataset; предусмотренная
библиографическая атрибуция находится в THIRD_PARTY_NOTICES.md.
См. [условия Waymo](https://waymo.com/open/terms/).

[EUVS](https://huggingface.co/datasets/ai4ce/EUVS-Benchmark) использует
[nuPlan](https://nuplan.org/nuplan); метка лицензии кода или devkit
не заменяет условия первоначального датасета. Регистрация и доступ
обеспечиваются пользователем.

Gaussian exports, кадры, prepared-артефакты и LoRA/observer не становятся
автоматически Apache-2.0. Их условия зависят также от исходных данных
и моделей; реальные файлы в публичный снимок не включены.

## Сборка не равна поставке готового образа

Docker получает зависимости отдельно. Их тексты сохраняются в package
metadata, /opt/upstream, /opt/licenses/conda и /opt/licenses/distil3d.
[Дополнительные уведомления](../containers/diffusion/licenses/README.md)
включают собственные тексты ANTLR, Decord, Triton, ModelOpt и GNU.

Для будущей передачи готового образа остаётся отдельная работа:
соответствующие исходники GPL/LGPL-компонентов, вложенные codecs и solver,
а также условия NVIDIA и некоммерческих компонентов. В частности:

- FFmpeg 7.0.2-static из imageio-ffmpeg использует GPLv3+.
  [Страница исходников](https://ffmpeg.org/download.html) — справочная ссылка,
  не подтверждение наличия соответствующих исходников этой статической сборки.
- CUDA-GDB: [исходники CUDA 12.4](https://github.com/NVIDIA/cuda-gdb/releases/tag/cuda-toolkit-12.4-release).
- Sysroot: [AlmaLinux glibc source RPM](https://vault.almalinux.org/9.5/BaseOS/Source/Packages/glibc-2.34-125.el9_5.8.alma.1.src.rpm);
  отдельно учитывать включённые kernel headers.
- CBC внутри PuLP 3.3.2 имеет собственные
  [уведомления EPL-1.0](https://github.com/coin-or/pulp/blob/3.3.2/pulp/solverdir/cbc/linux/i64/coin-license.txt).
- DINOv2 в Docker включает CellDINO-код под некоммерческими условиями,
  даже если стандартная метрика его не использует.

Полнота материалов готового образа не заявляется. В исходный код не
добавляются автоматические лицензионные проверки или запреты запуска.
