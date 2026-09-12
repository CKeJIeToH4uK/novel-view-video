# Область лицензии и сторонние уведомления

Copyright 2026 Bulat Mannanov.

Собственный код компонента `diffusion/code` распространяется по Apache-2.0
(см. LICENSE). Этот выбор не перелицензирует внешние библиотеки,
данные, веса или заимствования под отдельными условиями. Авторские вклады
перечислены в AUTHORS.md; принадлежность прав не стирает авторство.

## Научные компоненты

- Gen3C / Cosmos: код NVIDIA и участников, Apache-2.0; сохранённые
  LICENSE и ATTRIBUTIONS.md находятся в образе под `/opt/upstream/gen3c`.
  Интеграция, управление процессами и перенесённые вычисления переработаны
  в `novel_view`; это не неизменённая поставка upstream.
- DINOv2: стандартный код Meta Platforms и участников — Apache-2.0;
  каталог `/opt/upstream/dinov2` сохраняет отдельные LICENSE* для CellDINO
  и XRay. Копируемый CellDINO-код имеет некоммерческие ограничения.
- VGGT-Omega: Meta Platforms, FAIR Noncommercial Research License.
  Полный текст кода входит в установленный `vggt_omega` distribution;
  принятые условия gated-весов проверяются отдельно.
- SAM2: Apache-2.0 с отдельным BSD-3-Clause уведомлением cctorch;
  оба текста входят в установленный distribution.
- MoGe, utils3d, pipeline: MIT, с отдельными уведомлениями включённых
  частей DINOv2. Поставляются только в варианте `moge`.
- Apex: BSD-3-Clause. Transformer Engine: Apache-2.0; его вложенные
  cudnn-frontend и GoogleTest имеют соответственно MIT и BSD-3-Clause.
- Grounding DINO используется через Transformers; код Transformers и
  Diffusers — Apache-2.0. Условия конкретных weights рассматриваются отдельно.
- LPIPS 0.1.4: BSD-2-Clause, copyright Richard Zhang, Phillip Isola,
  Alexei A. Efros, Eli Shechtman, Oliver Wang. Установленный пакет сохраняет
  LICENSE и маленькие линейные веса; pretrained AlexNet — отдельный ресурс.

Полные тексты установленных зависимостей находятся в их `.dist-info`,
пакетных каталогах и системных каталогах лицензий. Ссылка на эти места не
заменяет отсутствующий текст: дополнения ANTLR/Decord/Triton/ModelOpt/FFmpeg
хранятся в `containers/diffusion/licenses` и копируются в образ под
`/opt/licenses/distil3d`. Они не заменяют лицензии вложенных компонентов.
Conda-тексты и рецепты из точных пакетов сохраняются в
`/opt/licenses/conda/<prefix>/<name-version-build>/`; наличие рецепта
само по себе не означает передачу всех соответствующих исходников.

## Waymo Open Dataset

Waymo/DDW-часть разработана с использованием Waymo Open Dataset.
Атрибуция в одной из форм, предусмотренных §1 условий датасета:

```bibtex
@misc{waymo_open_dataset,
  title = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year = {2019-2025}
}
```

Условия: https://waymo.com/open/terms/. Эта атрибуция не распространяет
ограничения датасета на весь независимый код и не подтверждает выполнение
всех условий конкретной поставки. Данные, prepared-артефакты и checkpoints
не выдаются вместе с Python-пакетом; оператор получает данные и внешние
модельные веса самостоятельно.

## Образы, данные и ограничения передачи

CUDA и другие NVIDIA binaries имеют собственные условия, не Apache-2.0
проекта. Сборка FFmpeg 7.0.2-static из imageio-ffmpeg использует GPLv3+;
лицензия BSD Python-обёртки её не заменяет. Conda/system packages также
включают GPL/LGPL/MPL-компоненты со своими текстами и обязанностями.

Этот файл сохраняет атрибуцию и границы лицензии, но не объявляет полноту
всех бинарных notices или право передачи готового Docker-архива. Неуточнённые
условия выбранного режима Gen3C и бинарных компонентов, сведения о FFmpeg
и самостоятельном получении ресурсов перечислены в `docs/licensing.md`
корня репозитория. Код 3DGS в эту публикацию не включён.
Внешние условия нельзя закрыть добавлением одного Apache LICENSE.

Собственные LICENSE и THIRD_PARTY_NOTICES.md включены в wheel/sdist
`novel-view-pipeline`. Внешние model packages, исходники, модели и данные
в этот wheel/sdist не включаются; Docker добавляет зависимости отдельно.
