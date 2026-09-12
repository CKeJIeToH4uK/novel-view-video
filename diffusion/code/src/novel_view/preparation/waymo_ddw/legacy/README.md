# Исторические DDW-исследования

Здесь находятся исторические canary, probe, survey, fit и visual audit. Поддерживаемая
подготовка R4c остаётся в родительской папке и не меняет свой порядок.

`spec.py` читает поля четырёх jobs и их научную ось без открытия данных,
результатов и моделей. Конкретный clip берётся из selection, готовое решение
по глубине — по явному locator в `/runs`; машинные пути остаются в profile.

Различия научных версий оформляются конкретными соседними функциями/модулями.
Общий registry, coordinator, повторные внутренние validators, staging,
автоматическое переиспользование старых запусков и поиск результатов здесь
не создаются.

- `axis`, `masks`, `headroom`, `selection` — варианты и прежние численные правила.
- `gate`/`calibration` и `v2_gate`/`v2_calibration` — разные научные версии.
- `source`, `path`, `render`, `local`, `reference` — вход, общая траектория,
  два one-source warp и отдельный global Cache4D reference.
- `canary`, `v2_canary` — четыре явных порядка и различия STOP/continue.
- `execution` — общий конкретный FRONT/MoGe input; canary сначала читает
  depth-selection, fit/audit открывают clip из assignment/result напрямую.
- `record` — простая запись новых family `/v2`; исторических readers нет.
- `assignment` — прежнее сбалансированное распределение вариантов по полной
  оси; поднабор выбирается после него.
- `fit` — headroom, reference и mask decision; только accepted получает
  повторный render, научное сравнение и condition RGB/packed-known.
- `audit` — отдельный AUDIT_MASK_GATE для rejected item, повторный render
  и визуальный preview без изменения accepted set.
- `fit_spec` — явные fit/collection/audit формы; `fit_record` — новые
  clip/v2, collection/v1 и visual-audit/v2. Collection сохраняет ссылку на
  каждый исходный item record; NPY пути относятся к папке этого record.
- `legacy_fit_record` — только чтение прежнего rejected clip/v1 для audit;
  это не reader старых accepted payload или всей кампании.

Новая научная версия получает отдельный явный порядок; общая часть
извлекается только при втором реальном совместимом использовании.
