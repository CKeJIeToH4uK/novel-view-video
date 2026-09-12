# Приёмочные задачи

Здесь лежат переносимые формы для уникальных EUVS и Gaussian цепочек
Stage 11, которых нет среди обычных jobs. Они используют открытые малые selection и
буквальные `replace-run`/`replace-attempt`/`replace-export` значения; перед
реальной кампанией `accept` создаёт копии в игнорируемой
`jobs/local/<campaign>/` и подставляет точные внешние входы и producer refs.

- `euvs-*` покрывают source views, две политики generation, CP2 parity,
  evaluation и comparison с внешним tuned evaluation reference;
- `gaussian-*` покрывают пять уже реализованных порядков v1–v5;
- существующие Waymo preparation, R4c v1/v2 training, checkpoint selection и
  DDW evaluation копируются из своих обычных папок;
- шесть исторических Waymo задач также переиспользуются из
  `jobs/legacy/waymo/`: depth-candidates, depth-selection-v2, ddw-canary-v1,
  ddw-canary-v2, ddw-probe и ddw-survey. Их YAML здесь не дублируются.

Роли внешних входов и связь точных результатов описаны в
`STAGE11_HANDOFF.md`. Эти шесть задач дополняют прежние 19 случаев в той же
команде `accept a100-4`, а не создают отдельное аппаратное окно.

Новая форма появляется только для уже реализованной versioned scientific
identity, если готовую форму нельзя переиспользовать. Исторические порядки
добавляются только по явному списку приёмки; superseded, `n/a`, DA3 и
machine-specific пути не становятся поддерживаемыми приёмочными задачами.
