# Выборки данных

Папка является read-only config mount root для устойчивых ID конкретных
сцен, пар и кадров. Она не хранит machine paths, научные параметры или
результаты запусков.

Внешние схемы версии 1 уже принадлежат
`diffusion/code/src/novel_view/inputs/euvs/spec.py` и
`diffusion/code/src/novel_view/inputs/gaussian/spec.py`. EUVS-файл хранит
ordered пары и токены кадров. Gaussian-файл выбирает один из пяти явных
видов и хранит только scene/clip/pose/dense-camera ID. `euvs/one-pair.yaml`
и пять файлов `gaussian/` являются открытыми формами Stage 11; реальные
устойчивые ID подставляются только в campaign-local копии. В `waymo/`
лежат малые примеры DDW preparation и training/validation split. Полная
DDW85 выборка, split и одноразовая карта старых fit indices остаются
закрытым handoff. `local/` исключён из Git.
Исторический полный Waymo split также хранится только в
`local/waymo-segments-v1.yaml`; его ordered роли напрямую читает
`jobs/legacy/waymo/keysets/`, без старой машинной конфигурации.

В `euvs/euvs-*.yaml` сохранены четыре исторических scope с неизменным
порядком пар/токенов; `legacy/scene9/` хранит конкретные исторические
clip/pose/dense-camera ID. Они используются заданиями `jobs/legacy/`, а не
подменяют малые формы приёмки. `legacy/waymo/` показывает формы одного
трёхкамерного depth clip и восьми ordered candidate records; настоящие
значения остаются в `local/`. Новая выборка добавляется вместе с реальным
потребителем; генераторы кадров и пути машины здесь не размещаются.
