# Точные зависимости diffusion

Эта папка хранит закреплённые зависимости Linux/amd64. `pyproject.toml`
описывает Python-пакет, а lock-файлы задают версии и сборки окружений.
Модели, данные, checkpoints и сторонние checkout в эту папку и в Git не
входят.

## Production candidate

Первая выпускаемая пара образов — только `core+moge`:

| Владелец | Locks | Результат |
|---|---|---|
| `/opt/envs/core` | `core.conda.lock`, `core.pip.lock` | лёгкий CLI, конфигурация, NumPy/OpenCV/PyArrow |
| `/opt/envs/gen3c` | `gen3c-eval.conda.lock`, `gen3c-eval.pip.lock` | Gen3C, evaluation, CUDA/PyTorch, SAM2, LPIPS, DINOv2 |
| `/opt/envs/vggt` | `vggt.conda.lock`, `vggt.pip.lock` | независимый VGGT-Omega prefix |
| MoGe delta | `moge-gen3c.pip.lock` | 29 wheels поверх `/opt/envs/gen3c`; отдельного prefix нет |

`production-locks.tsv` — машинный ordered inventory. Каждая его строка
имеет буквальный вид
`<relative-path>\tsha256:<lowercase-64-hex>\n`, включая последний перевод
строки. SHA-256 текущего файла является `lock_revision`. При изменении
locks их хеши пересчитываются в тех же семи строках. Это автоматический
учёт состава кандидата; значения не сравниваются со старыми артефактами.

По решению пользователя от 5 сентября 2026 года установка не требует
ручных SHA публичных или собранных wheels/sdist/installer, Conda `#md5`
либо побайтного совпадения повторных сборок. Все принятые версии, точные
Conda URLs, исходные Git revisions и Docker base digests сохраняются.

`cpu-test.pip.lock` остаётся только тестовым слоем: CPU Torch/Torchvision
исполняют малые R4c и raster/preprocess проверки, scikit-image/SciPy —
численные проверки метрик. Их версии переиспользуют принятый Gen3C-набор,
но CUDA wheels, модели и их API в тестовый слой не добавляются.
Эта дельта не меняет production-prefix или production-locks.tsv.
`da3.*.lock` сохраняет
историческое доказательство `0B`, но DA3 является `research-only` и не входит
ни в inventory, ни в production candidate. Удалённые дублирующие
`gen3c.*.lock` не возвращаются.

## Неподменяемые основы

Production stages используют буквальные образы:

```text
nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10
nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749
```

Miniforge также является буквальным входом:

```text
https://github.com/conda-forge/miniforge/releases/download/26.5.3-0/Miniforge3-26.5.3-0-Linux-x86_64.sh
```

Devel-образ на разрешённом Linux-узле показал Ubuntu 22.04.4, CUDA
12.4.1, `nvcc 12.4.131`, GCC/G++ 11.4.0 и Make 4.3. Сборщики CUDA wheels
используют не плавающий системный toolchain, а принятый
`/opt/envs/gen3c`: Python 3.10.20, pip 25.0.1, setuptools 76.0.0,
wheel 0.45.1, GCC/G++ 12.4.0, CMake 4.2.3, исполняемый Ninja 1.13.0,
PyTorch 2.6.0+cu124 и Triton 3.2.0. Conda-слой также содержит Ninja 1.13.2
и setuptools 84.0.0, но после replay владельцами команд `ninja` и
`setuptools` считаются версии из pip lock; это проверяется в builder до
компиляции.

Для CUDA-сборки задаются `CC`/`CXX` на компиляторы prefix,
`CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST=8.0` и, где требуется,
`NVTE_CUDA_ARCHS=80`. `LDFLAGS=-Wl,--strip-all` оставляет компактные
CUDA wheels без неисполняемой таблицы символов. Нормализация времени
исходников и заголовков, `SOURCE_DATE_EPOCH` и повторные сборки ради
совпадения байтов не требуются.
Для Transformer Engine заголовки и библиотека cuDNN
берутся из уже зафиксированного `nvidia-cudnn-cu12` внутри prefix. Системный
toolchain devel-образа, исходные checkout, wheelhouse и кеши остаются только
в builder. Сам точный Conda-prefix переносится без вырезания его lock-owned
пакетов, даже если среди них есть compiler/CUDA-devel пакеты.

## Особые публичные артефакты

Обычные binary wheels получает `pip download` по точным версиям строк
lock. Артефакты, которым нужен не стандартный PyPI resolver, перечислены
буквально:

| Distribution | Exact artifact |
|---|---|
| `torch==2.6.0+cu124` | `https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp310-cp310-linux_x86_64.whl` |
| `torchvision==0.21.0+cu124` | `https://download-r2.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp310-cp310-linux_x86_64.whl` |
| `triton==3.2.0` | `https://download-r2.pytorch.org/whl/triton-3.2.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` |
| `transformer_engine==1.12.0` | `https://files.pythonhosted.org/packages/e5/62/1449c13003748f6405ca6982946d1aeb522baf5e9b62b9def037772241cb/transformer_engine-1.12.0-py3-none-any.whl` |
| `transformer_engine_cu12==1.12.0` | `https://files.pythonhosted.org/packages/d1/9e/2dd16fef4187b5389d0ea64da5615c6a7446121cffe5dff6e1f4884736c2/transformer_engine_cu12-1.12.0-py3-none-manylinux_2_28_x86_64.whl` |
| `decord==0.6.0` | `https://files.pythonhosted.org/packages/11/79/936af42edf90a7bd4e41a6cac89c913d4b47fa48a26b042d5129a9242ee3/decord-0.6.0-py3-none-manylinux2010_x86_64.whl` |

Официальные wheels устанавливаются и импортируются на Python 3.10, однако
стандартный `pip check` отвергает их внутренние метаданные:
внешние `py3-none` filenames скрывают внутренние `WHEEL` tags
`cp36-cp36m-manylinux2010_x86_64` (Decord) и
`cp38-cp38-manylinux_2_28_x86_64` (TE-cu12). Это ошибка описания
совместимости пакетов, а не доказанная ошибка выполнения их библиотек.
Официальные artifacts из таблицы остаются входами сборки. Builder извлекает каждый
wheel, подтверждает ровно этот исходный tag, заменяет его на tag внешнего
имени и выполняет `python -m wheel pack`;
меняются только metadata и пересозданный `RECORD`, не бинарный payload.
`gen3c-eval.pip.lock` сохраняет версии Decord 0.6.0 и TE-cu12 1.12.0.
Малый извлечённый lock двух wheels заставляет offline `--force-reinstall`,
затем полный offline replay и `pip check` проверяют конечный prefix.

## Wheels из исходников

Каждый Git-источник получается по полному URL, переводится в detached exact
revision и проверяется `git rev-parse HEAD`. Общая форма команды:

```bash
<prefix>/bin/python -m pip wheel \
  --disable-pip-version-check --no-deps --no-build-isolation \
  --wheel-dir /wheelhouse <exact-source-directory>
```

Точные варианты:

| Distribution | Exact source | Дополнительные переменные | Ожидаемый wheel |
|---|---|---|---|
| `apex==0.1` | `https://github.com/NVIDIA/apex.git@6424da3b4faa6c8f062da4a48c424fff3f02d42d` | `APEX_CPP_EXT=1 APEX_CUDA_EXT=1 TORCH_CUDA_ARCH_LIST=8.0 LDFLAGS=-Wl,--strip-all`; `/opt/envs/gen3c/bin/python` | `apex-0.1-cp310-cp310-linux_x86_64.whl` |
| `transformer_engine_torch==1.12.0` | `https://github.com/NVIDIA/TransformerEngine.git@7a7225c403bc704264e7cf437369594aeb8b3ba3` | `NVTE_RELEASE_BUILD=1 NVTE_CUDA_ARCHS=80 TORCH_CUDA_ARCH_LIST=8.0 LDFLAGS=-Wl,--strip-all`; выполнить из `transformer_engine/pytorch` после установки точных public meta/common wheels; cuDNN include/lib берутся из lock-owned `nvidia-cudnn-cu12` | `transformer_engine_torch-1.12.0-cp310-cp310-linux_x86_64.whl` |
| `SAM-2==1.0` | `https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4` | `SAM2_BUILD_CUDA=0`; `/opt/envs/gen3c/bin/python` | `sam_2-1.0-py3-none-any.whl` |
| `vggt-omega==0.0.1` | `https://github.com/facebookresearch/vggt-omega.git@39a0cb8af88554f15ddcb5354cd52bde588fa014` | `/opt/envs/vggt/bin/python` | `vggt_omega-0.0.1-py3-none-any.whl` |
| `moge==2.0.0` | `https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030` | `/opt/envs/gen3c/bin/python`; `utils3d` и `pipeline` wheels уже в wheelhouse | `moge-2.0.0-py3-none-any.whl` |
| `utils3d==1.3` | `https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183` | `/opt/envs/gen3c/bin/python` | `utils3d-1.3-py3-none-any.whl` |
| `pipeline==1.0.0` | `https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6` | `/opt/envs/gen3c/bin/python` | `pipeline-1.0.0-py3-none-any.whl` |

Transformer Engine checkout также фиксирует gitlinks cuDNN frontend
`936021bfed8c91dc416af1588b2c4eca631a9e45` и GoogleTest
`f8d7d77c06936315286eb55f8de22cd23c188571`. Для PyTorch extension нужны
headers основного exact checkout; common wheel не пересобирается.

Три PyPI-пакета публикуются только как source distributions. Их архивы
сначала скачиваются, затем собираются той же точной
`/opt/envs/gen3c` командой:

| Distribution | Exact sdist | Ожидаемый wheel |
|---|---|---|
| `iopath==0.1.10` | `https://files.pythonhosted.org/packages/72/73/b3d451dfc523756cf177d3ebb0af76dc7751b341c60e2a21871be400ae29/iopath-0.1.10.tar.gz` | `iopath-0.1.10-py3-none-any.whl` |
| `antlr4-python3-runtime==4.9.3` | `https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/antlr4-python3-runtime-4.9.3.tar.gz` | `antlr4_python3_runtime-4.9.3-py3-none-any.whl` |
| `asciitree==0.3.3` | `https://files.pythonhosted.org/packages/2d/6a/885bc91484e1aa8f618f6f0228d76d0e67000b0fdd6090673b777e311913/asciitree-0.3.3.tar.gz` | `asciitree-0.3.3-py3-none-any.whl` |

Собранные wheels должны иметь принятые имена и версии, устанавливаться
из wheelhouse и проходить `pip check` и нужные холодные импорты.
Другие байты ZIP/ELF при тех же версиях и исходниках не останавливают сборку.

## Runtime и builder source trees

В runtime нужны ровно два исходных дерева без `.git` и кешей:

```text
/opt/upstream/gen3c   https://github.com/nv-tlabs/GEN3C.git@db2ffe12ced12ddafcec5e0422ee46ce8520746b
/opt/upstream/dinov2 https://github.com/facebookresearch/dinov2.git@7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

Gen3C подключается к `/opt/envs/gen3c` через image-owned `.pth`; DINOv2
передаётся его concrete owner как явный repository path. Это не host
`PYTHONPATH`. Checkouts Apex, Transformer Engine, SAM2, VGGT-Omega, MoGe,
utils3d и pipeline, а также wheelhouses и compiler caches остаются только в
builder stages.

## Один wheel проекта

Состав исходников для wheel — `diffusion/code/pyproject.toml`,
`diffusion/code/README.md`, `diffusion/code/LICENSE`,
`diffusion/code/THIRD_PARTY_NOTICES.md` и `diffusion/code/src/**`.
Оба юридических текста включаются штатным `license-files`. Общая Docker-stage
`project-wheel-builder` использует `/opt/envs/core/bin/python` и команду:

```bash
python -m pip wheel --disable-pip-version-check --no-build-isolation \
  --no-deps --wheel-dir /tmp/project-wheel .
```

Она обязана получить ровно один `novel_view_pipeline-0.1.0-py3-none-any.whl`.
Все три prefix напрямую получают один и тот же результат через
`COPY --from=project-wheel-builder` и устанавливают его `pip install --no-deps`
в `/opt/envs/core`, `/opt/envs/gen3c` и `/opt/envs/vggt`. Проверяются версия,
место импорта и `pip check`; отдельный экспорт на host и ожидаемый digest
не нужны.

Production `COPY` допускает только соответствующие locks, перечисленный
состав исходников проекта, проверенные wheels, готовые prefix и два runtime
source tree, а также дополнительные лицензионные тексты и сохранённые
Conda-рецепты из соответствующих стадий сборки. Помощник
`copy_conda_metadata.py` остаётся только в builder. Широкий `COPY .`
остаётся исключительно у `cpu-test`; tests,
jobs, logs, plans, launcher, `.git`, wheelhouse, данные и модели в
production targets не копируются.

## Изменение locks

Изменение версий lock принимается после предметного разбора зависимостей,
сборки нужных source wheels, чистой установки, `pip check`, нужных холодных
импортов и обновления фактического inventory. Упрощение от 5 сентября не
меняет ни одной версии. Исторические инструкции с editable install, host
`PYTHONPATH`, локальными Conda-окружениями или DA3 не являются production
recipe.
