# Runtime

Пакет содержит общую процессную границу между `novel-core` и изолированными
Python-окружениями моделей.

Новый launcher-путь также использует `context.py` как единственного
владельца фиксированных container roots и `presets.py` как владельца
доказанных process topology. `cpu_test` не получает GPU,
`euvs_source_views/v1` использует literal `inference_cp1`, а
`euvs_generation/v1–v2` принимают `inference_cp1` или `inference_cp2` с одной
launcher-assigned CP-группой. CP4 добавляется только вместе с конкретным
training workflow.

`executables.py` содержит только буквальные container-пути Python для core,
Gen3C и VGGT. В нём нет поиска по `PATH`, fallback, registry или preflight.

`context.py` также хранит уже выданные launcher-ом run/attempt ID,
attempt-root, container name, image metadata и необязательный opaque
checkpoint. Host `run/resume` теперь передаёт эти значения в один
attempt-container; resume не проверяет checkpoint и восстанавливает job из
предыдущей operational записи. Это только неизменяемые данные без
`validate()` и проверок файловой системы. `process.py` запускает одноразовый
worker отдельной process group, очищает пользовательские Python-пути и
токены, пишет прямо в выбранный caller-ом log и ждёт естественного выхода
без computation timeout. Тот же файл пересылает SIGTERM и сообщает runner о
штатном stop; он не содержит Docker или model policy. Host-команда `stop` теперь
явно вызывает эту ветку через Docker PID 1; stubborn synthetic tree после
grace завершается Docker SIGKILL без ложной финальной записи Python.

- `distributed.py` строит закреплённую single-node torchrun-команду поверх
  no-timeout `process.py`. Новый Gen3C parent session использует именно эту
  границу без project-source path и computation deadline.
  `_parent_guard.py` на Linux связывает launcher с жизнью создающего
  controller: при его аварийном исчезновении `torchrun` получает `SIGTERM`
  и штатно завершает локальные rank. Модельная семантика и протокол запросов
  при этом остаются в предметном адаптере, а не в общем runtime.

Worker наследует системные, CUDA- и прочие переменные основного процесса,
но не получает пользовательские `PYTHONHOME`, `PYTHONPATH` и известные
Hugging Face-токены. Поверх этого можно явно заменить только небольшой
список несекретных CUDA/cache/offline-переменных. Поддерживаемый Gen3C
сценарий направляет XDG, Cosmos, Hugging Face, Torch, TorchInductor, Triton,
CUDA и Warp в подпапки фиксированного `/cache`, а TMP и уникальный scratch —
туда же, не меняя глобальное окружение процесса.

Общий слой не содержит логики EUVS, выбора модели, checkpoint или научной
семантики массивов. Его узкий список разрешённых переменных может включать
проверенные несекретные cache-настройки конкретных worker; путь и назначение
задаёт предметный адаптер. Подготовка RGB, камер, depth и результатов
остаётся в соответствующих модельных пакетах.
