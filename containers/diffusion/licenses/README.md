# Дополнительные лицензионные тексты образа

Точные дополнительные тексты лицензий для установленного образа.
Это документы сборки, не код проверки запуска и не новая система учёта
зависимостей. Conda-тексты и рецепты сохраняются отдельно непосредственно
из пакетов во время сборки, а не копируются вручную в эту папку.

| Файл | Первичный источник, проверен 09.09.2026 |
| --- | --- |
| `antlr4-4.9.3.txt` | https://raw.githubusercontent.com/antlr/antlr4/4.9.3/LICENSE.txt |
| `decord-0.6.0.txt` | https://raw.githubusercontent.com/dmlc/decord/v0.6.0/LICENSE |
| `triton-3.2.0.txt` | https://raw.githubusercontent.com/triton-lang/triton/v3.2.0/LICENSE |
| `modelopt-0.42.0.txt` | https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/0.42.0/LICENSE |
| `GPL-3.0.txt` | https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.0.2/COPYING.GPLv3; общий неизменённый текст GPLv3 для FFmpeg и соответствующих GNU-компонентов |
| `gcc-runtime-exception-3.1.txt` | https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-12.4.0/COPYING.RUNTIME; исключение 3.1 для пакетов с такой лицензионной меткой |

Docker копирует эту папку в `/opt/licenses/distil3d`. Не редактировать
юридические тексты. При изменении относящейся версии обновить соответствующий
текст из первичного источника, не все зависимости сразу. Лицензии включённых
CUDA и других вложенных компонентов рассматриваются отдельно. GPLv3
покрывает бинарную сборку FFmpeg, а не только BSD-обёртку imageio-ffmpeg;
наличие текста ещё не закрывает предоставление соответствующих исходников.
