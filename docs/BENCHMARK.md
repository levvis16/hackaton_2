# Отчёт по производительности CV-модуля

Среда запуска: **Windows 10 / Python 3.13.5**

| Параметр | Значение |
|---|---|
| Backend инференса | yolo_onnx |
| Файл модели | C:\Users\levvis\OneDrive\Рабочий стол\fastapi_practice\vuz_practice_2\models\yolov8n.onnx |
| Источник видео | media/demo_person_bicycle_car.mp4 |
| Параметры кадра | process_width=640, input_size=640 |
| Длительность замера, с | 25.1 |

## Метрики

| Метрика | Значение |
|---|---|
| FPS (текущий, скользящее окно 10 с) | 8.55 |
| FPS (средний за сессию) | 11.1 |
| Latency (EMA), мс | 111.04 |
| Latency средняя, мс | 100.05 |
| Latency p95, мс | 126.22 |
| Обработано кадров | 279 |
| Пропущено кадров (frame_skip) | 0 |
| Создано событий | 29 |
| Ошибок | 0 |

## События по классам

| Класс | Событий |
|---|---|
| person | 12 |
| obstacle | 5 |
| person_without_helmet | 8 |
| vehicle | 4 |
