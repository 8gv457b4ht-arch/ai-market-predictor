# Аудит исходного проекта `AI_Market_Predictor_24_7_FINAL.zip` (v0.8 / «FINAL»)

Метод: распакованы и прочитаны все 72 файла, `compileall`, импорт каждого модуля, запуск всех 25 исходных тестов, проверка Docker/Compose.

## Результаты проверки исходника
- `compileall`: синтаксических ошибок нет.
- Импорты: `main.py` (нет `fastapi`), `live/market_stream.py` (нет `websockets`), `live/news_live.py` (нет `openai`) — требуют зависимостей; сами по себе корректны.
- Тесты: 24 из 25 прошли, `test_live_components.py` не собирается без `websockets`. `TEST_RESULT.txt` («5 passed») не соответствовал набору тестов.
- Тесты были поверхностными: не было ни одного теста API, обучения модели, предсказания, self-learning цикла или Docker-конфигурации.

## Функциональные ошибки
| Где | Проблема |
|---|---|
| `main.py` `/api/exchanges` | `SUPPORTED.keys()` у `set` → 500 на каждый запрос |
| `data.py::_bar` | Bybit `"1m"` → `"M"` (месячные свечи вместо минутных) |
| `data.py::_bar` | OKX требует `1H/4H/1D`; `1h/4h/1d` → ошибка API |
| `data.py` | OKX `/market/candles` максимум 300, а API просил 500/1000 |
| `data.py` | Незакрытая последняя свеча использовалась в `/api/predict` как полноценная (look-ahead в живом прогнозе) |
| `live/market_stream.py` | OKX `ws.okx.com:8443`, без keep-alive `ping` (OKX рвёт соединение через 30 с), без stale-детекции |
| `scripts/run_live.py` | Коммит в SQLite на каждый тик, без retention |
| `self_learning.py` | Текущая модель обучена на всей base-выборке, а валидационный блок брался из той же base-выборки → сравнение champion/challenger на данных, которые production уже видела |
| `self_learning.py` | В обучающий буфер попадали только сигналы `UP/DOWN` (NO TRADE никогда не размечались) → смещённая выборка |
| `model.py` | Бинарная цель: «флэт» помечался как DOWN; `CalibratedClassifierCV(cv=3)` — калибровка на перемешанных фолдах |
| `live_eval.py` | `register("baseline", …)` при каждом рестарте перезаписывал реестр |
| `backup_db.py` | `shutil.copy` живой SQLite (может дать повреждённую копию), бэкапилась одна БД из трёх, без ротации |
| `news_live.py` / `.env.example` | Неподтверждённые имена моделей (`gpt-6-astra`, `gpt-5.6-luna`) как значения по умолчанию |

## Декоративный / неподключённый код
`analytics/regime.py`, `analytics/data_quality.py`, `learning/policy.py`, `learning/registry.py`, `learning/metrics.py`, `core/storage.py`, `core/config.py`, `fusion/*`, `orderbook.py` (не использовался стримом), `storage.py::EventStore`, `news/*` (не вызывался ни API, ни воркерами), `live/news_live.py`, `backtest.py` (дубликат), `replay.py` (синтетика), `yfinance`/`python-multipart` в зависимостях — нигде не использовались.

## Деплой и безопасность
- **Self-learning в Docker не работал:** сервис `live` писал тики в `live_market.sqlite3`; `learning` читал предсказания из `live_shadow.sqlite3`, которые создаёт только `run_live_shadow.py` — его нет в compose. Предсказания в деплое не создавались вообще, модель для learning-сервиса не существовала.
- Нет аутентификации, `CORS_ORIGINS=*`, нет rate limit; `/api/health` и `/api/ready` всегда возвращали «ok» без проверок.
- Контейнеры от root; нет ротации логов; нет HTTPS-варианта.
- Фронтенд: кнопки «Train model»/«Analyze now» запускали синхронное обучение в HTTP-запросе; ошибки через `alert`; Plotly с внешнего CDN.

## Что сделано
Проект пересобран (см. README): все пункты выше исправлены, декоративный код удалён или заменён рабочими модулями, каждая функция покрыта тестом. Подробности проверки — `TEST_RESULTS.md`.
