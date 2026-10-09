# Протокол проверки (08.10.2026, среда сборки)

## Среда
Python 3.13, starlette 1.6.0, uvicorn 0.53.0, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1.
Исходящий доступ ограничен политикой прокси: **Binance, Bybit, OKX, новостные RSS, PyPI, apt и Docker Hub отвечают 403**.
Поэтому `pytest`, `websockets` и базовый Docker-образ здесь установить нельзя.

## 1. Тесты — 48/48 прошли
Те же тестовые файлы, что запускает `pytest` (`docker build --target test .`), выполнены минимальным
pytest-совместимым раннером `scripts/run_tests_without_pytest.py` (plain assert, `tmp_path`, `monkeypatch`),
потому что пакет `pytest` недоступен офлайн. На VPS тесты запускаются настоящим pytest внутри образа.

```
PASSED test_api.py::test_auth_and_actions
PASSED test_api.py::test_monitoring_and_dashboard
PASSED test_api.py::test_rate_limit
PASSED test_api.py::test_validation_errors
PASSED test_deploy_config.py::test_compose_services_restart_health_volumes
PASSED test_deploy_config.py::test_dockerfile_runs_as_non_root_and_has_test_stage
PASSED test_deploy_config.py::test_env_example_documents_every_setting
PASSED test_deploy_config.py::test_no_secrets_committed
PASSED test_deploy_config.py::test_no_trading_endpoints_or_private_api_usage
PASSED test_deploy_config.py::test_service_healthcheck_script
PASSED test_exchanges.py::test_binance_klines_parse_closed_flag_and_taker_volume
PASSED test_exchanges.py::test_binance_ws_parsing
PASSED test_exchanges.py::test_bybit_and_okx_klines_parse
PASSED test_exchanges.py::test_bybit_ws_parsing
PASSED test_exchanges.py::test_fetch_history_paginates_backwards
PASSED test_exchanges.py::test_fetch_klines_validates_and_sorts
PASSED test_exchanges.py::test_okx_ws_parsing
PASSED test_exchanges.py::test_symbol_and_interval_mapping
PASSED test_exchanges.py::test_tickers_and_orderbooks
PASSED test_features_ml.py::test_backtest_costs_and_latency
PASSED test_features_ml.py::test_ensemble_outputs_valid_probabilities
PASSED test_features_ml.py::test_features_are_causal_no_lookahead
PASSED test_features_ml.py::test_gate_confidence_threshold
PASSED test_features_ml.py::test_indicator_ranges
PASSED test_features_ml.py::test_labels_and_threshold
PASSED test_features_ml.py::test_multi_timeframe_join_uses_only_closed_higher_bars
PASSED test_features_ml.py::test_regime_priority
PASSED test_features_ml.py::test_walk_forward_does_not_find_signal_in_random_walk
PASSED test_features_ml.py::test_walk_forward_learns_a_real_signal
PASSED test_learning_predictor.py::test_block_bootstrap_probability
PASSED test_learning_predictor.py::test_compare_models_promotes_only_genuine_improvement
PASSED test_learning_predictor.py::test_learning_cycle_baseline_wait_and_challenger
PASSED test_learning_predictor.py::test_no_model_means_no_prediction
PASSED test_learning_predictor.py::test_predictor_ledger_snapshot_gating_and_resolution
PASSED test_market_stream.py::test_aggregator_book_and_resync
PASSED test_market_stream.py::test_aggregator_cvd_flow_bars_duplicates_gaps_invalid
PASSED test_market_stream.py::test_keepalive_is_sent
PASSED test_market_stream.py::test_local_orderbook_snapshot_delta_gap_and_crossed
PASSED test_market_stream.py::test_one_exchange_failure_does_not_stop_others
PASSED test_market_stream.py::test_stream_reconnects_after_errors_and_stale_data
PASSED test_quality_news_backup.py::test_backup_rotation_and_restore
PASSED test_quality_news_backup.py::test_feed_parsing_rss_and_atom
PASSED test_quality_news_backup.py::test_llm_analyzer_parses_and_clamps
PASSED test_quality_news_backup.py::test_news_features_freshness_decay
PASSED test_quality_news_backup.py::test_news_worker_falls_back_to_rules_and_dedupes
PASSED test_quality_news_backup.py::test_quality_exchange_disagreement_spread_and_new_gaps
PASSED test_quality_news_backup.py::test_quality_fresh_vs_stale_and_missing
PASSED test_quality_news_backup.py::test_rule_analyzer_structured_events
48 passed, 0 failed in 105.65s
```

Что покрыто: парсинг REST/WS всех трёх бирж (по формам из документации), локальный стакан (gap, out-of-order, crossed),
агрегатор (CVD, buy/sell volume, дубликаты, gap по trade id, невалидные/будущие тики, продолжение CVD после рестарта),
переподключение WS после ошибки и после «тишины» (stale), изоляция бирж, keep-alive; причинность признаков (изменение
будущего не меняет прошлые признаки), присоединение старших ТФ только по закрытым барам, метки, режимы, gating;
**канарейка утечки**: на случайном блуждании OOS log loss не лучше наивного baseline; на данных с заложенной закономерностью
модель её находит; backtest с комиссиями/спредом/проскальзыванием/латентностью; полный цикл self-learning
(baseline → ожидание без новых данных → challenger на непересекающемся holdout → промоут/отклонение → без повторной попытки
на тех же данных); ledger, снимок рынка, разметка исходов, вето по новостям, NO TRADE при плохих данных и при слабой
уверенности; Data Quality; RSS/Atom, rule-based и LLM-анализатор (с подменённым ответом), fallback на правила,
затухание веса новостей; бэкап/ротация/восстановление; API по настоящему HTTP (auth, 401/429/400, health/ready/status,
action-эндпоинты, CSP); compose/Dockerfile/.env.example; отсутствие торговых и приватных endpoint и секретов.

## 2. compileall — OK
`python -m compileall -q .` — без ошибок.

## 3. docker build — НЕ выполнен до конца
`docker build --target test .` останавливается на `FROM python:3.13-slim`: Docker Hub → 403 в этой среде.
`docker compose config` — конфигурация валидна, 6 сервисов (+ опциональный caddy).
**На VPS:** `./scripts/start.sh` сначала собирает test-стадию образа (в ней ставятся `websockets`, `pytest` и
выполняется весь набор тестов), и только потом поднимает сервисы.

## 4. Сквозной запуск всех сервисов (локально, без Docker)
Запущены api, collector, news, predictor, learner, backup на одной SQLite. Так как биржи недоступны, БД была
предзаполнена **синтетическими** свечами (только для этой проверки; в архив не входит).
- learner: обучил baseline BTC/USDT 15m и ETH/USDT 15m (walk-forward, 3 фолда), модели 1h честно ждут истории;
- predictor: загрузил production-модели, записал прогнозы в ledger со снимком рынка;
- кнопка «Run learning cycle» в дашборде → learner выполнил принудительный цикл → «waiting» (новых данных нет, переобучения нет);
- backup: создал сжатый консистентный бэкап; news: 0/8 лент (403), статус DOWN;
- collector: все 3 биржи в состоянии reconnecting с причиной ошибки, процесс не падает;
- `/api/health` 200, `/api/ready` 200, `/api/status` без ключа 401, с ключом — полный статус;
- дашборд отрисован в headless Chromium (1440 px и 390 px, светлая и тёмная тема): без ошибок в консоли,
  без горизонтального скролла на мобильном, все вкладки графиков и кнопки работают.

## 5. Не проверено (требует VPS с доступом к биржам)
- реальное подключение WebSocket и REST к Binance/Bybit/OKX и длительная работа потока;
- реальные новостные ленты и LLM-провайдер;
- точность на реальных данных и эффективность self-learning (появится в дашборде: OOS-метрики vs baseline, живой ledger, история challenger-ов);
- сборка образа, автоперезапуск контейнеров и восстановление после перезагрузки VPS;
- PostgreSQL-вариант (код написан под совместимый SQL, но не запускался).
