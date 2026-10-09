# Постоянный сервер на Oracle Cloud Free — пошагово

Сервер нужен для того, что GitHub Actions сделать не может: постоянные WebSocket-соединения с биржами,
прогнозы на секунды и минуты в реальном времени, публикация данных сайта каждые 2 минуты.
Сайт остаётся тем же: https://8gv457b4ht-arch.github.io/ai-market-predictor/

Стоимость: 0 (Always Free). При регистрации Oracle просит банковскую карту для проверки личности;
списаний на бесплатном тарифе нет, если не переходить на платный (Upgrade) сами.

## 1. Регистрация (≈10 минут)
1. https://signup.cloud.oracle.com → заполните форму.
2. **Home Region — выберите Европу: Germany Central (Frankfurt) или Netherlands Northwest (Amsterdam).**
   Регион потом не меняется. Из США Binance и Bybit блокируют доступ — поэтому не США.
3. Подтвердите почту и карту.

## 2. Создать сервер (≈5 минут)
1. Меню ☰ → Compute → Instances → **Create instance**.
2. Image: **Canonical Ubuntu 24.04** (или 22.04).
3. Shape → Ampere → **VM.Standard.A1.Flex**, 2 OCPU, 12 GB памяти (входит в Always Free).
   Если пишет «Out of capacity» — попробуйте позже или выберите **VM.Standard.E2.1.Micro** (тоже бесплатно, слабее;
   скрипт сам добавит swap).
4. Add SSH keys → **Generate a key pair for me** → **Save private key** (файл понадобится для входа).
5. **Create**. Через 1–2 минуты появится Public IP address.

Открывать порты не нужно: сервер сам подключается к биржам и к GitHub, входящих подключений нет.

## 3. Токен GitHub для публикации данных сайта
GitHub → ваш аватар → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token:
- Repository access: **Only select repositories** → `ai-market-predictor`;
- Permissions → Repository permissions → **Contents: Read and write**;
- Expiration — например, 1 год. Скопируйте токен (показывается один раз).

Токен хранится только в файле `.env` на сервере (права 600). В репозиторий и на сайт он не попадает.

## 4. Установка одной командой
Подключитесь к серверу:
- Windows (PowerShell) / macOS / Linux: `ssh -i путь/к/ключу.key ubuntu@ПУБЛИЧНЫЙ_IP`
- или в консоли Oracle: на странице сервера → **Cloud Shell** / **Console connection**.

Выполните:
```bash
curl -fsSL https://raw.githubusercontent.com/8gv457b4ht-arch/ai-market-predictor/main/deploy/oracle/install.sh -o install.sh
sudo bash install.sh
```
Скрипт спросит токен из шага 3 (ввод скрыт) и, по желанию, тему ntfy для уведомлений на телефон.
Он продолжит с текущего журнала и моделей (берёт их из ветки `data`), соберёт и запустит службы.

## 5. Последний шаг на GitHub
Settings → Secrets and variables → Actions → вкладка **Variables** → New repository variable:
`BACKEND_MODE` = `external`. После этого расписание GitHub перестаёт писать данные (CI и сайт работают как раньше).

## Проверка и обслуживание
```bash
cd /opt/ai-market-predictor
docker compose ps                                # все службы должны быть Up (healthy)
curl -s http://127.0.0.1:8000/api/health         # состояние API и базы
docker compose logs -f --tail 50 forecaster      # прогнозы по горизонтам
sudo bash deploy/oracle/install.sh               # обновление до новой версии кода
```
Службы перезапускаются сами после сбоя (`restart: unless-stopped` + healthcheck) и после перезагрузки сервера.
На сайте статус «API» станет LIVE, когда данные сервера свежие.

Вернуться на GitHub Actions: удалите переменную `BACKEND_MODE` (или поставьте `scheduled`) и остановите сервер
`docker compose down` — расписание продолжит с последнего опубликованного сервером состояния.
