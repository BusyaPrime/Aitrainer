<div align="center">

# AI Personal Trainer
### Real-time pose tracking + rep counting (WebRTC · MediaPipe Pose · OpenCV)

<!-- Tech badges (pills) -->
<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img alt="WebRTC" src="https://img.shields.io/badge/WebRTC-RealTime-333333?style=for-the-badge&logo=webrtc&logoColor=white" />
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-Pose-0B5FFF?style=for-the-badge&logo=google&logoColor=white" />
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Math-013243?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

<!-- Platform badges -->
<p>
  <img alt="Runs on" src="https://img.shields.io/badge/Platform-Web%20%2F%20Mobile-222222?style=for-the-badge&logo=googlechrome&logoColor=white" />
  <img alt="Deploy" src="https://img.shields.io/badge/Deploy-Streamlit%20Cloud-00A67E?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

</div>

---

## О проекте

**AI Personal Trainer** — веб-приложение, которое:
- получает видеопоток **из камеры пользователя прямо в браузере** через **WebRTC** (работает на телефоне и ПК),
- извлекает **pose landmarks** через **MediaPipe Pose**,
- считает углы суставов (геометрия через `atan2`) и отслеживает фазу движения,
- считает повторения через **state machine** (чтобы не ловить шум),
- рисует **скелет**, **progress bar**, **rep counter** и предупреждения поверх видео.

---

## Возможности

- ✅ **Bicep Curl** (левая/правая рука)
- ✅ **Squat** (по среднему углу двух коленей)
- ✅ визуальный overlay:
  - скелет (цвет по состоянию)
  - счётчик повторов
  - прогресс-бар движения
  - предупреждения “тело не найдено/часть тела не в кадре”
- ✅ **Reset Counter**
- ✅ cloud-ready: `packages.txt` + `runtime.txt` для Streamlit Cloud

---

## Tech Stack

- **Streamlit** — UI
- **streamlit-webrtc** — WebRTC-камера в браузере
- **MediaPipe Pose** — модель позы и landmarks
- **OpenCV** — рисование overlay и визуализация
- **NumPy** — математика углов/интерполяция

---

## Архитектура и пайплайн

**Pipeline (кадры в реальном времени):**
1. Браузер отправляет кадры через WebRTC
2. `TrainerVideoProcessor.recv(frame)` принимает `av.VideoFrame`
3. `PoseDetector`:
   - прогоняет кадр через MediaPipe Pose
   - отдаёт landmarks в пикселях
4. `calculate_angle(a,b,c)` считает суставной угол
5. State machine фиксирует фазу (`UP/DOWN`) и считает повторы
6. OpenCV рисует UI поверх кадра
7. Возврат обработанного кадра обратно в WebRTC stream

**Почему WebRTC важен:** если использовать `cv2.VideoCapture()` на сервере, в облаке нет “вашей” камеры. WebRTC решает это корректно — камера всегда клиентская.

---

## Математика угла (core logic)

Функция `calculate_angle(a, b, c)` считает угол **ABC** по трём точкам `a, b, c`:

- Строим два вектора: **BA** и **BC**
- Берём разность направлений через `atan2`:

`angle = |atan2(c.y-b.y, c.x-b.x) - atan2(a.y-b.y, a.x-b.x)|`

- Переводим радианы → градусы
- Нормализуем диапазон до **[0..180]**

Это даёт предсказуемое поведение для суставов и не зависит от того, “куда смотрит” человек в кадре.

---

## Подсчёт повторений (state machine)

### Bicep Curl (локоть)
**Пороговые значения:**
- `DOWN`, если угол локтя **> 160°**
- `UP +1 rep`, если угол **< 30°** и до этого был `DOWN`

> Логика специально stateful: иначе на шумных кадрах репы будут “накручиваться”.

### Squat (колено)
**Пороговые значения:**
- `UP` (стоим), если угол **> 170°**
- `DOWN +1 rep`, если угол **< 90°** и до этого был `UP`

Для приседа используется **средний угол двух ног**, если обе видимы.

---

## Landmarks (какие точки используются)

<details>
<summary><b>Показать индексы MediaPipe Pose</b></summary>

### Bicep Curl
- Left: shoulder **11**, elbow **13**, wrist **15**
- Right: shoulder **12**, elbow **14**, wrist **16**

### Squat
- Left leg: hip **23**, knee **25**, ankle **27**
- Right leg: hip **24**, knee **26**, ankle **28**
</details>

---

## Структура репозитория

- `app.py` — приложение (UI + realtime processing)
- `requirements.txt` — python зависимости
- `packages.txt` — системные пакеты для Streamlit Cloud (для OpenCV)
- `runtime.txt` — версия Python для Streamlit Cloud
- `README.md` — этот файл

---

## Запуск локально

```bash
pip install -r requirements.txt
streamlit run app.py
