# 🛡️ AI-OSINT

**Цифровой анализ информационного поля Казахстана в глобальном медиапространстве**

Конкурс «AI SANA — Digital Kazakhstan: Projects of the Future» | Секция: Социально-гуманитарные науки | 2026

## Методология

- **ABM** (Agent-Based Modeling) — 4 типа информационных агентов
- **Markov Chains** — моделирование состояний нарратива (5 состояний)
- **Monte Carlo** — стохастическое прогнозирование (1000+ итераций)
- **NLP Pipeline** — sentiment analysis, topic modeling, NER
- **OSINT** — данные GDELT, открытые API

## Быстрый деплой на Streamlit Cloud (5 минут)

### Шаг 1: GitHub репозиторий

1. Зайди на [github.com](https://github.com) → **New repository**
2. Имя: `ai-osint` (или любое)
3. **Public** (Streamlit Cloud требует публичный репо на бесплатном тарифе)
4. Загрузи 3 файла: `app.py`, `requirements.txt`, `README.md`

### Шаг 2: Streamlit Cloud

1. Зайди на [share.streamlit.io](https://share.streamlit.io/)
2. Войди через GitHub аккаунт
3. Нажми **New app**
4. Выбери репозиторий `ai-osint`
5. Branch: `main`
6. Main file path: `app.py`
7. Нажми **Deploy**
8. Через 2-3 минуты получишь ссылку вида: `https://ai-osint.streamlit.app`

### Шаг 3: Готово

Ссылка работает — можно показывать комиссии, команде, кому угодно.

## Локальный запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Структура

```
ai-osint/
├── app.py              # Основной файл (всё в одном)
├── requirements.txt    # Зависимости
└── README.md           # Этот файл
```

## Источники данных

- [GDELT Project](https://www.gdeltproject.org/)
- [Stanford Internet Observatory](https://cyber.fsi.stanford.edu/io/publications)
- [Meta Transparency Reports](https://transparency.meta.com/metasecurity/threat-reporting)
- [IRA Troll Tweets (FiveThirtyEight)](https://github.com/fivethirtyeight/russian-troll-tweets)

## Авторы

- КазУМОиМЯ имени Абылай хана
- Программа: 7М02211 — Востоковедение
- Науч. руководитель: к.полит.н., ассоц. проф. Абсаттаров Г.Р.
