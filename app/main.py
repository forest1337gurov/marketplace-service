"""ML-сервис: по заказу маркетплейса оценивает риск негативного отзыва.

Модель обучена заранее в ноутбуке и лежит в models/model.pkl. Сервис грузит её
ОДИН раз на старте и на каждый запрос только применяет — никакого обучения (.fit)
внутри обработчиков нет. Эндпоинты:
    GET  /health   — жив ли сервис;
    POST /predict  — принять заказ (PredictRequest) и вернуть оценку (PredictResponse).
"""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib                # загрузка обученной модели из .pkl
import pandas as pd          # модель ждёт на вход pandas DataFrame
from fastapi import FastAPI, Request

from app.schemas.predict import PredictRequest, PredictResponse

MODEL_PATH = Path("models/model.pkl")

# --- Загрузка модели на уровне модуля = ОДИН раз при старте сервиса. ---
# bundle собран в обучающем ноутбуке и хранит всё, что нужно для предсказания:
# саму модель, порог отсечения, порядок признаков и список категорий штатов.
_bundle = joblib.load(MODEL_PATH)
_MODEL = _bundle["model"]                        # обученный LGBMClassifier
_THRESHOLD = float(_bundle["threshold"])         # выше порога — флаг «высокий риск»
_FEATURES = _bundle["features"]                  # порядок колонок, в котором училась модель
_STATE_CATEGORIES = _bundle["state_categories"]  # 27 штатов в порядке обучения


def _featurize(req: PredictRequest) -> pd.DataFrame:
    """Собрать из запроса одну строку признаков РОВНО как при обучении.

    Любое расхождение с обучающим ноутбуком (другой порядок колонок, иначе
    посчитанный freight_ratio, пересозданные категории штата) даёт train/serve
    skew — модель молча начинает предсказывать ерунду. Поэтому формулы и порядок
    здесь копируют ноутбук один в один.
    """
    row = {
        "total_price": req.price,
        "total_freight": req.freight,
        # freight_ratio — та же формула, что в ноутбуке; max(.., 0.01) страхует от деления на 0
        "freight_ratio": req.freight / max(req.price, 0.01),
        "installments": req.installments,
        "n_items": req.n_items,
        "promised_lead_time": req.promised_lead_time_days,
        "purchase_month": req.purchase_month,
        "customer_state": req.customer_state,
    }
    # [_FEATURES] выстраивает колонки в тот же порядок, что видела модель на обучении
    df = pd.DataFrame([row])[_FEATURES]
    # Категории штата берём из обучения, а не из текущих данных: иначе внутренние коды
    # категорий LightGBM разъедутся с обучением и предсказание станет неверным.
    df["customer_state"] = pd.Categorical(df["customer_state"], categories=_STATE_CATEGORIES)
    return df


def _score(req: PredictRequest) -> PredictResponse:
    """Применить модель к одному заказу и собрать ответ.

    Единый источник правды для предсказания: и REST-эндпоинт /predict, и Gradio-форма
    (её добавим на шаге 6) зовут именно эту функцию — логика оценки нигде не дублируется.
    """
    # predict_proba(...)[0, 1]: [0] — единственная строка, [1] — вероятность класса 1 (негатив)
    proba = float(_MODEL.predict_proba(_featurize(req))[0, 1])
    return PredictResponse(
        negative_review_probability=round(proba, 4),
        is_high_risk=proba >= _THRESHOLD,   # сравнение с порогом из бандла
        threshold=round(_THRESHOLD, 4),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Код запуска и остановки приложения.

    Всё до yield выполняется один раз при старте сервиса, всё после — при остановке.
    Сюда складывают тяжёлые ресурсы на всё время работы. Модель уже загружена на
    уровне модуля выше; здесь для наглядности кладём ссылку на неё в app.state.
    """
    app.state.model = _MODEL
    yield


# lifespan=lifespan подключает функцию выше как код старта/остановки приложения
app = FastAPI(title="Marketplace satisfaction service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Жив ли сервис — по этому эндпоинту его пингуют nginx, Docker и CI."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """Принять заказ и вернуть оценку риска негативного отзыва.

    FastAPI уже проверил вход по схеме PredictRequest (иначе вернул бы 422), так что
    сюда приходит гарантированно валидный заказ. response_model=PredictResponse
    гарантирует, что и ответ будет ровно оговорённой формы.
    """
    return _score(req)