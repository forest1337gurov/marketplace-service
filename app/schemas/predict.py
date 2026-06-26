"""Pydantic-схемы сервиса: контракт входа (PredictRequest) и выхода (PredictResponse).

Это единственное место, где описано, какие поля заказа сервис принимает, в каких
границах они допустимы и что именно он возвращает. FastAPI использует эти классы,
чтобы автоматически разобрать входной JSON, проверить его и нарисовать /docs.
"""
from pydantic import BaseModel, Field, field_validator

# 27 штатов Бразилии — ровно те категории customer_state, что видела модель на обучении.
# Вынесены отдельной константой, потому что список нужен и валидатору ниже,
# и выпадающему списку Gradio-формы на шаге 6.
BR_STATES = frozenset({
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
})


class PredictRequest(BaseModel):
    """Один заказ маркетплейса в момент оформления — вход эндпоинта /predict.

    Каждое поле повторяет признак, на котором училась модель, с теми же границами.
    Поля с ограничениями (ge/le) Pydantic проверяет сам; штат — отдельный валидатор ниже.
    FastAPI парсит входной JSON в этот класс: если типы или границы не сходятся,
    клиент получает 422, а в обработчик /predict приходит уже гарантированно валидный объект.
    """

    price: float = Field(..., ge=0, description="Сумма товаров в заказе, R$")
    freight: float = Field(..., ge=0, description="Стоимость доставки, R$")
    installments: int = Field(..., ge=1, le=24, description="Число платежей по рассрочке")
    n_items: int = Field(..., ge=1, description="Количество позиций в заказе")
    customer_state: str = Field(..., description="Штат покупателя, 2 буквы (например, SP)")
    promised_lead_time_days: int = Field(..., ge=0, le=180, description="Дней до обещанной доставки")
    purchase_month: int = Field(..., ge=1, le=12, description="Месяц заказа, 1-12")

    @field_validator("customer_state")
    @classmethod
    def state_must_be_known(cls, value: str) -> str:
        """Привести штат к верхнему регистру и отвергнуть неизвестный.

        Модель знает только 27 бразильских штатов из обучения. Если придёт что-то
        другое (`ZZ`, пустая строка), поднимаем ValueError — FastAPI превратит его
        в ответ 422, и мусорный штат до модели не дойдёт.
        """
        normalized = value.strip().upper()
        if normalized not in BR_STATES:
            raise ValueError(f"unknown customer_state {value!r}")
        return normalized


class PredictResponse(BaseModel):
    """Оценка риска негативного отзыва — выход эндпоинта /predict.

    Контракт ответа: что сервис гарантированно вернёт клиенту. Описав его явно,
    мы и документацию получаем бесплатно, и страхуемся от случайного «не того» ответа.
    """

    negative_review_probability: float = Field(..., ge=0, le=1, description="Вероятность негатива, 0..1")
    is_high_risk: bool = Field(..., description="Превысила ли вероятность порог модели")
    threshold: float = Field(..., description="Порог, по которому выставлен флаг риска")