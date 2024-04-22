import os
import dill
import pandas as pd
import config
import logging
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode
from telegram import Update


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
path = os.environ.get('PROJECT_PATH', '..')


def load_model():
    models = [file for file in os.listdir(f'{path}/data/models') if file.endswith('.pkl')]
    latest_model = max(models, key=lambda x: x.split('_')[2].split('.')[0])
    with open(f'{path}/data/models/' + latest_model, 'rb') as f:
        model = dill.load(f)
        return model


model = load_model()


async def start(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text='Привет! Я жду строку с данными вида: '
                                                                          '"session_id,client_id,visit_date,...".\n'
                                                                          'Вернусь с предсказаниями🔮\n'
                                                                          '0: клиент НЕ совершит целевое действие\n'
                                                                          '1: клиент совершит целевое действие.',
                                   parse_mode=ParseMode.HTML)


async def predict(update: Update, message, model):
    data = [message.split(',')]
    columns = [
        'session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
        'device_category', 'device_os', 'device_brand', 'device_model',
        'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city'
    ]
    df = pd.DataFrame(data, columns=columns)
    df['visit_number'] = pd.to_numeric(df['visit_number'])
    prediction = model['model'].predict(df)[0]
    pred = str(prediction)
    await update.message.reply_text(f'{data[0][0]}: {pred}')


async def message_handle(update: Update, context: CallbackContext):

    _message = update.message.text

    if not _message:
        await update.message.reply_text("Пустое сообщение. Попробуй отправить еще раз.",
                                        parse_mode=ParseMode.HTML)
        return

    try:
        await predict(update, _message, model)
    except Exception as e:
        error_text = f'Something went wrong during prediction. Reason: {e}'
        logger.error(error_text)
        await update.message.reply_text('Произошла ошибка при попытке предсказания, проверь строку.',
                                        parse_mode=ParseMode.HTML)


def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .build()
    )

    user_filter = filters.ALL

    application.add_handler(
        CommandHandler("start", lambda update, context: start(update, context), filters=user_filter))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))

    application.run_polling()


if __name__ == '__main__':
    run_bot()
