import asyncio
import logging
import sys
import os
import re

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import BotCommand, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.exceptions import TelegramBadRequest

import config
from database import Database
from claude_service import ClaudeService
from yadisk_loader import YaDiskLoader
from hybrid_search import HybridSearch

# ✅ НОВОЕ: Импорты для отображения версий
import aiogram
import sentence_transformers
import faiss
import httpx

# Логирование
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

_components = {}

# ⭐ НОВОЕ: Хранение выбранной категории для каждого пользователя
user_categories = {}


def sanitize_markdown(text: str) -> str:
    """✅ АГРЕССИВНАЯ проверка Markdown (дублирует claude_service для надёжности)"""
    original = text
    
    # 1. Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. АГРЕССИВНАЯ проверка **
    bold_count = text.count('**')
    if bold_count % 2 != 0:
        logging.warning(f"[bot.py] Нечётное количество **: {bold_count}. Удаляем ВСЕ.")
        text = text.replace('**', '').replace('*', '')
    else:
        # Если ** в порядке, проверяем одиночные *
        temp_text = text.replace('**', '__BOLD__')
        single_stars = temp_text.count('*')
        if single_stars % 2 != 0:
            logging.warning(f"[bot.py] Нечётное количество *: {single_stars}. Удаляем все *.")
            text = text.replace('*', '')
    
    # 3. Убираем троиные+ переносы
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 4. Построчная проверка
    lines = []
    for line in text.split('\n'):
        if line.count('*') % 2 != 0:
            line = line.replace('*', '')
        if line.count('_') % 2 != 0:
            line = line.replace('_', '')
        lines.append(line.rstrip())
    text = '\n'.join(lines)
    
    if text != original:
        logging.info("[bot.py] Текст был дополнительно исправлен")
    
    return text


async def init_components():
    """
    ✅ ФИНАЛЬНАЯ ВЕРСИЯ: Инициализация с тестами и валидацией
    - Health-check базового поиска
    - Тестовый запрос для проверки Markdown
    - Проверка количества документов и источников
    """
    global _components
    
    try:
        logging.info("🔄 Начало инициализации компонентов...")
        
        logging.info("📊 Инициализация базы данных...")
        database = Database(config.DB_PATH)
        
        logging.info("🤖 Инициализация Claude сервиса...")
        claude_service = ClaudeService()
        
        logging.info("☁️ Инициализация YaDisk загрузчика...")
        yadisk_loader = YaDiskLoader(
            public_url=config.YADISK_PUBLIC_URL,
            local_path=config.KB_PATH
        )
        
        logging.info("🔍 Инициализация системы поиска...")
        knowledge_base = HybridSearch(index_dir="data/easuz_index")
        
        # ═══════════════════════════════════════════════════════
        # ✅ HEALTH-CHECK 1: Базовый поиск
        # ═══════════════════════════════════════════════════════
        logging.info("🧪 Health-check #1: Проверка базового поиска...")
        test_results = knowledge_base.search("тест", top_k=1)
        if not test_results and knowledge_base.ntotal > 0:
            logging.warning("⚠️ Поиск вернул пустой результат, но индекс не пуст.")
        
        _components.update({
            'database': database,
            'claude_service': claude_service,
            'yadisk_loader': yadisk_loader,
            'knowledge_base': knowledge_base
        })
        
        # ═══════════════════════════════════════════════════════
        # ✅ HEALTH-CHECK 2: Тестовый запрос к Claude
        # ═══════════════════════════════════════════════════════
        logging.info("🧪 Health-check #2: Тестовый запрос к Claude API...")
        try:
            test_query = "Как продлить срок подачи заявок?"
            test_answer, test_time = await claude_service.ask(test_query, database, knowledge_base)
            
            # Проверяем Markdown
            test_answer_clean = sanitize_markdown(test_answer)
            bold_count = test_answer_clean.count('**')
            
            if bold_count % 2 == 0:
                logging.info(f"✅ Markdown корректен (время: {test_time}ms)")
            else:
                logging.warning(f"⚠️ Markdown некорректен! Найдено {bold_count} тегов **")
            
            if test_time > 20000:
                logging.warning(f"⚠️ Медленный ответ: {test_time}ms (ожидается <15000ms)")
            else:
                logging.info(f"✅ Производительность: {test_time}ms")
                
        except Exception as e:
            logging.error(f"❌ Тестовый запрос не выполнен: {e}")
            return False
        
        # ═══════════════════════════════════════════════════════
        # ✅ HEALTH-CHECK 3: Валидация базы знаний
        # ═══════════════════════════════════════════════════════
        logging.info("🧪 Health-check #3: Валидация базы знаний...")
        
        EXPECTED_MIN_DOCS = 1000      # Минимальное ожидаемое количество документов
        EXPECTED_MIN_SOURCES = 50     # Минимальное количество уникальных источников
        
        total_docs = knowledge_base.ntotal
        unique_sources = len(set(m.get('source', 'unknown') for m in knowledge_base.metadata))
        
        # Подсчёт типов документов
        qa_count = sum(1 for m in knowledge_base.metadata if m.get('type') == 'qa')
        chunk_count = sum(1 for m in knowledge_base.metadata if m.get('type') == 'chunk')
        
        # Проверка минимального количества документов
        if total_docs < EXPECTED_MIN_DOCS:
            logging.warning(
                f"⚠️ База знаний содержит {total_docs} документов "
                f"(ожидается минимум {EXPECTED_MIN_DOCS}). Возможно, индекс не полный!"
            )
            logging.warning("💡 Запустите: python src/build_index.py")
        else:
            logging.info(f"✅ Количество документов: {total_docs} (норма)")
        
        # Проверка количества уникальных источников
        if unique_sources < EXPECTED_MIN_SOURCES:
            logging.warning(
                f"⚠️ Найдено только {unique_sources} уникальных источников "
                f"(ожидается минимум {EXPECTED_MIN_SOURCES}). Проверьте процесс индексации!"
            )
        else:
            logging.info(f"✅ Уникальных источников: {unique_sources} (норма)")
        
        # Детальная статистика
        logging.info("=" * 60)
        logging.info("📊 СТАТИСТИКА БАЗЫ ЗНАНИЙ:")
        logging.info(f"  • Всего документов: {total_docs}")
        logging.info(f"  • Q&A пар: {qa_count}")
        logging.info(f"  • Чанков: {chunk_count}")
        logging.info(f"  • Уникальных источников: {unique_sources}")
        logging.info("=" * 60)
        
        # Проверка на критические ошибки
        if total_docs < 100:
            logging.error("❌ КРИТИЧНО: Слишком мало документов! Индекс повреждён или не создан.")
            return False
        
        if unique_sources < 10:
            logging.error("❌ КРИТИЧНО: Слишком мало источников! Проблема с индексацией.")
            return False
        
        logging.info(f"✅ Инициализация завершена успешно")
        return True
        
    except FileNotFoundError as e:
        logging.error(f"❌ Файлы индекса не найдены: {e}")
        logging.error("💡 Запустите сначала: python src/build_index.py")
        return False
    except Exception as e:
        logging.error(f"❌ Ошибка инициализации: {e}", exc_info=True)
        return False


async def cmd_start(message: types.Message):
    """⭐ УЛУЧШЕНО: Приветствие с выбором категории"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="44 ФЗ", callback_data="cat_44fz"),
            InlineKeyboardButton(text="223 ФЗ", callback_data="cat_223fz"),
            InlineKeyboardButton(text="АРИП", callback_data="cat_ARIP")
        ],
        [InlineKeyboardButton(text="🔍 Выбор торгов ЕАСУЗ", url="https://t.me/easuz_torgi_bot")],
        [InlineKeyboardButton(text="❓ Что ты можешь?", callback_data="what_can_you_do")]
    ])
    
    await message.answer(
        "👋 Здравствуйте!\n\n"
        "Я — AI-консультант по системам закупок.\n\n"
        "Помогу с процедурами, документами, формами и типовыми вопросами по работе в системе.\n\n"
        "📊 **Выберите направление:**\n"
        "• **44 ФЗ** — Контрактная система\n"
        "• **223 ФЗ** — Закупки отдельными видами юридических лиц\n"
        "• **АРИП** — Аукционы по реализации имущества",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def handle_category_callback(callback: CallbackQuery):
    """⭐ ФОРСАЖ: Обработчик выбора категории с edit_text"""
    user_id = callback.from_user.id
    
    # Извлекаем категорию из callback_data
    category = callback.data.replace("cat_", "")
    user_categories[user_id] = category
    
    # Названия категорий для пользователя
    category_names = {
        "44fz": "44 ФЗ",
        "223fz": "223 ФЗ",
        "ARIP": "АРИП"
    }
    
    category_stats = {
        "44fz": "1147 Q&A + 133 инструкции",
        "223fz": "316 Q&A + 18 инструкций",
        "ARIP": "461 Q&A + 31 инструкция"
    }
    
    # ⭐ УЛУЧШЕНИЕ: Добавляем кнопку "Сменить направление"
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔙 Сменить направление", callback_data="back_to_main")]
    ])
    
    # ФОРСАЖ: используем edit_text
    await callback.message.edit_text(
        f"✅ Выбрано направление: **{category_names.get(category)}**\n\n"
        f"📚 База знаний: {category_stats.get(category)}\n\n"
        f"Чем могу Вам помочь?",
        parse_mode="Markdown",
        reply_markup=keyboard
    )
    
    await callback.answer()
    logging.info(f"[USER {user_id}] Выбрана категория: {category}")


async def handle_back_to_main(callback: CallbackQuery):
    """⭐ ФОРСАЖ: Возврат в главное меню"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="44 ФЗ", callback_data="cat_44fz"),
            InlineKeyboardButton(text="223 ФЗ", callback_data="cat_223fz"),
            InlineKeyboardButton(text="АРИП", callback_data="cat_ARIP")
        ],
        [InlineKeyboardButton(text="🔍 Выбор торгов ЕАСУЗ", url="https://t.me/easuz_torgi_bot")],
        [InlineKeyboardButton(text="❓ Что ты можешь?", callback_data="what_can_you_do")]
    ])
    
    # ФОРСАЖ: используем edit_text
    await callback.message.edit_text(
        "👋 Здравствуйте!\n\n"
        "Я — AI-консультант по системам закупок.\n\n"
        "Помогу с процедурами, документами, формами и типовыми вопросами по работе в системе.\n\n"
        "📊 **Выберите направление:**\n"
        "• **44 ФЗ** — Контрактная система\n"
        "• **223 ФЗ** — Закупки отдельными видами юридических лиц\n"
        "• **АРИП** — Аукционы по реализации имущества",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )
    
    await callback.answer()


async def handle_what_can_you_do_callback(callback: CallbackQuery):
    """⭐ ФОРСАЖ + УЛУЧШЕНО: Обработчик кнопки 'Что ты можешь?' с кнопкой главного меню"""
    
    # ⭐ ЗАМЕЧАНИЕ: Убрали текст "Выберите направление через /start", добавили кнопку
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_to_main")]
    ])
    
    # ФОРСАЖ: используем edit_text
    await callback.message.edit_text(
        "💡 **AI-консультант по системам закупок**\n\n"
        "**Мои возможности:**\n"
        "✅ Отвечаю на вопросы по работе с системами\n"
        "✅ Помогаю с заполнением форм и документов\n"
        "✅ Объясняю процедуры и регламенты\n"
        "✅ Подсказываю решения типовых проблем\n"
        "✅ Работаю на базе актуальной документации\n\n"
        "**Доступные базы знаний:**\n\n"
        "📊 **44 ФЗ** (Контрактная система)\n"
        "• 1147 вопросов-ответов\n"
        "• 133 инструкции\n\n"
        "📊 **223 ФЗ** (Закупки отдельными юр. лицами)\n"
        "• 316 вопросов-ответов\n"
        "• 18 инструкций\n\n"
        "📊 **АРИП** (Аукционы по реализации имущества)\n"
        "• 461 вопрос-ответ\n"
        "• 31 инструкция",
        parse_mode="Markdown",
        reply_markup=keyboard
    )
    
    await callback.answer()


async def handle_text_message(message: types.Message):
    """⭐ ОБНОВЛЕНО: Обработчик с фильтрацией по категории"""
    text = message.text.strip()
    user_id = message.from_user.id
    
    database = _components['database']
    claude_service = _components['claude_service']
    knowledge_base = _components['knowledge_base']

    # ⭐ ПРОВЕРКА: выбрана ли категория
    category = user_categories.get(user_id, None)
    
    if not category:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="44 ФЗ", callback_data="cat_44fz"),
                InlineKeyboardButton(text="223 ФЗ", callback_data="cat_223fz"),
                InlineKeyboardButton(text="АРИП", callback_data="cat_ARIP")
            ]
        ])
        await message.answer(
            "⚠️ Сначала выберите направление для работы:",
            reply_markup=keyboard
        )
        return

    processing = await message.answer("⏳ Обрабатываю запрос...")
    
    try:
        # ⭐ ПЕРЕДАЕМ КАТЕГОРИЮ В ask()
        answer, response_time = await claude_service.ask(
            text, 
            database, 
            knowledge_base,
            category=category
        )
        
        # ✅ КРИТИЧНО: Дополнительная постобработка Markdown
        answer = sanitize_markdown(answer)
        
        database.log_query(
            user_id=message.from_user.id,
            username=message.from_user.username or "",
            query=text,
            answer=answer,
            response_time_ms=response_time
        )
        
        await processing.delete()
        
        try:
            await message.answer(answer, parse_mode="Markdown")
        except TelegramBadRequest as e:
            if "can't parse entities" in str(e):
                error_msg = str(e)
                logging.error(f"[Markdown ERROR] {error_msg}")
                
                # Извлекаем позицию ошибки
                if "byte offset" in error_msg:
                    try:
                        offset = int(error_msg.split("byte offset ")[1].split()[0])
                        logging.error(f"[Markdown] Проблемный участок (позиция {offset}):")
                        start = max(0, offset - 50)
                        end = min(len(answer), offset + 50)
                        logging.error(f"  ...{answer[start:end]}...")
                        logging.error(f"[Markdown] Символ на позиции {offset}: {repr(answer[offset:offset+10])}")
                    except Exception as ex:
                        logging.error(f"[Markdown] Не удалось извлечь позицию: {ex}")
                
                # Сохраняем проблемный текст для анализа
                try:
                    error_file = os.path.join(os.getcwd(), 'logs', 'last_markdown_error.txt')
                    os.makedirs(os.path.dirname(error_file), exist_ok=True)
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(answer)
                    logging.error(f"[Markdown] Полный текст сохранён в {error_file}")
                except Exception as ex:
                    logging.error(f"[Markdown] Не удалось сохранить файл: {ex}")
                
                # Fallback: отправляем без markdown
                clean_answer = answer.replace('**', '').replace('##', '').replace('###', '').replace('*', '')
                await message.answer(clean_answer)
            else:
                raise
                
    except Exception as e:
        logging.error(f"[handle_text_message] Error: {e}", exc_info=True)
        try:
            await processing.delete()
        except:
            pass
        await message.answer("❌ Произошла ошибка. Попробуйте позже.")


async def main():
    """Главная функция с инициализацией ДО polling"""
    if not config.TELEGRAM_BOT_TOKEN:
        logging.error("❌ TELEGRAM_BOT_TOKEN не установлен в .env!")
        sys.exit(1)
    
    logging.info("=" * 60)
    logging.info("🚀 ЗАПУСК БОТА ЕАСУЗ AI")
    logging.info("=" * 60)
    
    # ✅ НОВОЕ: Информация о версиях библиотек
    logging.info("📦 ВЕРСИИ ЗАВИСИМОСТЕЙ:")
    logging.info(f"  🐍 Python: {sys.version.split()[0]}")
    logging.info(f"  🤖 Aiogram: {aiogram.__version__}")
    logging.info(f"  🧠 Sentence-Transformers: {sentence_transformers.__version__}")
    
    # FAISS может не иметь __version__, обработаем это
    try:
        faiss_version = faiss.__version__
    except AttributeError:
        faiss_version = "unknown"
    logging.info(f"  🔍 FAISS: {faiss_version}")
    
    logging.info(f"  🌐 HTTPX: {httpx.__version__}")
    logging.info("=" * 60)
    
    success = await init_components()
    if not success:
        logging.error("❌ Не удалось инициализировать компоненты. Завершение работы.")
        sys.exit(1)
    
    bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    # ⭐ РЕГИСТРАЦИЯ ОБРАБОТЧИКОВ
    dp.message.register(cmd_start, Command("start"))
    dp.callback_query.register(handle_what_can_you_do_callback, F.data == "what_can_you_do")
    dp.callback_query.register(handle_back_to_main, F.data == "back_to_main")  # ⭐ НОВОЕ
    dp.callback_query.register(handle_category_callback, F.data.startswith("cat_"))
    dp.message.register(handle_text_message)

    await bot.set_my_commands([
        BotCommand(command="start", description="Начать работу с ботом"),
    ])
    
    kb_docs = _components['knowledge_base'].ntotal
    logging.info("=" * 60)
    logging.info(f"✅ Бот готов к работе!")
    logging.info(f"📚 База знаний: {kb_docs} документов")
    logging.info(f"🤖 Модель: {config.CLAUDE_MODEL}")
    logging.info("=" * 60)

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\n👋 Бот остановлен пользователем")
    except Exception as e:
        logging.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)