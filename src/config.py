import os
from dotenv import load_dotenv

# Определяем корень проекта (на уровень выше src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')

# Загружаем .env из корня проекта
load_dotenv(dotenv_path=dotenv_path)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip() or None

# Claude AI (legacy - для обратной совместимости)
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '').strip() or None
CLAUDE_ENDPOINT = (os.getenv('CLAUDE_ENDPOINT', 'https://api.vsegpt.ru/v1/chat/completions') or '').strip()
CLAUDE_MODEL = (os.getenv('CLAUDE_MODEL', 'anthropic/claude-sonnet-4.5-1m-thinking') or '').strip()

# GigaChat API
GIGACHAT_AUTH_KEY = os.getenv('GIGACHAT_AUTH_KEY', '').strip() or None
GIGACHAT_SCOPE = (os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS') or '').strip()
GIGACHAT_MODEL = (os.getenv('GIGACHAT_MODEL', 'GigaChat-Max') or '').strip()
GIGACHAT_AUTH_KEY = os.getenv('GIGACHAT_AUTH_KEY', '').strip() or None
GIGACHAT_CLIENT_SECRET = os.getenv('GIGACHAT_CLIENT_SECRET', '').strip() or None  # добавить эту строку

# Яндекс.Диск
YADISK_PUBLIC_URL = (os.getenv('YADISK_PUBLIC_URL', 'https://disk.yandex.ru/d/Iq_KsadHSN9-Gg') or '').strip()

# Администраторы
ADMIN_IDS = [
    int(id.strip()) for id in (os.getenv('ADMIN_IDS', '1501905373') or '').split(',')
    if id.strip().isdigit()
]

# Пути
PERSISTENCE_PATH = os.getenv('PERSISTENCE_PATH', 'data').strip() or 'data'
KB_PATH = os.path.join(PERSISTENCE_PATH, 'knowledge_base')
DB_PATH = os.path.join(PERSISTENCE_PATH, 'bot.db')
LOG_PATH = 'logs/bot.log'

# Логирование
LOG_LEVEL = (os.getenv('LOG_LEVEL', 'INFO') or 'INFO').upper()

# Создание директорий
os.makedirs(KB_PATH, exist_ok=True)
os.makedirs('logs', exist_ok=True)