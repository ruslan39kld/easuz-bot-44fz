# test_api.py
import httpx

API_KEY = "sk-or-vv-7c4c9ea9b7da39922d6b2db888b5293c7ea2762e20bbbf61c8ae4e5b68616424"
ENDPOINT = "https://api.vsegpt.ru/v1/chat/completions"

response = httpx.post(
    ENDPOINT,
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "anthropic/claude-sonnet-4.5-1m-thinking",
        "messages": [{"role": "user", "content": "Привет!"}],
        "temperature": 0.05,
        "max_tokens": 100
    },
    timeout=30
)

print("Статус:", response.status_code)
print("Ответ:", response.text)