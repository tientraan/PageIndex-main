import os
from dotenv import load_dotenv
from google import genai

print("Đang load .env...")
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key có tồn tại:", bool(api_key))

client = genai.Client(
    api_key=api_key
)

print("Đang gọi Gemini...")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Trả lời đúng 1 câu: Xin chào Việt Nam"
)

print("Kết quả:")
print(response.text)