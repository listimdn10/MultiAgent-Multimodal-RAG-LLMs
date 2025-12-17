import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyCQIVVHLDDMLGEWoO56w4Wi06Jjts4BmxM")

print("Danh sách model bạn có quyền sử dụng:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")