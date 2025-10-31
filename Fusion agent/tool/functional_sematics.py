from openai import OpenAI
import pandas as pd
import google.generativeai as genai

import time
import os
import random
# # Cấu hình Gemini API
# genai.configure(api_key="AIzaSyCAz4jOl918hOprkoGNrStIcEJhW1ydQLw")
# model = genai.GenerativeModel("gemini-1.5-flash")


# # Cấu hình OpenAI API
# client = OpenAI(api_key="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")  # Thay bằng API Key thật của bạn)  # Thay API key của bạn

# MODEL_NAME = "gpt-3.5-turbo"  # hoặc "gpt-4" nếu bạn có quyền

# Đọc file Excel gốc
input_file = "../solidity_code_with_labels-3k.xlsx"
df = pd.read_excel(input_file)

if "functional_semantics" not in df.columns:
    df["functional_semantics"] = ""

def build_prompt(code):
    return f"""
What is the purpose of the following Solidity code? Please summarize the answer in one sentence starting with:
“Abstract purpose:”.

Then list the key functionalities in the following format:
“Detail behaviors:
1. ...
2. ...
3. ...”

Here is the Solidity code:
{code}
"""


# # === HÀM GỌI GEMINI CÓ RETRY ===
# def safe_generate_content(prompt, retries=3):
#     for attempt in range(retries):
#         try:
#             response = model.generate_content(prompt)
#             return response.text.strip()
#         except Exception as e:
#             print(f"⚠️ Lỗi (lần {attempt+1}): {e}")
#             if "quota" in str(e).lower() or "rate" in str(e).lower():
#                 wait = random.uniform(30, 60)
#                 print(f"⏳ Đang chờ {wait:.1f}s trước khi thử lại...")
#                 time.sleep(wait)
#             else:
#                 break
#     return "ERROR"

# # === XỬ LÝ TỪNG DÒNG ===
# for idx, row in df.iterrows():
#     if pd.notna(row["functional_semantics"]) and row["functional_semantics"].strip() != "":
#         print(f"⚠️ [{idx+1}] Đã có, bỏ qua.")
#         continue

#     code = row["code"]
#     prompt = build_prompt(code)
#     output = safe_generate_content(prompt)
#     df.at[idx, "functional_semantics"] = output

#     print(f"✅ [{idx+1}/{len(df)}] Đã xử lý xong.")

#     # === Chờ giữa các request ===
#     time.sleep(random.uniform(5.0, 8.0))

#     # === Ghi tạm mỗi 10 dòng ===
#     if (idx + 1) % 10 == 0:
#         df.to_excel(input_file, index=False)
#         print("💾 Đã lưu tạm vào file Excel.")

# # === GHI FILE CUỐI CÙNG ===
# df.to_excel(input_file, index=False)
# print(f"✅ Đã ghi hoàn tất vào file: {input_file}")




# # ✅ HÀM GỌI OPENAI CÓ RETRY
# def safe_generate_content(prompt, retries=3):
#     for attempt in range(retries):
#         try:
#             completion = client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[
#                     {"role": "system", "content": "You are a specialist in Smart Contract analyzing. Talk like an expert in Smart Contract."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.5
#             )
#             return completion.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"⚠️ Lỗi (lần {attempt+1}): {e}")
#             if "rate" in str(e).lower() or "quota" in str(e).lower():
#                 wait = random.uniform(30, 60)
#                 print(f"⏳ Đang chờ {wait:.1f}s trước khi thử lại...")
#                 time.sleep(wait)
#             else:
#                 break
#     return "ERROR"

# # ✅ XỬ LÝ TỪNG DÒNG
# for idx, row in df.iterrows():
#     if pd.notna(row["functional_semantics"]) and row["functional_semantics"].strip() != "":
#         print(f"⚠️ [{idx+1}] Đã có, bỏ qua.")
#         continue

#     code = row["code"]
#     prompt = build_prompt(code)
#     output = safe_generate_content(prompt)
#     df.at[idx, "functional_semantics"] = output

#     print(f"✅ [{idx+1}/{len(df)}] Đã xử lý xong.")

#     # ⏳ Chờ giữa các request
#     time.sleep(random.uniform(5.0, 8.0))

#     # 💾 Ghi tạm mỗi 10 dòng
#     if (idx + 1) % 10 == 0:
#         df.to_excel(input_file, index=False)
#         print("💾 Đã lưu tạm vào file Excel.")

# # ✅ GHI FILE CUỐI CÙNG
# df.to_excel(input_file, index=False)
# print(f"✅ Đã ghi hoàn tất vào file: {input_file}")



###OLLAMA

import requests

def safe_generate_content(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma2:9b",  # Model bạn đã pull bằng `ollama run gemma2`
                    "prompt": prompt,
                    "stream": False
                }
            )
            result = response.json()
            return result["response"].strip()
        except Exception as e:
            print(f"⚠️ Lỗi (lần {attempt+1}): {e}")
            time.sleep(random.uniform(5, 10))
    return "ERROR"



for idx, row in df.iterrows():
    if pd.notna(row["functional_semantics"]) and row["functional_semantics"].strip() != "":
        print(f"⚠️ [{idx+1}] Đã có, bỏ qua.")
        continue

    code = row["code"]
    prompt = build_prompt(code)
    output = safe_generate_content(prompt)
    df.at[idx, "functional_semantics"] = output

    print(f"✅ [{idx+1}/{len(df)}] Đã xử lý xong.")
    time.sleep(random.uniform(5.0, 8.0))

    if (idx + 1) % 10 == 0:
        df.to_excel(input_file, index=False)
        print("💾 Đã lưu tạm vào file Excel.")

df.to_excel(input_file, index=False)
print(f"✅ Đã ghi hoàn tất vào file: {input_file}")
