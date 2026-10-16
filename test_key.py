# pip install openai

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from openai import AuthenticationError, RateLimitError, BadRequestError, APIError

# Fix lỗi UnicodeEncodeError trên Windows terminal
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment or .env before running this check.")

client = OpenAI(api_key=API_KEY)

try:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Say OK only.",
        max_output_tokens=16,  # OpenAI yêu cầu tối thiểu 16
    )

    print("[OK] Key dung duoc va con quota/credit.")
    print("Response:", response.output_text)

except AuthenticationError as e:
    print("[SAI KEY] API key sai, bi thu hoi, hoac khong co quyen truy cap.")
    print("Chi tiet:", str(e).encode("utf-8", errors="replace").decode("utf-8"))

except RateLimitError as e:
    msg = str(e).lower()

    if "insufficient_quota" in msg or "quota" in msg or "billing" in msg:
        print("[HET CREDIT] Key hop le nhung het credit/quota hoac chua bat billing API.")
    else:
        print("[RATE LIMIT] Key hop le nhung dang bi rate limit. Thu lai sau.")

    print("Chi tiet:", str(e).encode("utf-8", errors="replace").decode("utf-8"))

except BadRequestError as e:
    print("[BAD REQUEST] Request sai tham so.")
    print("Chi tiet:", str(e).encode("utf-8", errors="replace").decode("utf-8"))

except APIError as e:
    print("[API ERROR] Loi API khac.")
    print("Chi tiet:", str(e).encode("utf-8", errors="replace").decode("utf-8"))

except Exception as e:
    print("[UNKNOWN ERROR] Loi khong xac dinh.")
    print("Chi tiet:", str(e).encode("utf-8", errors="replace").decode("utf-8"))