import subprocess
import os
import glob
import sys
import time

def clean_json_files():
    """
    Xóa tất cả các file .json trong thư mục hiện tại để đảm bảo chạy sạch (clean run).
    Giữ lại các file cấu hình nếu cần (ở đây xóa hết theo yêu cầu).
    """
    print("\n🧹 [1/4] DỌN DẸP MÔI TRƯỜNG (CLEAN UP)...")
    
    # Lấy danh sách tất cả file .json
    json_files = glob.glob("*.json")
    
    if not json_files:
        print("   ✅ Không có file JSON nào cần xóa.")
        return

    for file_path in json_files:
        try:
            # Có thể thêm logic if file_path != "config.json" nếu cần giữ file nào đó
            os.remove(file_path)
            print(f"   🗑️  Đã xóa: {file_path}")
        except OSError as e:
            print(f"   ⚠️  Lỗi khi xóa {file_path}: {e}")
    
    print("   ✅ Đã dọn dẹp xong.")

def run_script(script_name):
    """
    Chạy một script python con bằng subprocess.
    Nếu script con lỗi, dừng toàn bộ quy trình.
    """
    print(f"\n{'='*50}")
    print(f"🚀 ĐANG CHẠY: {script_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # Sử dụng sys.executable để đảm bảo dùng đúng python env hiện tại
        result = subprocess.run(
            [sys.executable, script_name], 
            check=True,      # Raise error nếu script con trả về exit code != 0
            text=True        # Capture output dưới dạng text (nếu cần pipe)
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ {script_name} HOÀN THÀNH trong {elapsed:.2f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ LỖI: {script_name} thất bại với mã lỗi {e.returncode}.")
        return False
    except Exception as e:
        print(f"\n❌ LỖI KHÔNG XÁC ĐỊNH khi chạy {script_name}: {e}")
        return False

def main():
    # --- BẮT ĐẦU TÍNH GIỜ TỔNG ---
    total_start_time = time.time()
    
    # Bước 1: Clean
    clean_json_files()

    # Bước 2: Chạy invoke.py (Gemini -> RAG -> Embedding)
    # Output mong đợi: rag_output.json
    if not run_script("invoke.py"):
        print("\n🛑 Quy trình bị dừng do lỗi ở bước invoke.")
        return

    # Bước 3: Chạy crew_run.py (Fusion -> Explainer)
    # Output mong đợi: explainer_output.json
    if not run_script("crew_run.py"):
        print("\n🛑 Quy trình bị dừng do lỗi ở bước crew_run.")
        return

    # Bước 4: Chạy consensus_agent.py (Tổng hợp kết quả)
    # Output mong đợi: consensus_output.json
    if not run_script("consensus_agent.py"):
        print("\n🛑 Quy trình bị dừng do lỗi ở bước consensus.")
        return

    # --- KẾT THÚC TÍNH GIỜ TỔNG ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Chuyển đổi sang phút giây cho dễ đọc nếu chạy lâu
    minutes = int(total_duration // 60)
    seconds = total_duration % 60

    print("\n" + "="*50)
    print("🎉🎉 TOÀN BỘ QUY TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG! 🎉🎉")
    if minutes > 0:
        print(f"⏱️  Tổng thời gian chạy: {minutes} phút {seconds:.2f} giây")
    else:
        print(f"⏱️  Tổng thời gian chạy: {seconds:.2f} giây")
    print("="*50)

if __name__ == "__main__":
    main()