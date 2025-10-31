import os
import pandas as pd

def extract_sol_with_label_to_excel(root_folder, output_excel):
    records = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".sol"):
                file_path = os.path.join(dirpath, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        code = f.read()
                        code = code.replace('\r\n', '\n').replace('\r', '\n').strip()

                        # Lấy tên folder cha làm nhãn (label)
                        label = os.path.basename(os.path.dirname(file_path))

                        records.append({
                            'code': code,
                            'label': label
                        })
                except Exception as e:
                    print(f"Lỗi khi đọc {file_path}: {e}")

    # Tạo DataFrame và xuất ra file Excel
    df = pd.DataFrame(records)
    df.to_excel(output_excel, index=False)

    print(f"✅ Đã tạo file Excel: {output_excel} với {len(records)} file Solidity.")

# === Đường dẫn ===
root_folder = "Dataset_1/Dataset"
output_excel = "solidity_code_with_labels-3k.xlsx"

extract_sol_with_label_to_excel(root_folder, output_excel)
