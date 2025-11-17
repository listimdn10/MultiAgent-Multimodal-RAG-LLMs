# -*- coding: utf-8 -*-
import os
import json
from crewai import Crew, Process, Task

# Import Agent và Tool từ file của bạn
from embedding_agent import embedding_agent, embedding_tool

# ==============================================================================
# 1. TẠO GIẢ DỮ LIỆU INPUT (rag_output.json)
# ==============================================================================
print("🛠️ Đang tạo file giả lập rag_output.json...")

mock_rag_data = {
    "type": "vuln_analysis_result",
    "vuln_type": "Reentrancy",
    "summary": "The contract contains a reentrancy vulnerability in the withdraw function.",
    "description": "State changes happen after the external call.",
    "recommendation": "Use Checks-Effects-Interactions pattern.",
    "functional_semantic": "This contract allows users to deposit and withdraw funds.",
    
    # Code mẫu đơn giản để test biên dịch và sinh CFG
    "code": """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;

    contract TestWallet {
        mapping(address => uint256) public balances;

        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }

        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            (bool sent, ) = msg.sender.call{value: amount}("");
            require(sent, "Failed to send Ether");
            balances[msg.sender] -= amount;
        }
    }
    """
}

with open("rag_output.json", "w", encoding="utf-8") as f:
    json.dump(mock_rag_data, f, indent=2, ensure_ascii=False)

print("✅ Đã tạo rag_output.json")

# ==============================================================================
# 2. KIỂM TRA CÁC FILE PHỤ THUỘC
# ==============================================================================
# Kiểm tra xem EtherSolve.jar có ở cùng thư mục không (vì code gọi java -jar EtherSolve.jar)
if not os.path.exists("EtherSolve.jar"):
    print("⚠️ CẢNH BÁO: Không thấy file 'EtherSolve.jar' tại thư mục này.")
    print("   Code có thể bị lỗi ở bước extract_cfg_embedding.")
    print("   Vui lòng copy EtherSolve.jar vào đây trước khi chạy tiếp.")
    # Bạn có thể comment dòng dưới nếu muốn test lỗi luôn
    # exit(1) 

# ==============================================================================
# 3. ĐỊNH NGHĨA TASK RIÊNG CHO TEST
# ==============================================================================
# Chúng ta định nghĩa lại Task ở đây để đảm bảo đúng input rỗng
test_task = Task(
    name="test_embedding_task",
    description="Test sinh embedding từ rag_output.json giả lập.",
    expected_output="File parser_output.json được tạo thành công.",
    agent=embedding_agent,
)

# ==============================================================================
# 4. CHẠY CREW VỚI 1 AGENT
# ==============================================================================
print("\n🚀 Bắt đầu chạy Embedding Agent...")

crew = Crew(
    agents=[embedding_agent],
    tasks=[test_task],
    process=Process.sequential,
    verbose=True
)

try:
    result = crew.kickoff()
    print("\n################################################")
    print("✅ KẾT QUẢ TEST:")
    print(result)
    
    # Kiểm tra file output
    if os.path.exists("parser_output.json"):
        print("\n📂 Kiểm tra file output:")
        with open("parser_output.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            keys = data.keys()
            print(f"   - File parser_output.json tồn tại.")
            print(f"   - Các keys tìm thấy: {list(keys)}")
            
            # Check sơ bộ dữ liệu
            if data.get('cfg_embeddings'):
                print(f"   - CFG Embeddings: OK (Len: {len(data['cfg_embeddings'])})")
            else:
                print("   - ⚠️ CFG Embeddings rỗng!")
    else:
        print("\n❌ Lỗi: File parser_output.json chưa được tạo!")

except Exception as e:
    print(f"\n❌ Lỗi Runtime: {e}")