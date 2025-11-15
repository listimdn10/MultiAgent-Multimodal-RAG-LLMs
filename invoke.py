from rag_agent_defined import rag_agent, rag_task, rag_tool
from rag_agent_setup import (
    gemini_model,
    embeddings
)
from embedding_agent import embedding_tool, embedding_agent, embedding_task
import pprint
import json
from crewai import Agent, Task, Crew, Process

class FunctionalSemantic:
    def __init__(self, gemini_model):
        self.gemini = gemini_model
        self.embeddings = embeddings

    def _make_prompt(self, code: str) -> str:
        return f"""
            You are a concise code summarizer.

            Task: given the Solidity code below, produce a short functional-semantic description
            followed by the code snippet encoded as an escaped string (so that newlines are
            represented as \\n and code block fences are included). Then add a short impact/issue
            description. Your output must follow this exact structure and punctuation (no extra commentary):

            1) A single-paragraph summary (one or two sentences), plain text.
            2) A code block rendered inside the output string but with newlines escaped. Use triple backticks and escaped newlines exactly like this example:
            \\n```\\n<CODE LINES WITH \\n AT LINE END>\\n```\\n
            (example of formatting shown below — follow it exactly)

            3) Another paragraph (one or two sentences) explaining the consequence or security note.

            4) (Optional) Repeat the whole block again (exact duplicate), if the original example shows duplication.

            Example desired output for a sample:
            LSP6 supports relaying of calls using a supplied signature. The encoded message is defined as:\\n```\\n bytes memory encodedMessage = abi.encodePacked( LSP6_VERSION,\\n block.chainid,\\n nonce,\\n msgValue,\\n payload\\n );\\n```\\n\\nThe message doesn't include a gas parameter, which means the relayer can specify any gas amount. If the provided gas is insufficient, the entire transaction will revert. However, if the called contract exhibits different behavior depending on the supplied gas, a relayer (attacker) has control over that behavior.

            Now produce the same-structured output for the following Solidity code. Important: do NOT include extra headings, notes, or metadata — produce only the described three parts (summary, escaped code block, impact paragraph), and nothing else.

            Solidity code:
            {code}
        """

    def analyze(self, code: str):
        if not self.gemini:
            raise ValueError("Gemini model not initialized.")
        try:
            response = self.gemini.generate_content(self._make_prompt(code))
            print("=== [Gemini Output] ===")
            print(response.text)
            return response.text
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return "Error generating functional semantic."

code_to_check = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;
    contract VulnerableWallet {
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
        function getBalance() public view returns (uint256) {
            return balances[msg.sender];
        }
    }
    """
'''
functional_semantic = """
**Abstract purpose:** This code defines a token contract called "ZildFinanceCoin" implementing the ERC20 standard, 
with added functionalities for ownership management, safe math operations, founder allocation with vesting, and 
account freezing/burning mechanisms.

**Detail Behaviors:**
1.  Defines the IERC20 interface with standard token functionalities.
2.  Implements an Ownable contract for managing contract ownership.
3.  Includes a SafeMath library to prevent integer overflow/underflow during arithmetic operations.
4.  Defines the ZildFinanceCoin contract, inheriting from Ownable and implementing IERC20.
5.  Initializes token parameters like name, symbol, decimals, and total supply, and allocates tokens to marketing.
6.  Implements a founder token release mechanism with a cliff and vesting schedule.
7.  Provides functions for token transfer, balance checking, and allowance management (approve/transferFrom).
8.  Adds a burn function for destroying tokens, up to a certain limit.
9.  Includes functions for changing the founder address and setting specific addresses for minter and furnace accounts.
10. Allows the owner to freeze and unfreeze accounts, preventing transfers.
"""
'''

fs = FunctionalSemantic(gemini_model)
functional_semantic = str(fs.analyze(code_to_check))
print("\n--- [FUNCTIONAL SEMANTIC] ---")
print(functional_semantic)

# ============================================================
# 4️⃣ Khởi tạo Crew tuần tự (RAG → Embedding)
# ============================================================

input = {
    "code": code_to_check,
    "functional_semantic": functional_semantic
}
with open("input.json", "w") as f:
    json.dump(input, f, indent=4)

print("✅ Đã lưu vào input.json")

crew = Crew(
    agents=[rag_agent, embedding_agent],
    tasks=[rag_task, embedding_task],
    process=Process.sequential
)
result = crew.kickoff()
print("\n--- [RAW 2 AGENT'S RESULT] ---")
print(result)


