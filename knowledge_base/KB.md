# Smart Contract Auditor Knowledge Base (Comprehensive)

This document summarizes a wide range of common, critical, and nuanced smart contract vulnerabilities. Use this as a reference when evaluating audit reports.

---

## 1. Critical Core Vulnerabilities

---

### Reentrancy
* **Description:** Occurs when a function makes an external call to an untrusted contract *before* it updates its own state (e.g., balances). The untrusted contract can then call back (re-enter) the original function, bypassing the initial checks and executing the logic multiple times.
* **Impact:** Complete draining of funds from the contract.
* **Prevention:**
    1.  **Checks-Effects-Interactions Pattern:** Strictly follow this order: 1. **Check** (e.g., `require` statements), 2. **Effect** (e.g., update balances), 3. **Interaction** (e.g., make the external call).
    2.  **Reentrancy Guard:** Use a mutex or a `nonReentrant` modifier (like from OpenZeppelin) to lock the function during execution.

---

### Arithmetic (Integer Overflow and Underflow)
* **Description:** An arithmetic operation results in a value that is outside the valid range of the data type (e.S`uint256`). Adding to the maximum value can wrap it to `0` (overflow). Subtracting from `0` can wrap it to the maximum value (underflow).
* **Impact:** Incorrect balance calculations, bypassing security checks (e.g., `require(balance >= amount)`), or minting infinite tokens.
* **Prevention:**
    1.  **Use Solidity 0.8.0+:** The compiler includes built-in overflow and underflow checks that will revert the transaction.
    2.  **Use SafeMath:** For versions before 0.8.0, use a `SafeMath` library for all arithmetic operations.

---

### Improper Access Control
* **Description:** A general category for functions that can be called by unauthorized users. This includes several specific vulnerabilities:
    * **Function Default Visibility:** In old Solidity versions (<0.5.0), functions without a visibility specifier (e.g., `public`, `private`) defaulted to `public`.
    * **State Variable Default Visibility:** State variables have `internal` visibility by default, but can be `public` (creating a getter) or `private`. Public state variables that should be private can leak information.
    * **Unprotected Ether Withdrawal:** A function that sends Ether (e.g., `transfer()`, `.call()`) is `public` and lacks an `onlyOwner` or similar check.
    * **Unprotected SELFDESTRUCT Instruction:** A function containing `selfdestruct` is callable by any user.
    * **Incorrect Constructor Name:** A historical bug (pre-`0.4.22`) where a typo in the constructor name (which was just a function with the same name as the contract) made it a regular, callable `public` function, allowing re-initialization.
* **Impact:** Total contract takeover, loss of all funds, or permanent destruction of the contract.
* **Prevention:**
    1.  Explicitly set visibility for all functions: `external`, `public`, `internal`, or `private`.
    2.  Use modifiers like `onlyOwner` (from OpenZeppelin's `Ownable`) for all administrative or privileged functions.
    3.  Use the `constructor()` keyword (standard in modern Solidity).

---

### Unchecked Low-Level Calls (Unhandled Exceptions / Unchecked-Send)
* **Description:** Low-level functions `.call()`, `.send()`, and `.delegatecall()` do *not* revert the parent transaction if they fail. Instead, they just return a `false` boolean. If the code does not check this return value, it will continue executing as if the call succeeded.
* **Impact:** Failed transactions go unnoticed. A contract might "fail to send" Ether but still update its state as if it succeeded, leading to inconsistent state, locked funds, or DoS.
* **Prevention:**
    1.  **Always check the `success` boolean** returned from a low-level call: `(bool success, ) = target.call{value: 1 ether}(""); require(success, "External call failed");`
    2.  Avoid `.send()` entirely.

---

### Authorization through `tx.origin`
* **Description:** Using the global variable `tx.origin` to check a user's identity. `tx.origin` is the *original wallet* that started the transaction chain, not the *immediate caller* (`msg.sender`). An attacker can trick a user (e.g., the owner) into calling a malicious intermediary contract, which then calls the victim contract. The victim contract sees `tx.origin` as the owner and grants access.
* **Impact:** A malicious contract can impersonate a user and execute privileged actions on their behalf.
* **Prevention:**
    1.  **Never use `tx.origin` for authorization.**
    2.  **Always use `msg.sender`** to identify the immediate caller.

---

## 2. Transaction, Block & Gas Vulnerabilities

---

### Transaction Order Dependence (TOD / Front-Running)
* **Description:** An attacker observes a sensitive transaction (like a DEX trade or revealing a solution) in the public mempool. They copy the transaction and submit their own with a higher gas fee to get it mined first, thereby stealing the opportunity or profit.
* **Impact:** Financial loss for users (e.g., "sandwich attacks" on DEXs), stolen arbitrage opportunities.
* **Prevention:**
    1.  **Commit-Reveal Scheme:** For puzzles, have users first submit a *hash* of their solution (Commit), and then submit the actual solution in a later block (Reveal).
    2.  **Slippage Protection:** For DEXs, enforce a maximum slippage parameter on trades.

---

### Timestamp Dependence (Time Manipulation / Block values as a proxy for time)
* **Description:** Using `block.timestamp` as a trigger for critical state changes (e.g., deciding a winner in a game). Miners (or validators in PoS) have some control over the timestamp and can manipulate it by a few seconds to favor their own transaction.
* **Impact:** Unfair or manipulated outcomes, vulnerability to exploitation by miners.
* **Prevention:**
    1.  **Do not use `block.timestamp` for entropy or randomness.**
    2.  Only use `block.timestamp` for long-running time periods (e.g., unlocking funds after 1 month), where a few seconds of manipulation do not matter.

---

### Denial of Service (DoS)
* **Description:** An attacker prevents the contract from operating normally. Common patterns:
    * **DoS With Block Gas Limit:** A loop iterates over an array of users (e.g., to pay dividends). If the array grows too large, the gas cost of the loop exceeds the block gas limit, and the function can *never* be successfully called again.
    * **DoS with Failed Call:** A contract relies on an external call to proceed (e.g., paying a user). An attacker can become that user with a contract that always reverts, causing the original function to fail forever.
    * **Insufficient Gas Griefing:** An attacker calls a function with *just enough* gas to execute some steps but *not enough* to complete, causing the transaction to fail but still consuming the user's gas.
* **Impact:** Critical functions become stuck, locking funds or preventing operation.
* **Prevention:**
    1.  **Favor Pull-over-Push:** Don't "push" payments to an array of users. Implement a `withdraw()` function that lets users "pull" their funds individually.
    2.  Avoid iterating over arrays that can grow indefinitely.

---

### Bad Randomness
* **Description:** Using on-chain, predictable values like `block.timestamp`, `blockhash`, `block.difficulty`, or `msg.sender` as a source of "randomness" to determine winners or critical outcomes. These are all public and can be predicted or manipulated by an attacker or miner.
* **Impact:** The "random" outcome is predictable and can be exploited.
* **Prevention:**
    1.  Use a **Verifiable Random Function (VRF)** from an off-chain oracle service like Chainlink.

---

### Unexpected Ether balance
* **Description:** A contract has logic that relies on `address(this).balance == 0`. However, Ether can be *forcibly* sent to any contract (e.g., by mining to it, or via `selfdestruct`).
* **Impact:** This can break contract logic or cause `require` statements to fail, leading to a DoS.
* **Prevention:**
    1.  **Never** write logic that depends on the contract's Ether balance being an exact value.

---

## 3. Storage, Memory & EVM Vulnerabilities

---

### Uninitialized Storage Pointer
* **Description:** A critical vulnerability, especially in proxy patterns. If a developer declares a state variable *in a function* (e.g., `MyStruct storage myStruct;`) but forgets to initialize it, the pointer defaults to storage slot `0`.
* **Impact:** The function accidentally modifies critical state variables, such as the `owner` or `implementation` address (which are often stored in slot 0). This can lead to complete contract takeover.
* **Prevention:**
    1.  Always initialize local storage pointers when declaring them.
    2.  Use modern proxy patterns (like UUPS) which have built-in protections.

---

### Write to Arbitrary Storage Location
* **Description:** A high-severity vulnerability where an attacker gains the ability to write arbitrary data to any storage slot in the contract.
* **Impact:** Complete contract takeover. The attacker can overwrite the `owner` variable, change the implementation address of a proxy, modify user balances, or corrupt any other piece of data.
* **Vectors (How it happens):**
    1.  **Delegatecall to Untrusted Callee:** (This is the most common vector). If an attacker can control the address used in a `delegatecall`, they can execute their own malicious contract in the context of the victim's storage and write anything they want.
    2.  **Uninitialized Storage Pointer:** (See above). Assigning to fields of an uninitialized struct pointer (e.g., `myStruct.owner = msg.sender`) will overwrite whatever is in slot `0`.

---

### Delegatecall to Untrusted Callee
* **Description:** `delegatecall` executes code from another contract *in the context of the calling contract* (using its storage). If the address for the `delegatecall` can be influenced by a user, they can point it to a malicious contract.
* **Impact:** The attacker's contract can overwrite any storage variable, including the owner, or self-destruct the contract. This is a primary vector for "Write to Arbitrary Storage Location."
* **Prevention:**
    1.  **Never use `delegatecall` with a user-supplied address.**
    2.  Hard-code trusted implementation addresses.

---

### Shadowing State Variables
* **Description:** A contract (Child) inherits from another contract (Parent) and declares a state variable with the *exact same name* as a variable in the Parent. This does *not* override the parent's variable; it creates a new, separate storage slot.
* **Impact:** Extreme confusion. Functions in the Parent contract will read/write to the original variable, while functions in the Child contract will read/write to the new, "shadowing" variable, leading to inconsistent state.
* **Prevention:**
    1.  Use `virtual` and `override` keywords (Solidity 0.6.0+).
    2.  Maintain a clear and unique naming convention for all inherited variables.

---

### Arbitrary Jump with Function Type Variable
* **Description:** This is a low-level vulnerability (pre-Solidity 0.4.5). Internal function pointers were not properly validated. An attacker could potentially overwrite the pointer in storage to point to an arbitrary code location (e.g., the start of a `selfdestruct`) and then execute it by calling the function that uses the pointer.
* **Impact:** Complete contract takeover, bypass of access control, or execution of arbitrary logic.
* **Prevention:**
    1.  This vulnerability is fixed in all modern compilers. Do not use any compiler version before `0.4.5`.

---

## 4. Cryptography & Signature Vulnerabilities

---

### Signature Malleability
* **Description:** The `ecrecover` function in Solidity can accept multiple valid signatures (differing in their `s` and `v` values) for the same message and signer. If a contract uses a signature as a unique ID (e.g., for a one-time vote), an attacker can copy a valid signature, slightly modify it, and "replay" it as a second, different-looking-but-valid signature.
* **Impact:** Bypassing "one-time-use" checks for signatures.
* **Prevention:**
    1.  Use OpenZeppelin's `ECDSA.sol` library, which enforces a single, non-malleable signature format.
    2.  Include `msg.sender` or a dedicated nonce in the signed hash to prevent replay attacks.

---

### Lack of Proper Signature Verification
* **Description:** A contract uses `ecrecover` to verify a signature but fails to handle edge cases.
    * **Replay Attack:** The signature is valid, but there is no `nonce` or other unique data to prevent it from being submitted multiple times.
    * **`ecrecover` Returns `address(0)`:** If the signature is invalid, `ecrecover` returns `address(0)`. The contract must check for this, otherwise it might accidentally grant permissions to the "zero address."
* **Impact:** An attacker can replay old signatures or exploit invalid ones to gain unauthorized access.
* **Prevention:**
    1.  Include a `nonce` (a unique, incrementing number) in the data that is signed. The contract must track the `nonce` used by each signer.
    2.  Always check: `address signer = ecrecover(...); require(signer != address(0), "Invalid signature");`

---

### Unencrypted Private Data On-Chain
* **Description:** A developer believes that marking a state variable as `private` makes it secret and unreadable.
* **Impact:** All data on the blockchain is **public**. `private` only prevents other contracts from reading it. Anyone can read the data from off-chain by directly inspecting the contract's storage slots. Attackers can easily read "private" passwords, API keys, or puzzle answers.
* **Prevention:**
    1.  **Never store unencrypted, sensitive secrets on-chain.**
    2.  Use a **commit-reveal scheme** for data that must be hidden temporarily.

---

### Hash Collisions With Multiple Variable Length Arguments
* **Description:** Using `keccak256()` with multiple dynamic arguments (like `string` or `bytes`) packed together: `keccak256(abi.encodePacked("a", "b"))` is the same as `keccak256(abi.encodePacked("ab"))`.
* **Impact:** An attacker can craft inputs that produce an expected hash collision, potentially bypassing security checks.
* **Prevention:**
    1.  **Use `abi.encode()`:** This function pads arguments correctly and is not vulnerable. `keccak256(abi.encode("a", "b"))` is *not* the same as `keccak256(abi.encode("ab"))`.

---

## 5. Token & Standard-Specific Issues

---

### ERC20 `approve()` Race Condition
* **Description:** A front-running vulnerability. A user first approves a spender for 100 tokens. Later, they send a new `approve(spender, 50)` transaction. The spender sees this in the mempool, front-runs it, and submits `transferFrom(user, 100)`. After the user's transaction is mined, they *again* submit `transferFrom(user, 50)`, stealing a total of 150 tokens.
* **Impact:** Theft of tokens beyond the user's intended allowance.
* **Prevention:**
    1.  **Use `increaseAllowance()` and `decreaseAllowance()`:** These functions (from OpenZeppelin's `SafeERC20`) perform relative changes, making this attack impossible.
    2.  **Set to Zero:** Manually set the allowance to 0 first, wait for that transaction to be mined, and then set the new allowance.

---

### Non-Standard / Fee-on-Transfer Tokens
* **Description:** A contract (e.g., a DEX) interacts with an ERC20 token and *assumes* that `token.transferFrom(user, address, 100)` will increase its own balance by exactly 100. However, "fee-on-transfer" tokens take a percentage of the amount as a fee.
* **Impact:** The contract's internal accounting desynchronizes from its *actual* token balance. This discrepancy can be exploited to drain the contract of other assets.
* **Prevention:**
    1.  **Do not trust the input amount.**
    2.  Check the contract's token balance *before* and *after* the transfer, and use the *difference* as the actual amount received.

---

## 6. Versioning, Best Practices & Logic Errors

---

### Outdated Compiler Version
* **Description:** Using an old, unpatched version of the Solidity compiler (e.g., `0.4.x`, `0.5.x`).
* **Impact:** The contract is vulnerable to all the publicly known bugs and security flaws that have been fixed in newer versions.
* **Prevention:**
    1.  Use a recent, stable version of the compiler (e.g., `0.8.20` or higher).

---

### Floating Pragma
* **Description:** Using a "floating" pragma like `pragma solidity ^0.8.0;`. This allows the contract to be compiled with *any* version from `0.8.0` up to `0.8.255`.
* **Impact:** A new compiler version could be released with a new, unknown bug that affects the contract. It also makes auditing and verification difficult.
* **Prevention:**
    1.  **Lock the pragma:** Use a specific, non-floating version: `pragma solidity 0.8.20;`

---

### Requirement Violation
* **Description:** Refers to flaws in `require()` statements.
    1.  **Missing `require()`:** The code fails to validate inputs (e.g., a zero address, an impossible amount) that breaks contract logic.
    2.  **Incorrect `require()`:** The logic *inside* the `require` statement is wrong (e.g., `>` instead of `>=`).
* **Impact:** Varies from a simple revert (mild DoS) to allowing attackers to bypass critical security checks and corrupt state.
* **Prevention:**
    1.  **Sanitize All Inputs:** Treat all `external` and `public` function parameters as untrusted.
    2.  Thoroughly test all boundary conditions.

---

### Assert Violation
* **Description:** `require()` is for inputs, but `assert()` is for internal invariants (things that should *never* be false). An "Assert Violation" vulnerability means an `assert()` statement can be triggered by external inputs.
* **Impact:** `assert()` consumes all gas on failure. An attacker can use this to cause a DoS or exploit a critical logic error.
* **Prevention:**
    1.  Use `require()` for all input validation.
    2.  Use `assert()` only to check for internal invariants. `assert()` should never fail.

---

### Use of Deprecated Solidity Functions
* **Description:** Using old functions that have been renamed or removed for security reasons. Examples: `suicide()` (now `selfdestruct()`), `sha3()` (now `keccak256()`), `block.blockhash(uint blockNumber)` (only works for 256 most recent blocks).
* **Impact:** Confusion, or critical logic failures (especially when `blockhash` returns 0).
* **Prevention:**
    1.  Use the modern equivalents: `selfdestruct`, `keccak256`, etc.

---

### Short Address Attack
* **Description:** A (mostly historical) vulnerability. If a user sent an ERC20 transfer to an address missing its last byte, the EVM padded the call data with zeros. If the *amount* parameter came after the address, it was shifted and multiplied by 256.
* **Impact:** An attacker could send a tiny amount of tokens and have it be interpreted as a massive amount.
* **Prevention:**
    1.  Modern wallets and `SafeERC20` libraries prevent this.

---

### Typographical Error
* **Description:** A simple human error where a developer makes a typo (e.g., `tx.origin` instead of `msg.sender`, `>` instead of `<`).
* **Impact:** Varies from harmless bugs to critical access control bypasses.
* **Prevention:**
    1.  **Thorough Testing** and **Code Reviews** are the best defense.
    2.  Use Code Linters (like Solhint).

---

### Right-To-Left-Override control character (U+202E)
* **Description:** A deceptive attack. A contract name contains the U+202E character, which reverses the text that follows it. E.g., "Fee [U+202E] 01%" renders as "Fee 10%".
* **Impact:** Tricking users during audits or UI interactions into approving a malicious contract.
* **Prevention:**
    1.  Code linters and UIs should scan for and flag these characters.

---

### Presence of Unused Variables
* **Description:** The code contains variables (state, local, or parameters) that are declared but never used.
* **Impact:** Not a direct vulnerability, but it indicates "dead code," which can cause confusion, increase gas costs, or be a symptom of a *forgotten* logic check.
* **Prevention:**
    1.  Pay attention to compiler warnings.
    2.  Regularly refactor code.

---

### Code With No Effects
* **Description:** A statement or function call is made that has no effect on the state (e.g., calling an ERC20 `balanceOf()` function and not using the result).
* **Impact:** Wasted gas and indicates a logical bug or misunderstanding by the developer.
* **Prevention:**
    1.  Review code for statements that do not assign a value or change state.