// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract VulnerableWallet {
    mapping(address => uint256) public balances;
    // Allow users to deposit Ether into the wallet
    function deposit() public payable {
        balances[msg.sender] += msg.value; // Vulnerable to integer overflow/underflow in older Solidity versions
    }
    // Withdraw Ether from the wallet
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        // Incorrect order of operations; vulnerable to reentrancy
        (bool sent, ) = msg.sender.call{value: amount}("");
        require(sent, "Failed to send Ether");
        balances[msg.sender] -= amount; // State update happens after external call
    }
    // Get the wallet balance of the caller
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}