// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VulnerableWallet {
    mapping(address => uint256) public balances;

    // Allow users to deposit Ether into the wallet
    function deposit() public payable {
        balances[msg.sender] += msg.value; // vulnerable to overflow in older Solidity versions
    }

    // Withdraw Ether from the wallet
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        // Vulnerable to reentrancy attack
        (bool sent, ) = msg.sender.call{value: amount}("");
        require(sent, "Failed to send Ether");
        balances[msg.sender] -= amount; // state updated after external call
    }

    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}
