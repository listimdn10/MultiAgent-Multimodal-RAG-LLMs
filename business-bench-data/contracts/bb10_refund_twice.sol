// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract RefundTwiceBench {
    mapping(address => uint256) public deposits;
    bool public saleCancelled;

    function deposit() external payable {
        deposits[msg.sender] += msg.value;
    }

    function cancelSale() external {
        saleCancelled = true;
    }

    function claimRefund() external returns (uint256 amount) {
        require(saleCancelled, "active");
        amount = deposits[msg.sender];
        require(amount > 0, "none");
    }
}
