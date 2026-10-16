// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract TotalCapBypassBench {
    uint256 public totalDeposits;
    uint256 public cap = 1_000 ether;
    mapping(address => uint256) public balanceOf;

    function deposit() external payable {
        require(msg.value <= cap, "deposit too large");
        totalDeposits += msg.value;
        balanceOf[msg.sender] += msg.value;
    }
}
