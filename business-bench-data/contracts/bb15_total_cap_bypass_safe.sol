// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract TotalCapBypassBenchSafe {
    uint256 public totalDeposits;
    uint256 public cap = 1_000 ether;
    mapping(address => uint256) public balanceOf;

    function deposit() external payable {
        require(totalDeposits + msg.value <= cap, "cap exceeded");
        totalDeposits += msg.value;
        balanceOf[msg.sender] += msg.value;
    }
}
