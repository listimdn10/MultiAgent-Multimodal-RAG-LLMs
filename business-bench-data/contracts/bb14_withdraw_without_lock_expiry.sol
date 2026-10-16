// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract WithdrawWithoutLockExpiryBench {
    mapping(address => uint256) public lockedBalance;
    mapping(address => uint256) public unlockTime;

    function depositLocked(uint256 lockSeconds) external payable {
        lockedBalance[msg.sender] += msg.value;
        unlockTime[msg.sender] = block.timestamp + lockSeconds;
    }

    function withdraw() external returns (uint256 amount) {
        amount = lockedBalance[msg.sender];
        require(amount > 0, "none");
        lockedBalance[msg.sender] = 0;
    }
}
