// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DoubleClaimRewardBench {
    mapping(address => uint256) public stakeOf;
    mapping(address => uint256) public rewardAccrued;

    function stake() external payable {
        stakeOf[msg.sender] += msg.value;
        rewardAccrued[msg.sender] += msg.value / 10;
    }

    function claim() external returns (uint256 reward) {
        reward = rewardAccrued[msg.sender];
        require(reward > 0, "nothing");
    }
}
