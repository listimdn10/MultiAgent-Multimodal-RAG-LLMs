// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract HealthyLiquidationBench {
    mapping(address => uint256) public collateral;
    mapping(address => uint256) public debt;
    uint256 public liquidationFactorBps = 8000;

    function openPosition(uint256 debtAmount) external payable {
        collateral[msg.sender] += msg.value;
        debt[msg.sender] += debtAmount;
    }

    function liquidate(address user) external {
        require((collateral[user] * liquidationFactorBps) / 10_000 >= debt[user], "not liquidatable");
        collateral[user] = 0;
        debt[user] = 0;
    }
}
