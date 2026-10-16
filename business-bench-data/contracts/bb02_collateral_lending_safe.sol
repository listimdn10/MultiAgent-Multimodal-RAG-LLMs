// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract CollateralBoundBenchSafe {
    mapping(address => uint256) public collateral;
    mapping(address => uint256) public debt;
    uint256 public collateralFactorBps = 7500;

    function depositCollateral() external payable {
        collateral[msg.sender] += msg.value;
    }

    function borrow(uint256 amount) external {
        uint256 allowed = (collateral[msg.sender] * collateralFactorBps) / 10_000;
        require(debt[msg.sender] + amount <= allowed, "not enough collateral");
        debt[msg.sender] += amount;
    }
}
