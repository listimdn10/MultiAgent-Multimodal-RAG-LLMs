// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract StaleOracleBorrowBench {
    uint256 public oraclePrice = 2_000 ether;
    uint256 public updatedAt;
    mapping(address => uint256) public collateral;
    mapping(address => uint256) public debt;

    function setOracle(uint256 newPrice, uint256 timestamp) external {
        oraclePrice = newPrice;
        updatedAt = timestamp;
    }

    function depositCollateral() external payable {
        collateral[msg.sender] += msg.value;
    }

    function borrow(uint256 amount) external {
        uint256 collateralValue = (collateral[msg.sender] * oraclePrice) / 1 ether;
        require(collateralValue >= amount * 2, "insufficient collateral");
        debt[msg.sender] += amount;
    }
}
