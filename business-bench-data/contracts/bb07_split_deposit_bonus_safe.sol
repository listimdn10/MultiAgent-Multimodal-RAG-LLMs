// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract SplitDepositBonusBenchSafe {
    mapping(address => uint256) public shares;

    function deposit(uint256 assets) external {
        uint256 bonus = assets >= 100 ether ? 5 ether : 0;
        shares[msg.sender] += assets + bonus;
    }

    function withdraw(uint256 shareAmount) external returns (uint256 assetsOut) {
        require(shares[msg.sender] >= shareAmount, "insufficient shares");
        shares[msg.sender] -= shareAmount;
        assetsOut = shareAmount;
    }
}
