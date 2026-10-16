// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract TransferMintBench {
    mapping(address => uint256) public balances;

    function mint(address user, uint256 amount) external {
        balances[user] += amount;
    }

    function transferLike(address from, address to, uint256 amount) external returns (bool) {
        uint256 fromBalance = balances[from];
        uint256 toBalance = balances[to];

        require(to != address(0), "zero");
        require(fromBalance >= amount, "insufficient");

        uint256 fromNew = fromBalance - amount;
        balances[from] = fromNew;

        uint256 toNew = toBalance + amount;
        balances[to] = toNew;
        return true;
    }
}
