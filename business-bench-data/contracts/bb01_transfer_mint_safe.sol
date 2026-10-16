// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract TransferMintBenchSafe {
    mapping(address => uint256) public balances;

    function mint(address user, uint256 amount) external {
        balances[user] += amount;
    }

    function transferLike(address from, address to, uint256 amount) external returns (bool) {
        require(to != address(0), "zero");
        if (from == to) {
            require(balances[from] >= amount, "insufficient");
            return true;
        }

        uint256 fromBalance = balances[from];
        require(fromBalance >= amount, "insufficient");
        balances[from] = fromBalance - amount;
        balances[to] += amount;
        return true;
    }
}
