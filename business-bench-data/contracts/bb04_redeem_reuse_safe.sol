// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract RedeemReuseBenchSafe {
    mapping(address => uint256) public credits;

    function grant(address user, uint256 amount) external {
        credits[user] += amount;
    }

    function redeem(uint256 amount) external returns (bool) {
        require(credits[msg.sender] >= amount, "insufficient credits");
        credits[msg.sender] -= amount;
        return true;
    }
}
