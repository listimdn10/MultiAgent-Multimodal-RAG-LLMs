// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract EarlyVestingReleaseBench {
    struct Grant {
        uint256 amount;
        uint256 releaseTime;
        bool released;
    }

    mapping(address => Grant) public grants;

    function grant(address user, uint256 amount, uint256 releaseTime) external {
        grants[user] = Grant(amount, releaseTime, false);
    }

    function release() external returns (uint256 amount) {
        Grant storage grantInfo = grants[msg.sender];
        require(!grantInfo.released, "released");
        amount = grantInfo.amount;
        grantInfo.released = true;
    }
}
