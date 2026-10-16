// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FinalizeWithoutCommitBench {
    address public leader;
    uint256 public highestBid;
    bool public finalized;

    function placeBid() external payable {
        require(!finalized, "finalized");
        if (msg.value > highestBid) {
            highestBid = msg.value;
            leader = msg.sender;
        }
    }

    function finalize() external {
        require(!finalized, "already");
        finalized = true;
    }
}
