// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DuplicateVoteBench {
    struct Proposal {
        uint256 forVotes;
        uint256 againstVotes;
    }

    mapping(uint256 => Proposal) public proposals;
    mapping(address => uint256) public votingPower;

    constructor() {
        votingPower[msg.sender] = 100;
    }

    function vote(uint256 proposalId, bool support) external {
        uint256 weight = votingPower[msg.sender];
        require(weight > 0, "no voting power");
        if (support) {
            proposals[proposalId].forVotes += weight;
        } else {
            proposals[proposalId].againstVotes += weight;
        }
    }
}
