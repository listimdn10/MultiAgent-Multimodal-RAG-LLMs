// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract BridgeNonceReuseBenchSafe {
    mapping(bytes32 => bool) public approvedMessage;
    mapping(bytes32 => bool) public consumedMessage;
    mapping(address => uint256) public minted;

    function approveMessage(bytes32 messageId) external {
        approvedMessage[messageId] = true;
    }

    function finalizeWithdraw(bytes32 messageId, address to, uint256 amount) external {
        require(approvedMessage[messageId], "not approved");
        require(!consumedMessage[messageId], "already consumed");
        consumedMessage[messageId] = true;
        minted[to] += amount;
    }
}
