// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract LoanToken {
    mapping(address => uint256) public balances;

    function mint(address user, uint256 amount) external {
        balances[user] += amount;
    }

    function transferLike(address _from, address _to, uint256 _value) external returns (bool) {
        uint256 _balancesFrom = balances[_from];
        uint256 _balancesTo = balances[_to];

        require(_to != address(0), "zero address");
        require(_balancesFrom >= _value, "insufficient");

        uint256 _balancesFromNew = _balancesFrom - _value;
        balances[_from] = _balancesFromNew;

        uint256 _balancesToNew = _balancesTo + _value;
        balances[_to] = _balancesToNew;

        return true;
    }
}
