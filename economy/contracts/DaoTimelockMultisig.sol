pragma solidity ^0.8.20; contract DaoTimelockMultisig{ uint256 public delay; mapping(address=>bool) public owners; uint256 public threshold;
 struct Tx{address to; uint256 value; bytes data; uint256 eta; uint256 sigs; bool executed;} mapping(bytes32=>Tx) public queue; /* TODO full impl */ }
