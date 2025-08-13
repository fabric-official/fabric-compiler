pragma solidity ^0.8.20; contract XpChain{ bytes32 public head; function append(bytes32 eventHash) external { head=keccak256(abi.encodePacked(head,eventHash)); } }
