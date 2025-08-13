pragma solidity ^0.8.20; contract PaymasterHardened{ mapping(bytes32=>bool) public used; function _domainSep() internal view returns(bytes32){return keccak256("FABRIC");}
 function validate(bytes calldata meta,bytes calldata sig) external{ bytes32 h=keccak256(abi.encode(_domainSep(),keccak256(meta))); require(!used[h],"replay"); /* TODO ecrecover signer check */ used[h]=true; } }
