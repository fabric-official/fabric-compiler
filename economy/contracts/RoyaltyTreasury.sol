pragma solidity ^0.8.20; library MerkleProof{function verify(bytes32[] memory proof,bytes32 root,bytes32 leaf) internal pure returns(bool){bytes32 c=leaf; for(uint i=0;i<proof.length;i++){bytes32 p=proof[i]; c=(c<p)?keccak256(abi.encodePacked(c,p)):keccak256(abi.encodePacked(p,c));} return c==root;}}
contract RoyaltyTreasury{ bytes32 public claimsRoot; mapping(bytes32=>bool) public claimed;
 function claim(bytes32 leaf,bytes32[] calldata proof,address payable to,uint256 amount) external{
  require(MerkleProof.verify(proof,claimsRoot,leaf),"bad-proof"); require(!claimed[leaf],"dup"); claimed[leaf]=true;
  (bool ok,)=to.call{value:amount}(""); require(ok,"xfer-fail"); } }
