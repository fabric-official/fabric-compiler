export function checkedAdd(a:number,b:number){const x=a+b;if(!Number.isSafeInteger(x))throw new Error("overflow");return x;}
export function checkedLen(buf:Uint8Array,start:number,len:number){if(start<0||len<0)throw new Error("negative");if(start>buf.length)throw new Error("oob");if(start+len>buf.length)throw new Error("oob");return [start,len] as [number,number];}
