declare function peg$subclass(child: any, parent: any): void;
declare function peg$SyntaxError(message: any, expected: any, found: any, location: any): Error;
declare namespace peg$SyntaxError {
    var buildMessage: (expected: any, found: any) => string;
}
declare function peg$padEnd(str: any, targetLength: any, padString: any): any;
declare function peg$parse(input: any, options: any): any;
