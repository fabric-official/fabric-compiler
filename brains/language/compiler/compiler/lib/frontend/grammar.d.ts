declare function peg$SyntaxError(message: any, expected: any, found: any, location: any): Error;
declare namespace peg$SyntaxError {
    var buildMessage: (expected: any, found: any) => string;
}
declare function peg$parse(input: any, options: any): void;
export { peg$SyntaxError as SyntaxError, peg$parse as parse };
