export interface SourceLocation {
    start: {
        line: number;
        column: number;
    };
    end?: {
        line: number;
        column: number;
    };
}
export declare class CompilerError extends Error {
    loc?: SourceLocation;
    constructor(message: string, loc?: SourceLocation);
    toString(): string;
}
export { CompilerError as SemanticError };
