// frontend/errors.ts

export interface SourceLocation {
    start: { line: number; column: number };
    end?: { line: number; column: number };
}

export class CompilerError extends Error {
    public loc?: SourceLocation;

    constructor(message: string, loc?: SourceLocation) {
        super(message);
        this.name = "CompilerError";
        this.loc = loc;

        // Preserve stack trace (V8 only)
        if (Error.captureStackTrace) {
            Error.captureStackTrace(this, CompilerError);
        }
    }

    toString(): string {
        const locationStr = this.loc
            ? ` at line ${this.loc.start.line}, column ${this.loc.start.column}`
            : "";
        return `[CompilerError] ${this.message}${locationStr}`;
    }
}

// Optional: You can still alias for older references
export { CompilerError as SemanticError };

