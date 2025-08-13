"use strict";
// frontend/errors.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.SemanticError = exports.CompilerError = void 0;
class CompilerError extends Error {
    constructor(message, loc) {
        super(message);
        this.name = "CompilerError";
        this.loc = loc;
        // Preserve stack trace (V8 only)
        if (Error.captureStackTrace) {
            Error.captureStackTrace(this, CompilerError);
        }
    }
    toString() {
        const locationStr = this.loc
            ? ` at line ${this.loc.start.line}, column ${this.loc.start.column}`
            : "";
        return `[CompilerError] ${this.message}${locationStr}`;
    }
}
exports.CompilerError = CompilerError;
exports.SemanticError = CompilerError;
//# sourceMappingURL=errors.js.map