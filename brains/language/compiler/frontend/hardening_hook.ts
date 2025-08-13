import { sanitizePolicy, inferTypesAndGuard } from "./ast_sanitize";
export function hardenFabAST(ast:any){ if(ast?.policy) sanitizePolicy(ast.policy); inferTypesAndGuard(ast); return ast; }
