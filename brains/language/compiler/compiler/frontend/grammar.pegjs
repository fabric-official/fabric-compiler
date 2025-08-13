{
  // Top-level initializer block (used by generated grammar.ts)
  const makeNode = (type, props) => Object.assign({ type }, props);
}

Program
  = _ statements:StatementList _ { return statements; }

StatementList
  = head:Statement tail:(_ Statement)* {
      return [head, ...tail.map(e => e[1])];
    }

Statement
  = ModuleDecl
  / AgentDecl
  / PolicyDecl
  / CoordinationDecl         // ✅ added

ModuleDecl
  = "module" __ name:Identifier __ "{" __ body:StatementList __ "}" {
      return makeNode("ModuleDecl", { name, body });
    }

AgentDecl
  = "agent" __ name:Identifier __ "{" __ body:StatementList __ "}" {
      return makeNode("AgentDecl", { name, body });
    }

PolicyDecl
  = "policy" __ name:Identifier __ "{" __ body:StatementList __ "}" {
      return makeNode("PolicyDecl", { name, body });
    }

CoordinationDecl
  = "coordination" __ "{" __ entries:CoordinationEntryList __ "}" {
      return makeNode("CoordinationBlock", { entries });
    }

CoordinationEntryList
  = head:CoordinationEntry tail:(_ CoordinationEntry)* {
      return [head, ...tail.map(e => e[1])];
    }

CoordinationEntry
  = EntangleEntry
  / ChannelEntry
  / CollapseEntry

EntangleEntry
  = "entangle" __ ":" __ "[" _ atoms:IdentifierList _ "]" {
      return makeNode("EntangleEntry", { atoms });
    }

ChannelEntry
  = "channel" __ ":" __ name:Identifier {
      return makeNode("ChannelEntry", { name });
    }

CollapseEntry
  = "collapse" __ ":" __ "on" __ trigger:Identifier {
      return makeNode("CollapseEntry", { trigger });
    }

IdentifierList
  = first:Identifier rest:(_ "," _ Identifier)* {
      return [first, ...rest.map(r => r[3])];
    }

Identifier
  = $([a-zA-Z_][a-zA-Z0-9_]*)

__ "whitespace"
  = [ \t\n\r]+

_ "optional whitespace"
  = [ \t\n\r]*


