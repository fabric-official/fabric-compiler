// Fabric DSL core grammar in PEG syntax

// -- Whitespace & Comments --------------------------------------------------

WS              = [ \t\r\n]+
COMMENT         = "//" (![\\r\\n] .)*      // single-line
BLOCK_COMMENT   = "/*" (!"*/" .)* "*/"     // multi-line
_               = (WS / COMMENT / BLOCK_COMMENT)*

// -- Basic Tokens ------------------------------------------------------------

Identifier      = !Reserved [A-Za-z_][A-Za-z0-9_]*
StringLiteral   = "\"" (!"\"" .)* "\"" / "\"\"\"" (!"\"\"\"" .)* "\"\"\""
NumberLiteral   = [0-9]+ ("." [0-9]+)?
UUID            = hex8 "-" hex4 "-" hex4 "-" hex4 "-" hex12
hex             = [0-9a-fA-F]
hex4            = hex hex hex hex
hex8            = hex4 hex4
hex12           = hex4 hex4 hex4
Timestamp       = digit digit digit digit "-" digit digit "-" digit digit "T" digit digit ":" digit digit ":" digit digit ( "Z" / ("+" digit digit ":" digit digit) )
digit           = [0-9]

Reserved        = "module" ![A-Za-z0-9_] 
                / "import" ![A-Za-z0-9_] 
                / "device" ![A-Za-z0-9_] 
                / "policy" ![A-Za-z0-9_]
                / "goal" ![A-Za-z0-9_] 
                / "type" ![A-Za-z0-9_] 
                / "agent" ![A-Za-z0-9_]
                / "executionBlock" ![A-Za-z0-9_] 
                / "polyBlock" ![A-Za-z0-9_]
                / "workflow" ![A-Za-z0-9_] 
                / "coordinationBlock" ![A-Za-z0-9_]
                / "auditTrail" ![A-Za-z0-9_]

EOF             = !.
document        = _ Module _ EOF

// -- Top-level ---------------------------------------------------------------

Module          = "module" _ Identifier _ "{" _ TopLevel* _ "}" _
TopLevel        = Import / Device / Policy / Goal / TypeDef / Agent / ExecutionBlock / PolyBlock / Workflow / CoordinationBlock / AuditTrail

Import          = "import" _ IdentList _ ";" _
IdentList       = Identifier (_ "," _ Identifier)*

Device          = "device" _ Identifier _ "{" _ DeviceBody _ "}" _
DeviceBody      = "caps" _ ":" _ "[" _ CapList _ "]" _ "," _ "policy" _ ":" _ PolicyBlock
CapList         = Cap (_ "," _ Cap)*
Cap             = Identifier (_ "@" _ Identifier)?

// -- POLICY: now completely generic KV pairs -------------------------------

Policy          = "policy" _ PolicyBlock _
PolicyBlock     = "{" _ KVPairList? _ "}" _
KVPairList      = KVPair (_ "," _ KVPair)*
KVPair          = Identifier _ ":" _ Value

// A policy value may be a string, number, boolean, or a list thereof.
Value           = StringLiteral
                / NumberLiteral
                / "true" / "false"
                / ListLiteral

ListLiteral     = "[" _ (Value (_ "," _ Value)*)? _ "]"

// -- Goal --------------------------------------------------------------------

Goal            = "goal" _ StringLiteral _ GoalBlock _
GoalBlock       = "{" _ GoalEntries _ "}" _
GoalEntries     = ConstraintsOpt OptimizeOpt
ConstraintsOpt  = "constraints" _ "{" _ ConstraintList _ "}" / ""
ConstraintList  = Constraint (_ ";" _ Constraint)* 
Constraint      = Identifier _ Comparator _ NumberLiteral _ Identifier
Comparator      = "<=" / ">=" / "<" / ">" / "==" 
OptimizeOpt     = "optimize" _ "for" _ "{" _ IdentifierList _ "}" / ""
IdentifierList  = Identifier (_ "," _ Identifier)*

// -- Types -------------------------------------------------------------------

TypeDef         = "type" _ Identifier _ "=" _ TypeExpr _ ";" _
TypeExpr        = Identifier _ "[" _ NumberLiteral _ "]"
                 / Identifier
                 / "union" _ "(" _ TypeExprList _ ")"
TypeExprList    = TypeExpr (_ "," _ TypeExpr)*

// -- Agent ------------------------------------------------------------------

Agent           = "agent" _ Identifier _ AgentAttrs _ "{" _ AgentBody _ "}" _
AgentAttrs      = (Attr)*
Attr            = ("id" / "model_id" / "created_by" / "timestamp") _ ":" _ (UUID / StringLiteral) _
AgentBody       = IOBlock (_ ";" _ Assignment)* _
IOBlock         = "inputs" _ ":" _ "[" _ IOList _ "]" _ "," _ "outputs" _ ":" _ "[" _ IOList _ "]"
IOList          = IODecl (_ "," _ IODecl)* 
IODecl          = Identifier (_ "@" _ Identifier)?
Assignment      = "learns" _ ":" _ (Identifier / "continuous")
                 / "explain" _ ":" _ "[" _ StringLiteralList _ "]"
                 / "device" _ ":" _ (Identifier / "any")
                 / "fallback" _ ":" _ Identifier _ "{" _ "state" _ ":" _ Identifier _ "}"
                 / "policy" _ ":" _ PolicyBlock
StringLiteralList = StringLiteral (_ "," _ StringLiteral)*

// -- ExecutionBlock ----------------------------------------------------------

ExecutionBlock  = "executionBlock" _ Identifier _ BlockAttrs _ "{" _ ExecBlockBody _ "}" _
BlockAttrs      = (Attr)*
ExecBlockBody   = "block" _ ":" _ "{" _ ExecEntries _ "}" _
ExecEntries     = ExecEntry (_ "," _ ExecEntry)* 
ExecEntry       = "type" _ ":" _ Identifier
                 / "agent" _ ":" _ Identifier
                 / "entry" _ ":" _ StringLiteral
                 / "inputs" _ ":" _ "[" _ IOList _ "]"
                 / "outputs" _ ":" _ "[" _ IOList _ "]"

// -- PolyBlock ---------------------------------------------------------------

PolyBlock       = "polyBlock" _ Identifier _ BlockAttrs _ "{" _ PolyEntries _ "}" _
PolyEntries     = PolyEntry (_ "," _ PolyEntry)* 
PolyEntry       = "lang" _ ":" _ StringLiteral
                 / "code" _ ":" _ "\"\"\"" (!"\"\"\"" .)* "\"\"\""
                 / "entry" _ ":" _ StringLiteral
                 / "inputs" _ ":" _ "[" _ IOList _ "]"
                 / "outputs" _ ":" _ "[" _ IOList _ "]"
                 / "container" _ ":" _ "{" _ ContainerList _ "}"
                 / "policy" _ ":" _ PolicyBlock
ContainerList   = Identifier (_ "," _ Identifier)*

// -- Workflow ---------------------------------------------------------------

Workflow        = "workflow" _ Identifier _ "{" _ WFBody _ "}" _
WFBody          = "plan" _ ":" _ Plan _ "," _
                   "coordination" _ ":" _ CoordBlock _ "," _
                   "feedback" _ ":" _ Feedback (_ "," _ WorkflowPrims)? _
Plan            = Identifier (_ "->" _ Identifier)+
CoordBlock      = "{" _ "consensus" _ ":" _ ("true" / "false") (_ "," _ "conflict" _ ":" _ Identifier _ "(" _ StringLiteralList _ ")")? _ "}"
Feedback        = "{" _ "metrics" _ ":" _ "[" _ IdentifierList _ "]" _ "," _ "interval" _ ":" _ StringLiteral _ "}" _

WorkflowPrims   = OnError / Schedule / Alert
OnError         = "onError" _ "retry" _ NumberLiteral _ "times" (_ "with" _ "backoff" _ "(" _ StringLiteral _ ")")? _ ";"
Schedule        = "schedule" _ "cron" _ "(" _ StringLiteral _ ")" (_ "backfill" _ "(" _ "window" _ ":" _ StringLiteral _ ")")? _ ";"
Alert           = "alert" _ "if" _ Identifier _ Comparator _ NumberLiteral _ (Identifier)? _ "then" _ "notify" _ "(" _ StringLiteral _ ")" _ ";"

// -- CoordinationBlock -------------------------------------------------------

CoordinationBlock = "coordinationBlock" _ Identifier _ "{" _ CBBody _ "}" _
CBBody           = "agents" _ ":" _ "[" _ IdentifierList _ "]" _ "," _ "protocol" _ ":" _ Identifier _ "," _ "on_commit" _ ":" _ "{" _ (!"}" .)* _ "}"

// -- AuditTrail --------------------------------------------------------------

AuditTrail      = "auditTrail" _ Identifier _ "{" _ ATBody _ "}" _
ATBody          = "snapshotOn" _ ":" _ "[" _ StringLiteralList _ "]" _ "," _ "store" _ ":" _ Identifier

