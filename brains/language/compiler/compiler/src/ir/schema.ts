export interface AgentIR {
    name: string;
    model_id: string;
    inputs: string[];
    outputs: string[];
    policy?: Record<string, any>;
    auditTrail?: boolean;
}

export interface IRFunction {
    name: string;
    instructions: string[];
}

export interface IRModule {
    agents: AgentIR[];
    functions: IRFunction[];
}

