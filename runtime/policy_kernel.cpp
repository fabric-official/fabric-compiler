#include <stdexcept> struct Policy{bool has_allow=false,has_deny=false;};
static void defaultDenyIfMissing(const Policy&p){ if(!p.has_allow && !p.has_deny) throw std::runtime_error("policy default-deny: missing allow/deny block"); }
void enforce_group_policy(const Policy&p){ defaultDenyIfMissing(p); /* TODO: enforce group rules */ }
