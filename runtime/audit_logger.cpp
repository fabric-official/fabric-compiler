#include <vector> #include <string> #include "sha256.h"
struct Entry{std::string data,prev_hash,hash;};
class AuditLog{ std::vector<Entry> log;
public: void append(const std::string&data){std::string prev=log.empty()?std::string(64,'0'):log.back().hash; std::string h=sha256(prev+data); log.push_back({data,prev,h});}
 bool verify()const{std::string prev(64,'0'); for(auto&e:log){ if(sha256(prev+e.data)!=e.hash) return false; prev=e.hash;} return true; } };
