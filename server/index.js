const fs = require("fs");
const path = require("path");
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const { hideBin } = require("yargs/helpers");
const yargs = require("yargs/yargs");

const argv = yargs(hideBin(process.argv))
  .option("bind",       { type:"string", default: process.env.BIND || "0.0.0.0:8891" })
  .option("tls_cert",   { type:"string", default: process.env.TLS_CERT })
  .option("tls_key",    { type:"string", default: process.env.TLS_KEY })
  .option("mtls_ca",    { type:"string", default: process.env.MTLS_CA })
  .option("auth_token", { type:"string", default: process.env.AUTH_TOKEN })
  .strict(false).argv;

const PROTO = path.join(__dirname, "..", "proto", "fabric", "core", "language", "v1", "language.proto");
const pkgDef = protoLoader.loadSync(PROTO, { keepCase:true, longs:String, enums:String, defaults:true, oneofs:true });
const proto  = grpc.loadPackageDefinition(pkgDef).fabric.core.language.v1;

const runner = require("./runner");
const solidity = require("./solidityGen");

// --- auth interceptor (optional bearer) ---
function authInterceptor(token) {
  return (options, nextCall) => {
    return new grpc.InterceptingCall(nextCall(options), {
      start: (metadata, listener, next) => {
        if (token) {
          const got = (metadata.get("authorization")[0] || "").toString();
          const ok  = got.toLowerCase().startsWith("bearer ") && got.slice(7).trim() === token;
          if (!ok) {
            listener.onReceiveStatus({ code: grpc.status.UNAUTHENTICATED, details: "Missing/invalid bearer token" });
            return;
          }
        }
        next(metadata, listener);
      }
    });
  };
}

// --- TLS / mTLS ---
let creds;
if (argv.tls_cert && argv.tls_key) {
  const key = fs.readFileSync(argv.tls_key);
  const crt = fs.readFileSync(argv.tls_cert);
  const ca  = argv.mtls_ca ? fs.readFileSync(argv.mtls_ca) : null;
  creds = grpc.ServerCredentials.createSsl(ca, [{ private_key: key, cert_chain: crt }], !!argv.mtls_ca);
  console.log("[brain] TLS enabled" + (argv.mtls_ca ? " (mTLS)" : ""));
} else {
  creds = grpc.ServerCredentials.createInsecure();
  console.log("[brain] INSECURE (dev) listener; provide --tls_cert/--tls_key for TLS");
}

// --- service impl ---
const svc = {
  Compile: async (call, cb) => {
    try {
      const { src = [], flags = {} } = call.request || {};
      const targets = Array.isArray(flags.targets) ? flags.targets : [];
      const res = await runner.compile({ src, flags });
      if (targets.includes("sol")) {
        await solidity.verifySolidityOutputs(res); // throws if missing .sol
      }
      cb(null, res);
    } catch (e) { cb(e); }
  },
  Atomize: async (call, cb) => {
    try { cb(null, await runner.atomize(call.request||{})); }
    catch(e){ cb(e); }
  },
  PolicyLint: async (call, cb) => {
    try { cb(null, await runner.policyLint(call.request||{})); }
    catch(e){ cb(e); }
  }
};

const server = new grpc.Server({ "grpc.max_receive_message_length": 64*1024*1024, interceptors: [authInterceptor(argv.auth_token)] });
server.addService(proto.LanguageBrain.service, svc);
server.bindAsync(argv.bind, creds, (err, port) => {
  if (err) { console.error(err); process.exit(1); }
  console.log(`[brain] listening on ${argv.bind}`);
  server.start();
});