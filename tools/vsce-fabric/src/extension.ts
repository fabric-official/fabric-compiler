import * as vscode from 'vscode';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

function cfg() {
  const c = vscode.workspace.getConfiguration('fabric');
  return {
    fab: c.get<string>('fabPath') || 'D:\\\\Fabric\\\\fabric_Lang\\\\bin\\\\fab.exe',
    schema: c.get<string>('policySchema') || 'D:\\\\Fabric\\\\fabric_Lang\\\\brains\\\\language\\\\compiler\\\\compiler\\\\backend\\\\schema\\\\policy.schema.json',
    brain: c.get<string>('brainPath') || 'D:\\\\Fabric\\\\fabric_Lang\\\\AgentVM.exe',
    out: c.get<string>('outputDir') || 'D:\\\\Fabric\\\\fabric_Lang\\\\out'
  };
}

function compileDoc(doc: vscode.TextDocument) {
  const { fab, schema, out } = cfg();
  if (!fs.existsSync(fab)) { vscode.window.showErrorMessage('fab.exe not found: ' + fab); return; }
  if (!fs.existsSync(out)) fs.mkdirSync(out, { recursive: true });
  const ir = path.join(out, path.basename(doc.fileName).replace(/\\.fab$/i, '.ir.json'));
  const args = ['build', '--stdin'];
  if (fs.existsSync(schema)) args.push('--schema', schema);

  const p = spawn(fab, args, { stdio: ['pipe','pipe','pipe'] });
  p.stdin.write(doc.getText()); p.stdin.end();

  let outBuf = ''; let errBuf = '';
  p.stdout.on('data', d => outBuf += d.toString());
  p.stderr.on('data', d => errBuf += d.toString());
  p.on('close', code => {
    if (code === 0 && outBuf.trim().length) {
      fs.writeFileSync(ir, outBuf);
      vscode.window.showInformationMessage('Compiled  ' + ir);
      vscode.workspace.openTextDocument(ir).then(vscode.window.showTextDocument);
    } else {
      const ch = vscode.window.createOutputChannel('Fabric');
      ch.clear(); ch.appendLine(errBuf || outBuf || ('Exited ' + code)); ch.show(true);
      vscode.window.showErrorMessage('Compile failed (see OUTPUT  Fabric).');
    }
  });
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand('fabric.compile', () => {
      const ed = vscode.window.activeTextEditor;
      if (!ed || ed.document.languageId !== 'fabric') return vscode.window.showErrorMessage('Open a .fab file.');
      compileDoc(ed.document);
    }),
    vscode.commands.registerCommand('fabric.validatePolicy', () => {
      const ed = vscode.window.activeTextEditor;
      if (!ed || ed.document.languageId !== 'fabric') return vscode.window.showErrorMessage('Open a .fab file.');
      const { fab, schema } = cfg();
      if (!fs.existsSync(fab)) return vscode.window.showErrorMessage('fab.exe not found: ' + fab);
      if (!fs.existsSync(schema)) return vscode.window.showErrorMessage('policy.schema.json not found: ' + schema);
      const p = spawn(fab, ['build','--stdin','--schema', schema], { stdio: ['pipe','ignore','pipe'] });
      p.stdin.write(ed.document.getText()); p.stdin.end();
      let err = '';
      p.stderr.on('data', d => err += d.toString());
      p.on('close', code => code === 0 ? vscode.window.showInformationMessage('Policy OK.') :
        (vscode.window.showErrorMessage('Policy failed (see OUTPUT  Fabric).'),
         (function(){ const ch=vscode.window.createOutputChannel('Fabric'); ch.clear(); ch.appendLine(err); ch.show(true); })()));
    }),
    vscode.commands.registerCommand('fabric.run', () => {
      const { brain, out } = cfg();
      if (!fs.existsSync(brain)) return vscode.window.showErrorMessage('Brain not found: ' + brain);
      const files = fs.existsSync(out) ? fs.readdirSync(out).filter(f => f.endsWith('.ir.json')) : [];
      if (!files.length) return vscode.window.showErrorMessage('No IR found. Compile first.');
      const latest = files.map(f => ({f, t: fs.statSync(path.join(out,f)).mtimeMs}))
                          .sort((a,b)=>b.t-a.t)[0].f;
      const p = spawn(brain, ['--ir', path.join(out, latest)], { stdio: 'inherit' });
      vscode.window.showInformationMessage('Brain running with ' + latest + ' (see terminal).');
    })
  );
}

export function deactivate() {}