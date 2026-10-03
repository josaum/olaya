#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo"
mkdir -p dist
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp extensions/claude-desktop/manifest.json "$stage/"
cp extensions/claude-desktop/README.md "$stage/"
cp -R extensions/claude-desktop/skills "$stage/"
mkdir -p "$stage/server"
bun build extensions/claude-desktop/server/index.ts --target node --format esm --outfile "$stage/server/index.mjs"
npx --yes @anthropic-ai/mcpb@2.1.2 validate "$stage/manifest.json"
npx --yes @anthropic-ai/mcpb@2.1.2 pack "$stage" "$repo/dist/olaya-desktop-0.1.0.mcpb"
python3 - "$stage" "$repo/dist" <<'PY'
import pathlib,sys,zipfile,hashlib,json
stage,out=map(pathlib.Path,sys.argv[1:])
for skill in sorted((stage/'skills').iterdir()):
    with zipfile.ZipFile(out/(skill.name+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(skill.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(stage/'skills'))
plugin=out/'olaya-claude-code-plugin.zip'
with zipfile.ZipFile(plugin,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('.claude-plugin/plugin.json',json.dumps({'name':'olaya-ontology','version':'0.1.0','description':'TBox workflow authoring and evidence review skills','author':{'name':'O-Laya contributors'}}))
    z.writestr('.mcp.json',json.dumps({'olaya':{'command':'node','args':['${CLAUDE_PLUGIN_ROOT}/server/index.mjs']}}))
    for f in sorted(stage.rglob('*')):
        if f.is_file() and f.name!='manifest.json':z.write(f,f.relative_to(stage))
artifacts=[out/'olaya-desktop-0.1.0.mcpb',plugin,*[out/(s.name+'.zip') for s in sorted((stage/'skills').iterdir())]]
(out/'claude-desktop-sha256.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in artifacts))
PY
