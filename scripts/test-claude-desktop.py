"""Exercise the extracted release over stdio, independent of repository/runtime packages."""
import json
import pathlib
import shutil
import subprocess
import tempfile
import zipfile

repo = pathlib.Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='olaya offline bundle ') as folder:
    root = pathlib.Path(folder)
    with zipfile.ZipFile(repo / 'dist/olaya-desktop-0.1.0.mcpb') as archive:
        assert all(not p.startswith('/') and '..' not in pathlib.PurePosixPath(p).parts for p in archive.namelist())
        archive.extractall(root)
    node = shutil.which('node')
    assert node, 'Node required on development host for artifact verification'
    proc = subprocess.Popen([node, str(root / 'server/index.mjs')], cwd=root, env={'PATH': '/usr/bin:/bin'}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    counter = 0
    def rpc(method, params=None):
        global counter
        counter += 1
        proc.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': counter, 'method': method, 'params': params or {}}) + '\n')
        proc.stdin.flush()
        value = json.loads(proc.stdout.readline())
        assert value['id'] == counter, value
        return value
    init = rpc('initialize', {'protocolVersion': '2025-11-25', 'capabilities': {}, 'clientInfo': {'name': 'artifact-test', 'version': '1'}})
    assert set(init['result']['capabilities']) == {'tools', 'resources', 'prompts'}
    assert len(rpc('tools/list')['result']['tools']) == 3
    assert len(rpc('prompts/list')['result']['prompts']) == 2
    for prompt in ['olaya-tbox-workflows', 'olaya-evidence-review']:
        assert rpc('prompts/get', {'name': prompt})['result']['messages'][0]['content']['text'].startswith('---')
    resources = rpc('resources/list')['result']['resources']
    assert len(resources) >= 14
    for item in resources:
        assert rpc('resources/read', {'uri': item['uri']})['result']['contents'][0]['text']
    def tool(name, args):
        return rpc('tools/call', {'name': name, 'arguments': args})['result']
    assert tool('olaya_capabilities', {})['structuredContent']['trained'] is False
    concepts = [dict(id=0, curie='urn:test:Storage', label='Storage', parents=[], disjointWith=[], synonyms=[]), dict(id=1, curie='urn:test:Cold', label='Cold refrigerated storage', parents=[0], disjointWith=[], synonyms=['frozen'])]
    data = tool('olaya_ground', {'state': 'frozen refrigerated', 'concepts': concepts})['structuredContent']
    assert data['decision']['selectedConcept']['curie'] == 'urn:test:Cold'
    assert data['jsonld']['predictedClass'] == 'urn:test:Cold'
    assert 'frozen refrigerated' not in json.dumps(data['jsonld'])
    assert tool('olaya_read_reference', {'path': '../../etc/passwd'})['isError']
    assert tool('olaya_ground', {'state': 'x', 'concepts': concepts + [concepts[0]]})['isError']
    concepts[0]['parents'] = [1]
    assert tool('olaya_ground', {'state': 'x', 'concepts': concepts})['isError']
    assert rpc('resources/read', {'uri': 'file:///etc/passwd'})['error']['code'] == -32602
    assert rpc('unknown')['error']['code'] == -32601
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0
    assert not proc.stderr.read()
    print(f'PASS: extracted bundle, stdio tools/prompts, {len(resources)} resources, JSON-LD, malformed DAG and path traversal rejection')
