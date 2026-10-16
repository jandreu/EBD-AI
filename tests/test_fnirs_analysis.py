"""Tests for the fNIRS analysis demo (Demos/fnirs_analysis).

The demo's own offline suite (regression_test.py) drives the FastAPI backend
with a fake language model; it needs fastapi and httpx installed. The Claude
CLI client is tested against a stand-in script, so no login is needed.
"""
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pandas as pd
import pytest

DEMO = pathlib.Path(__file__).resolve().parents[1] / 'Demos' / 'fnirs_analysis'


@pytest.fixture(scope='module')
def claude_cli():
    spec = importlib.util.spec_from_file_location('fnirs_claude_cli', DEMO / 'claude_cli.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_CLAUDE = """#!{python}
import json, os, sys
with open(os.environ['FAKE_CLAUDE_LOG'], 'w') as log:
    json.dump({{'argv': sys.argv[1:], 'stdin': sys.stdin.read(), 'cwd': os.getcwd()}}, log)
print(json.dumps({{'type': 'result', 'is_error': False, 'result': os.environ['FAKE_CLAUDE_REPLY']}}))
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    binary = tmp_path / 'claude'
    binary.write_text(FAKE_CLAUDE.format(python=sys.executable))
    binary.chmod(0o755)
    log = tmp_path / 'call.json'
    monkeypatch.setenv('FAKE_CLAUDE_LOG', str(log))
    return binary, log


def test_shipped_rules_use_the_dataset_features():
    results = json.loads((DEMO / 'data' / 'extracted_rules_final_raw.json').read_text())
    table = pd.read_csv(DEMO / 'data' / 'block_avg_tmb.csv', nrows=1)
    assert results['feature_names'] == list(table.columns[3:75])
    atlas = json.loads((DEMO / 'data' / 'channel_atlas.json').read_text())
    for rule in results['rules']:
        assert rule['consequent'] in ('0 back', '2/3 back')
        for feature, label in re.findall(r'(\S+) IS (\w+)', rule['antecedent']):
            assert feature in results['feature_names']
            assert label in ('Low', 'Medium', 'High')
            assert feature.split('_ch', 1)[1] in atlas['channels']


def test_regression_suite_passes(tmp_path):
    for module in ('fastapi', 'httpx'):
        pytest.importorskip(module)
    env = dict(os.environ, INTERACTIONS_DB_PATH=str(tmp_path / 'interactions.db'),
               LIVE_LITERATURE='', VENICE_API_KEY='')
    done = subprocess.run([sys.executable, str(DEMO / 'regression_test.py')], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert 'All checks passed.' in done.stdout


TOOLS = [{'type': 'function', 'function': {'name': 'show_topomap', 'description': 'Draw a scalp map.',
                                           'parameters': {'type': 'object', 'properties': {'channels': {'type': 'array'}}}}}]


def test_claude_calls_are_locked_down(claude_cli, fake_claude, monkeypatch):
    binary, log = fake_claude
    monkeypatch.setenv('FAKE_CLAUDE_REPLY', 'Rule 1 says AF7 is High.')
    client = claude_cli.ClaudeCliClient(str(binary), model='haiku')
    reply = client.chat(model='haiku', messages=[{'role': 'system', 'content': 'Be grounded.'},
                                                  {'role': 'user', 'content': 'What does AF7 tell us?'}])
    assert reply == {'message': {'content': 'Rule 1 says AF7 is High.', 'tool_calls': []}}
    call = json.loads(log.read_text())
    argv = call['argv']
    assert argv[argv.index('--tools') + 1] == ''
    for flag in ('-p', '--safe-mode', '--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence'):
        assert flag in argv
    assert argv[argv.index('--system-prompt') + 1] == 'Be grounded.'
    assert call['stdin'] == 'What does AF7 tell us?'
    assert os.listdir(call['cwd']) == []


def test_claude_tool_calls_round_trip(claude_cli, fake_claude, monkeypatch):
    binary, log = fake_claude
    monkeypatch.setenv('FAKE_CLAUDE_REPLY', 'TOOL_CALL {"name": "show_topomap", "arguments": {"channels": ["AF7"]}}')
    client = claude_cli.ClaudeCliClient(str(binary))
    reply = client.chat(messages=[{'role': 'user', 'content': 'Where is AF7?'}], tools=TOOLS)
    assert reply['message'] == {'content': '', 'tool_calls': [
        {'function': {'name': 'show_topomap', 'arguments': {'channels': ['AF7']}}}]}
    assert 'show_topomap' in json.loads(log.read_text())['argv'][-3]
    # The narration turn sees the picture's data as a transcript.
    _, prompt = claude_cli.render_conversation([
        {'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'Where is AF7?'},
        {'role': 'assistant', 'content': '', 'tool_calls': reply['message']['tool_calls']},
        {'role': 'tool', 'content': 'AF7: left frontal'}, {'role': 'user', 'content': 'Explain it.'}])
    assert 'show_topomap' in prompt and 'AF7: left frontal' in prompt and prompt.index('Where is AF7?') < prompt.index('Explain it.')


def test_unknown_or_malformed_tool_calls_are_plain_text(claude_cli):
    assert claude_cli.parse_tool_call('TOOL_CALL {"name": "rm_rf", "arguments": {}}', TOOLS) is None
    assert claude_cli.parse_tool_call('TOOL_CALL {not json', TOOLS) is None
    assert claude_cli.parse_tool_call('AF7 is on the left.', TOOLS) is None


def test_claude_errors_raise(claude_cli, fake_claude):
    binary, _ = fake_claude
    binary.write_text('#!/bin/sh\necho "not logged in" >&2\nexit 1\n')
    with pytest.raises(claude_cli.ClaudeCliError, match='not logged in'):
        claude_cli.ClaudeCliClient(str(binary)).chat(messages=[{'role': 'user', 'content': 'hi'}])


@pytest.fixture(scope='module')
def paper_search():
    spec = importlib.util.spec_from_file_location('fnirs_paper_search', DEMO / 'paper_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_corpus_is_open_access_and_attributed():
    papers = json.loads((DEMO / 'data' / 'papers.json').read_text())
    assert len(papers) >= 10
    for paper in papers:
        assert paper['licence'] in ('CC BY', 'CC BY-SA', 'CC0')
        assert paper['url'].startswith('https://') and paper['citation'] and paper['short']
        assert paper['passages'] and all(len(p['text']) > 80 for p in paper['passages'])


def test_paper_search_normalises_spelling_and_finds_sources(paper_search):
    assert 'hbr' in paper_search.tokens('deoxygenated haemoglobin')
    assert 'hbo' in paper_search.tokens('oxygenated hemoglobin')
    assert paper_search.tokens('2-back') == paper_search.tokens('2 back') == ['2back']
    index = paper_search.load_index()
    hits = index.search('How do HbO and HbR change with n-back working memory load?')
    assert hits and all(h['citation'] and h['url'] for h in hits)
    assert len({h['paper_id'] for h in hits}) >= 2
    assert index.search('What is the capital of France?') == []
    excerpts = paper_search.format_excerpts(hits[:1])
    assert excerpts.startswith(f'<excerpt cite="{hits[0]["short"]}"')


class _FakeVenice:
    """A local stand-in for the Venice chat API that records each request."""

    def __init__(self, reply):
        import http.server
        import threading
        fake = self
        self.requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fake.requests.append({'auth': self.headers['Authorization'], 'body': body})
                out = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': reply}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        self.server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}/chat'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def venice_client(monkeypatch):
    monkeypatch.syspath_prepend(str(DEMO))
    spec = importlib.util.spec_from_file_location('fnirs_venice_client', DEMO / 'venice_client.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_venice_requests_and_replies(venice_client):
    fake = _FakeVenice('<think>hidden</think>Rule 1 says AF7 is High.')
    try:
        client = venice_client.VeniceClient('KEY123', model='zai-org-glm-4.7-flash', url=fake.url)
        reply = client.chat(messages=[{'role': 'system', 'content': 'Be grounded.'},
                                      {'role': 'user', 'content': 'What does AF7 tell us?'}])
        assert reply == {'message': {'content': 'Rule 1 says AF7 is High.', 'tool_calls': []}}
        sent = fake.requests[0]
        assert sent['auth'] == 'Bearer KEY123'
        assert sent['body']['model'] == 'zai-org-glm-4.7-flash'
        assert sent['body']['messages'] == [{'role': 'system', 'content': 'Be grounded.'},
                                            {'role': 'user', 'content': 'What does AF7 tell us?'}]
        assert sent['body']['venice_parameters']['include_venice_system_prompt'] is False
        assert 'KEY123' not in repr(client)
    finally:
        fake.server.shutdown()
    fake = _FakeVenice('TOOL_CALL {"name": "show_topomap", "arguments": {"channels": ["AF7"]}}')
    try:
        client = venice_client.VeniceClient('KEY123', url=fake.url)
        reply = client.chat(messages=[{'role': 'user', 'content': 'Where is AF7?'}], tools=TOOLS)
        assert reply['message']['tool_calls'] == [{'function': {'name': 'show_topomap', 'arguments': {'channels': ['AF7']}}}]
    finally:
        fake.server.shutdown()


def test_admin_gets_claude_and_classes_get_venice(monkeypatch):
    pytest.importorskip('fastapi')
    monkeypatch.syspath_prepend(str(DEMO))
    import backend
    claude, venice = object(), object()
    monkeypatch.setattr(backend, 'state', {'client': claude, 'venice_client': venice,
                                           'client_error': None, 'venice_error': None})
    assert backend.pick_client('admin')[:2] == (claude, 'claude-cli')
    assert backend.pick_client('class')[:2] == (venice, 'venice')
    assert backend.pick_client(None)[:2] == (claude, 'claude-cli')
    backend.state['venice_client'] = None
    assert backend.pick_client('class')[0] is None  # never falls back to the admin's Claude
    backend.state.update(client=None, venice_client=venice)
    assert backend.pick_client(None)[:2] == (venice, 'venice')


def test_wrong_haemoglobin_signal_is_caught(monkeypatch):
    monkeypatch.syspath_prepend(str(DEMO))
    import rag_core
    import viz_tools
    rules = rag_core.load_rules()['rules']  # Rule 1 uses AF7's HbR only
    atlas = viz_tools.load_channel_atlas()
    wrong = 'AF7 is a channel that measures oxygenated haemoglobin (HbO).'
    issues = rag_core.check_answer_consistency(wrong, 'What does AF7 tell us?', rules, atlas)
    assert [(i['type'], i['channel'], i['said'], i['actual']) for i in issues] == [('signal_mismatch', 'AF7', 'HbO', 'HbR')]
    assert 'HbR' in rag_core._build_correction_message(issues, rules)
    for right in ('AF7 measures deoxygenated haemoglobin (HbR).', 'AF7 is used as HbR, not HbO.',
                  'HbO rises with load while AF7 and AFp8 are HbR channels.'):
        assert rag_core.check_answer_consistency(right, 'What does AF7 tell us?', rules, atlas) == []


PRINTED_RULES = """Rules for consequent: 0 back
----------------
IF tmb_s2_chAF7 IS High AND tmb_s2_chCPP3 IS High WITH DS 0.065, ACC 0.857
IF tmb_s1_chP3h IS Medium WITH DS 0.069, ACC 0.604

Rules for consequent: 2/3 back
----------------
IF tmb_s1_chAFF3h IS Low AND tmb_s2_chAFF6 IS Medium (MOD very) WITH DS 0.10, ACC 0.89, WGHT 0.5
IF tmb_s2_chAF8 IS Low WITH DS 0.229, ACC 0.897
"""


def test_rule_bases_in_ex_fuzzy_text_and_json_are_read(monkeypatch):
    monkeypatch.syspath_prepend(str(DEMO))
    import workspace
    data = workspace.parse_rule_base(PRINTED_RULES.encode(), 'rules.txt')
    assert [r['consequent'] for r in data['rules']] == ['0 back', '0 back', '2/3 back', '2/3 back']
    assert data['rules'][2]['antecedent'] == 'tmb_s1_chAFF3h IS Low AND tmb_s2_chAFF6 IS Medium (MOD very)'
    assert (data['rules'][2]['dominance_score'], data['rules'][2]['accuracy'], data['rules'][2]['weight']) == (0.10, 0.89, 0.5)
    assert data['feature_names'] == ['tmb_s2_chAF7', 'tmb_s2_chCPP3', 'tmb_s1_chP3h', 'tmb_s1_chAFF3h', 'tmb_s2_chAFF6', 'tmb_s2_chAF8']
    shipped = workspace.parse_rule_base((DEMO / 'data' / 'extracted_rules_final_raw.json').read_bytes(), 'r.json')
    assert len(shipped['rules']) == 6 and shipped['cv_performance_estimate'] is not None
    for bad in (b'no rules here', b'{"rules": 3}', b'IF x WITH DS 1', b'\\xff\\xfe'):
        with pytest.raises(workspace.UploadError):
            workspace.parse_rule_base(bad, 'bad.txt')


def test_uploads_change_only_their_own_workspace(monkeypatch, tmp_path):
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    monkeypatch.syspath_prepend(str(DEMO))
    monkeypatch.setenv('INTERACTIONS_DB_PATH', str(tmp_path / 'log.db'))
    monkeypatch.setenv('VENICE_API_KEY', '')
    import backend

    class Recorder:
        def __init__(self):
            self.systems = []

        def chat(self, model=None, messages=(), tools=None):
            self.systems.append(messages[0]['content'])
            return {'message': {'content': 'Rule 4 uses AF8.', 'tool_calls': []}}

    with TestClient(backend.app) as http:
        llm = Recorder()
        backend.state['client'] = llm
        ws = http.post('/api/workspace').json()['id']
        mine = {'X-Workspace': ws}
        assert http.put(f'/api/workspace/{ws}/rules?filename=rules.txt', content=PRINTED_RULES.encode()).status_code == 200
        notes = 'Our pilot used a 16-channel prefrontal cap with twelve students.'
        assert http.put(f'/api/workspace/{ws}/context', json={'text': notes}).status_code == 200
        reference = b'Our pilot report. The 2-back blocks raised oxygenated haemoglobin at left frontal channels. ' * 3
        summary = http.post(f'/api/workspace/{ws}/references?filename=pilot.txt', content=reference).json()
        assert summary['rules']['n_rules'] == 4 and summary['references'][0]['name'] == 'pilot.txt'

        assert http.get('/api/rules', headers=mine).json()['source'] == 'rules.txt'
        assert http.get('/api/rules').json()['n_rules'] == 6          # others keep the shipped study
        assert {p['bare_channel'] for p in http.get('/api/visualize/topomap', headers=mine).json()} >= {'AF7', 'AF8'}

        body = http.post('/api/ask', json={'question': 'What does AF8 tell us about 2-back load?'}, headers=mine).json()
        assert body['retrieved'][0]['antecedent'].startswith('tmb_s2_chAF8') or any(
            r['antecedent'] == 'tmb_s2_chAF8 IS Low' for r in body['retrieved'])
        assert notes in llm.systems[-1] and 'Our pilot report' in llm.systems[-1]
        assert any(p['source'] == 'upload' for p in body['papers'])

        ref_id = summary['references'][0]['id']
        assert http.delete(f'/api/workspace/{ws}/references/{ref_id}').json()['references'] == []
        assert http.delete(f'/api/workspace/{ws}/rules').json()['rules'] is None
        assert http.put(f'/api/workspace/{ws}/rules', content=b'nothing useful').status_code == 400
        assert http.get('/api/rules', headers={'X-Workspace': 'expired'}).status_code == 404
