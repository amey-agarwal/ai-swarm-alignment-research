import pytest

from swarmguard.safety.preflight import check_static, scan_package, scan_source

FORBIDDEN = [
    "eval('1+1')",
    "exec('x = 1')",
    "compile('1', 'f', 'eval')",
    "__import__('os')",
    "import subprocess",
    "from subprocess import run",
    "import os\nos.system('ls')",
    "import os\nos.popen('ls')",
    "import os\nos.execv('/bin/sh', [])",
    "from os import system",
    "import os\ngetattr(os, 'system')('ls')",
    "import pickle",
    "import socket",
    "import requests",
    "import urllib.request",
    "import httpx",
    "from http import client",
    "import openai",
    "import ctypes",
    "import builtins\nbuiltins.eval('1')",
    "client.chat.completions.create(model='m', messages=[], tools=[])",
    "client.create(tool_choice='auto')",
    "payload = {'functions': []}",
    "payload = {}\npayload['tools'] = []",
    "def f(:\n  pass",
]


def test_real_package_is_clean():
    from swarmguard.config import REPO_ROOT

    v = scan_package(REPO_ROOT / "src" / "swarmguard")
    assert v == [], [str(x) for x in v]


def test_temp_copy_is_clean(repo_copy):
    assert check_static(repo_copy).status == "pass"


@pytest.mark.parametrize("snippet", FORBIDDEN)
def test_injected_forbidden_code_fails_scan(repo_copy, snippet):
    target = repo_copy / "src" / "swarmguard" / "injected.py"
    target.write_text(snippet + "\n")
    res = check_static(repo_copy)
    assert res.status == "fail", snippet
    assert "injected.py" in res.detail


def test_injection_into_existing_module_detected(repo_copy):
    p = repo_copy / "src" / "swarmguard" / "rundir.py"
    p.write_text(p.read_text() + "\nimport subprocess\n")
    assert check_static(repo_copy).status == "fail"


def test_socket_allowed_only_in_probes():
    assert scan_source("import socket\n", "safety/probes.py") == []
    assert scan_source("import socket\n", "safety/governor.py") != []


def test_network_allowed_only_in_llm_client():
    for mod in ("import openai", "import httpx", "import urllib.request"):
        assert scan_source(mod + "\n", "agents/llm_client.py") == []
        assert scan_source(mod + "\n", "agents/agent.py") != []


def test_tools_param_forbidden_even_in_llm_client():
    src = "import openai\nopenai.OpenAI().chat.completions.create(model='m', messages=[], tools=[])\n"
    assert scan_source(src, "agents/llm_client.py") != []


def test_benign_code_not_flagged():
    src = "import re, json, os\nx = re.compile('a')\nos.path.join('a', 'b')\njson.dumps({'to': 1})\n"
    assert scan_source(src, "tasks/base.py") == []


def test_compiled_module_flagged(repo_copy):
    (repo_copy / "src" / "swarmguard" / "evil.pyc").write_bytes(b"\x00")
    assert check_static(repo_copy).status == "fail"
