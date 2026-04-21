from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'mertformer_5080_final_onefile.py'
BUILD = ROOT / 'scripts' / 'build_mertformer_5080_final_delivery.py'
DECRYPT = ROOT / 'scripts' / 'decrypt_mertformer_result_package.py'

os.environ.setdefault('MERTFORMER_SELF_BOOTSTRAP', '0')
os.environ.setdefault('MERTFORMER_ALLOW_PIP', '0')

spec = importlib.util.spec_from_file_location('mertformer_5080_final_onefile', SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_active_model_alias_points_to_repo_parity_class() -> None:
    assert module.ACTIVE_MODEL_CLASS_NAME == 'RepoParityMertFormerModel'
    assert module.LEGACY_COMPAT_MODEL_CLASS_NAME == 'LegacyOnecellMertFormerTiny'
    assert module.MertFormerTiny is module.RepoParityMertFormerModel
    assert module.MertFormerTiny is not module.LegacyOnecellMertFormerTiny


def test_profiles_and_parser_expose_safe_and_challenge_paths() -> None:
    parser = module.build_arg_parser()
    profile_action = next(a for a in parser._actions if a.dest == 'profile')
    mode_action = next(a for a in parser._actions if a.dest == 'mode')
    assert 'safe_5080' in profile_action.choices
    assert 'challenge_5080' in profile_action.choices
    assert {'run', 'verify', 'smoke', 'benchmark', 'package', 'chat'} <= set(mode_action.choices)


def test_parity_report_contains_truth_boundary_fields(tmp_path: Path) -> None:
    report = module._m5080_write_parity_files(tmp_path)
    assert report['active_model_class'] == 'RepoParityMertFormerModel'
    assert report['legacy_compat_model_class'] == 'LegacyOnecellMertFormerTiny'
    assert report['experimental_component_policy']['policy'] == 'keep_but_be_honest'
    assert (tmp_path / 'parity_report.json').exists()


def test_source_manifest_is_portable_and_ci_safe() -> None:
    manifest = json.loads(module.MERTFORMER_SOURCE_MANIFEST)
    serialized = json.dumps(manifest, ensure_ascii=False)
    forbidden_local_prefix = '/Users/' + 'mertyunlu/Desktop/'
    assert forbidden_local_prefix not in serialized
    assert manifest['repo_root'] == '<REPO_ROOT>'
    assert manifest['canonical_output'] == '<REPO_ROOT>/scripts/mertformer_5080_final_onefile.py'
    assert all(
        entry.get('abs_path', '').startswith('<REPO_ROOT>/')
        for entry in manifest['files']
        if entry.get('abs_path')
    )


def test_delivery_and_decrypt_helpers_compile() -> None:
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(str(BUILD), doraise=True)
    py_compile.compile(str(DECRYPT), doraise=True)
