"""Storage changes preserve table bytes, evidence hashes and builder output."""
from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from eval_card_backend.datafix import builders, core, table_io


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_table_reads_and_hashes_use_uncompressed_bytes(tmp_path, monkeypatch, suffix):
    raw = '{ "source": "évidence", "settings": {"temperature": 0} }\n'.encode()
    path = tmp_path / f"fixture{suffix}"
    if suffix == ".json.gz":
        path.write_bytes(gzip.compress(raw, mtime=123))
    else:
        path.write_bytes(raw)
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)

    assert table_io.read_json_bytes(path) == raw
    assert table_io.read_json(path) == json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert core.vendored_sha256(path.name) == digest
    assert core.load_table(path.name) == {**json.loads(raw), "_table_sha256": digest}


def test_gzip_write_is_reproducible_without_time_or_filename(tmp_path):
    raw = b'{"values": [1, 2, 3]}\n'
    first = tmp_path / "first.json.gz"
    second = tmp_path / "different-name.json.gz"
    table_io.write_json_bytes(first, raw)
    initial = first.read_bytes()
    table_io.write_json_bytes(second, raw)
    table_io.write_json_bytes(first, raw)

    assert initial == first.read_bytes() == second.read_bytes()
    assert initial[:3] == b"\x1f\x8b\x08"
    assert initial[3] == 0  # No optional fields, including the original filename.
    assert initial[4:8] == b"\0" * 4
    assert initial[8] == 2  # Maximum-compression marker.
    assert gzip.decompress(initial) == raw


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_builders_roundtrip_canonical_json(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(builders, "DATA_DIR", tmp_path)
    table = {"settings": {"temperature": 0}, "source": "fixture"}
    name = f"nested/table{suffix}"

    assert builders.vendored_table(name) is None
    path = builders.write_table(name, table)
    initial = path.read_bytes()
    assert table_io.read_json_bytes(path) == core.canonical_json(table).encode()
    assert builders.vendored_table(name) == table
    builders.write_table(name, table)
    assert path.read_bytes() == initial
    assert sorted(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_failed_replace_preserves_table_and_removes_temporary_file(
    tmp_path, monkeypatch, suffix,
):
    path = tmp_path / f"table{suffix}"
    original = b'{"value": "original"}\n'
    table_io.write_json_bytes(path, original)
    initial = path.read_bytes()

    def fail_replace(source, destination):
        assert destination == path
        assert table_io.read_json_bytes(path) == original
        raise OSError("replacement refused")

    monkeypatch.setattr(table_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement refused"):
        table_io.write_json_bytes(path, b'{"value": "replacement"}\n')
    assert path.read_bytes() == initial
    assert list(tmp_path.iterdir()) == [path]


def test_malformed_gzip_is_not_treated_as_json(tmp_path):
    path = tmp_path / "table.json.gz"
    path.write_bytes(b'{"looks": "like plain JSON"}')
    with pytest.raises(gzip.BadGzipFile):
        table_io.read_json(path)
