from wo.sim.world import _physics_cache_valid


def test_physics_cache_requires_source_content_hash(tmp_path):
    source = tmp_path / 'asset.sdf'
    source.write_text('<sdf><model name="one"><link name="base_link"/></model></sdf>')
    cache = tmp_path / 'prepared.sdf'
    cache.write_text('prepared')
    ok = tmp_path / '.ok'
    ok.write_text(str(source))
    path = tmp_path / '.path'
    path.write_text(str(cache))
    digest = tmp_path / '.source_sha256'
    import hashlib
    digest.write_text(hashlib.sha256(source.read_bytes()).hexdigest())

    assert _physics_cache_valid(source, ok, path, digest)
    source.write_text('<sdf><model name="changed"><link name="base_link"/></model></sdf>')
    assert not _physics_cache_valid(source, ok, path, digest)


def test_physics_cache_rejects_legacy_path_only_marker(tmp_path):
    source = tmp_path / 'asset.sdf'
    source.write_text('source')
    cache = tmp_path / 'prepared.sdf'
    cache.write_text('prepared')
    ok = tmp_path / '.ok'
    ok.write_text(str(source))
    path = tmp_path / '.path'
    path.write_text(str(cache))
    assert not _physics_cache_valid(source, ok, path, tmp_path / '.source_sha256')
