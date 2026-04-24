from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from slop_code import problem_catalog


@dataclass
class _FakeResponse:
    payload: dict[str, object]
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, object]:
        return self.payload


def _create_release_tarball(
    tar_path: Path,
    *,
    valid_problem_names: list[str],
    extra_dirs: list[str] | None = None,
) -> None:
    staging = tar_path.parent / "release_src"
    repo_root = staging / "scb-problems-v1.0.0"
    repo_root.mkdir(parents=True)
    (repo_root / "README.md").write_text("# release\n", encoding="utf-8")
    for name in valid_problem_names:
        problem_dir = repo_root / name
        problem_dir.mkdir(parents=True)
        (problem_dir / "config.yaml").write_text(
            f"name: {name}\nversion: 1\nentry_file: main.py\ntags: [x]\n"
            "description: d\ncheckpoints: {checkpoint_1: {version: 1, order: 1}}\n",
            encoding="utf-8",
        )
    for extra in extra_dirs or []:
        (repo_root / extra).mkdir(parents=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(repo_root, arcname=repo_root.name)


def test_get_scbench_home_prefers_env(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom-home"
    monkeypatch.setenv("SCBENCH_HOME", str(custom))
    assert problem_catalog.get_scbench_home() == custom


def test_get_override_problem_root_prefers_env(
    monkeypatch, tmp_path: Path
) -> None:
    problems_root = tmp_path / "local-problems"
    problems_root.mkdir()
    monkeypatch.setenv("SCBENCH_PROBLEMS_PATH", str(problems_root))

    assert problem_catalog.get_override_problem_root() == problems_root


def test_get_override_problem_root_returns_none_when_unset(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SCBENCH_PROBLEMS_PATH", raising=False)

    assert problem_catalog.get_override_problem_root() is None


def test_get_scbench_home_defaults_to_cache(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.delenv("SCBENCH_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    assert problem_catalog.get_scbench_home() == home / ".cache" / "scbench"


def test_manifest_round_trip(tmp_path: Path) -> None:
    home = tmp_path / "scbench"
    manifest = problem_catalog.CatalogManifest(
        version="v1.2.3",
        commit="abc123",
    )
    problem_catalog.save_manifest(home, manifest)
    loaded = problem_catalog.load_manifest(home)
    assert loaded == manifest


def test_get_problem_root_uses_env_override_without_bootstrap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench-home"
    local_root = tmp_path / "local-problems"
    problem_dir = local_root / "alpha"
    problem_dir.mkdir(parents=True)
    (problem_dir / "config.yaml").write_text("name: alpha\n", encoding="utf-8")
    (local_root / "scripts").mkdir(parents=True)
    (local_root / "tmp").mkdir(parents=True)
    (local_root / ".agents" / "skills").mkdir(parents=True)
    monkeypatch.setenv("SCBENCH_PROBLEMS_PATH", str(local_root))
    monkeypatch.setattr(
        problem_catalog,
        "ensure_catalog_installed",
        lambda _home: (_ for _ in ()).throw(
            AssertionError("must not bootstrap managed catalog")
        ),
    )

    resolved = problem_catalog.get_problem_root(home, bootstrap=True)
    assert resolved == local_root
    assert problem_catalog.discover_problem_dirs(resolved) == [problem_dir]


def test_get_problem_root_env_override_requires_flat_problem_dirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench-home"
    local_root = tmp_path / "local-problems"
    (local_root / "nested" / "alpha").mkdir(parents=True)
    (local_root / "nested" / "alpha" / "config.yaml").write_text(
        "name: alpha\n", encoding="utf-8"
    )
    monkeypatch.setenv("SCBENCH_PROBLEMS_PATH", str(local_root))

    with pytest.raises(problem_catalog.CatalogError) as exc:
        problem_catalog.get_problem_root(home, bootstrap=False)
    assert "SCBENCH_PROBLEMS_PATH" in str(exc.value)
    assert "flat directory" in str(exc.value)


def test_ensure_catalog_installed_uses_env_override_without_sync(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench-home"
    local_root = tmp_path / "local-problems"
    problem_dir = local_root / "alpha"
    problem_dir.mkdir(parents=True)
    (problem_dir / "config.yaml").write_text("name: alpha\n", encoding="utf-8")
    (local_root / "scripts").mkdir(parents=True)
    monkeypatch.setenv("SCBENCH_PROBLEMS_PATH", str(local_root))
    monkeypatch.setattr(
        problem_catalog,
        "sync_catalog",
        lambda _home, _version: (_ for _ in ()).throw(
            AssertionError("must not sync managed catalog")
        ),
    )

    manifest = problem_catalog.ensure_catalog_installed(home)
    assert manifest.version == "env-override"
    assert manifest.commit == str(local_root.resolve())


def test_resolve_latest_release_fetches_release_and_commit(
    monkeypatch,
) -> None:
    seen_urls: list[str] = []

    def fake_get(url: str, **_kwargs: object) -> _FakeResponse:
        seen_urls.append(url)
        if url.endswith("/releases/latest"):
            return _FakeResponse(
                {"tag_name": "v1.4.0", "tarball_url": "https://tarball"}
            )
        if url.endswith("/commits/v1.4.0"):
            return _FakeResponse({"sha": "deadbeef"})
        raise AssertionError(url)

    monkeypatch.setattr(problem_catalog.httpx, "get", fake_get)

    release = problem_catalog.resolve_latest_release()
    assert release.version == "v1.4.0"
    assert release.commit == "deadbeef"
    assert release.tarball_url == "https://tarball"
    assert seen_urls == [
        problem_catalog._LATEST_RELEASE_URL,
        f"{problem_catalog._COMMITS_URL}/v1.4.0",
    ]


def test_resolve_release_fetches_tag_and_commit(monkeypatch) -> None:
    seen_urls: list[str] = []

    def fake_get(url: str, **_kwargs: object) -> _FakeResponse:
        seen_urls.append(url)
        if url.endswith("/releases/tags/v0.2"):
            return _FakeResponse(
                {"tag_name": "v0.2", "tarball_url": "https://tarball-v0.2"}
            )
        if url.endswith("/commits/v0.2"):
            return _FakeResponse({"sha": "cafebabe"})
        raise AssertionError(url)

    monkeypatch.setattr(problem_catalog.httpx, "get", fake_get)

    release = problem_catalog.resolve_release("v0.2")
    assert release.version == "v0.2"
    assert release.commit == "cafebabe"
    assert release.tarball_url == "https://tarball-v0.2"
    assert seen_urls == [
        f"{problem_catalog._RELEASE_TAGS_URL}/v0.2",
        f"{problem_catalog._COMMITS_URL}/v0.2",
    ]


def test_discover_problem_dirs_only_immediate_children_with_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "problems"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "config.yaml").write_text("name: alpha\n")
    (root / "beta").mkdir()
    (root / "nested" / "gamma").mkdir(parents=True)
    (root / "nested" / "gamma" / "config.yaml").write_text("name: gamma\n")

    discovered = problem_catalog.discover_problem_dirs(root)
    assert discovered == [root / "alpha"]


def test_sync_catalog_noop_when_requested_version_already_installed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench"
    catalog_root = problem_catalog.get_catalog_root(home)
    (catalog_root / "alpha").mkdir(parents=True)
    (catalog_root / "alpha" / "config.yaml").write_text("name: alpha\n")
    existing = problem_catalog.CatalogManifest(version="v1.0.0", commit="abc")
    problem_catalog.save_manifest(home, existing)

    monkeypatch.setattr(
        problem_catalog,
        "resolve_release",
        lambda _version: problem_catalog.ReleaseInfo(
            version="v1.0.0",
            commit="abc",
            tarball_url="https://tarball",
        ),
    )

    def _should_not_download(_url: str, _dest: Path) -> None:
        raise AssertionError("download should not be called on no-op")

    monkeypatch.setattr(
        problem_catalog, "_download_release_tarball", _should_not_download
    )

    result = problem_catalog.sync_catalog(home, "v1.0.0")
    assert result == existing


def test_sync_catalog_installs_only_valid_problem_dirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench"
    tarball = tmp_path / "release.tar.gz"
    _create_release_tarball(
        tarball,
        valid_problem_names=["alpha", "beta"],
        extra_dirs=["docs", "scripts", "tmp"],
    )
    monkeypatch.setattr(
        problem_catalog,
        "resolve_release",
        lambda _version: problem_catalog.ReleaseInfo(
            version="v1.0.0",
            commit="deadbeef",
            tarball_url="https://unused",
        ),
    )
    monkeypatch.setattr(
        problem_catalog,
        "_download_release_tarball",
        lambda _url, dest: dest.write_bytes(tarball.read_bytes()),
    )

    manifest = problem_catalog.sync_catalog(home, "v1.0.0")
    assert manifest == problem_catalog.CatalogManifest(
        version="v1.0.0",
        commit="deadbeef",
    )

    catalog_root = problem_catalog.get_catalog_root(home)
    assert sorted(path.name for path in catalog_root.iterdir()) == [
        "alpha",
        "beta",
    ]
    assert not (catalog_root / "README.md").exists()
    assert not (catalog_root / "docs").exists()
    assert not (catalog_root / "scripts").exists()
    assert not (catalog_root / "tmp").exists()

    manifest_data = json.loads(
        problem_catalog.get_manifest_path(home).read_text(encoding="utf-8")
    )
    assert manifest_data == {"version": "v1.0.0", "commit": "deadbeef"}


def test_sync_catalog_failure_leaves_existing_install_untouched(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench"
    current_root = problem_catalog.get_catalog_root(home)
    (current_root / "stable").mkdir(parents=True)
    (current_root / "stable" / "config.yaml").write_text("name: stable\n")
    existing = problem_catalog.CatalogManifest(
        version="v0.9.0",
        commit="oldsha",
    )
    problem_catalog.save_manifest(home, existing)

    bad_tarball = tmp_path / "bad-release.tar.gz"
    _create_release_tarball(
        bad_tarball,
        valid_problem_names=[],
        extra_dirs=["nested/problems/alpha"],
    )
    monkeypatch.setattr(
        problem_catalog,
        "resolve_release",
        lambda _version: problem_catalog.ReleaseInfo(
            version="v1.0.0",
            commit="newsha",
            tarball_url="https://unused",
        ),
    )
    monkeypatch.setattr(
        problem_catalog,
        "_download_release_tarball",
        lambda _url, dest: dest.write_bytes(bad_tarball.read_bytes()),
    )

    with pytest.raises(problem_catalog.CatalogError):
        problem_catalog.sync_catalog(home, "v1.0.0")

    assert (current_root / "stable" / "config.yaml").exists()
    assert problem_catalog.load_manifest(home) == existing


def test_ensure_catalog_installed_bootstraps_and_recovers_partial_install(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench"
    calls: list[str | None] = []
    bootstrapped = problem_catalog.CatalogManifest(
        version="v2.0.0",
        commit="bootstrapsha",
    )

    def fake_sync(
        target_home: Path, version: str | None
    ) -> problem_catalog.CatalogManifest:
        assert target_home == home
        calls.append(version)
        catalog_root = problem_catalog.get_catalog_root(home)
        (catalog_root / "alpha").mkdir(parents=True, exist_ok=True)
        (catalog_root / "alpha" / "config.yaml").write_text("name: alpha\n")
        problem_catalog.save_manifest(home, bootstrapped)
        return bootstrapped

    monkeypatch.setattr(problem_catalog, "sync_catalog", fake_sync)

    # Missing install bootstraps.
    assert problem_catalog.ensure_catalog_installed(home) == bootstrapped
    assert calls == [None]

    # Partial/corrupt install also bootstraps.
    problem_catalog.get_catalog_root(home).rename(home / "problems-bak")
    assert problem_catalog.ensure_catalog_installed(home) == bootstrapped
    assert calls == [None, None]


def test_validate_resume_catalog_accepts_matching_commit(
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = problem_catalog.CatalogManifest(
        version="v1.0.0",
        commit="abc123",
    )
    problem_catalog.save_manifest(home, manifest)
    catalog_root = problem_catalog.get_catalog_root(home)
    (catalog_root / "alpha").mkdir(parents=True)
    (catalog_root / "alpha" / "config.yaml").write_text("name: alpha\n")
    problem_catalog.save_run_catalog_manifest(run_dir, manifest)

    installed = problem_catalog.validate_resume_catalog(run_dir, home)
    assert installed == manifest


def test_validate_resume_catalog_rejects_commit_mismatch(
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    expected = problem_catalog.CatalogManifest(
        version="v1.0.0",
        commit="abc123",
    )
    installed = problem_catalog.CatalogManifest(
        version="v1.0.0",
        commit="def456",
    )
    problem_catalog.save_manifest(home, installed)
    catalog_root = problem_catalog.get_catalog_root(home)
    (catalog_root / "alpha").mkdir(parents=True)
    (catalog_root / "alpha" / "config.yaml").write_text("name: alpha\n")
    problem_catalog.save_run_catalog_manifest(run_dir, expected)

    with pytest.raises(problem_catalog.CatalogError) as exc:
        problem_catalog.validate_resume_catalog(run_dir, home)
    message = str(exc.value)
    assert "Run expects version v1.0.0 at commit abc123" in message
    assert "Installed catalog is version v1.0.0 at commit def456" in message
    assert "slop-code sync v1.0.0" in message


def test_validate_resume_catalog_uses_env_override_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench-home"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    local_root = tmp_path / "local-problems"
    problem_dir = local_root / "alpha"
    problem_dir.mkdir(parents=True)
    (problem_dir / "config.yaml").write_text("name: alpha\n", encoding="utf-8")
    monkeypatch.setenv("SCBENCH_PROBLEMS_PATH", str(local_root))

    expected = problem_catalog.CatalogManifest(
        version="env-override", commit=str(local_root.resolve())
    )
    problem_catalog.save_run_catalog_manifest(run_dir, expected)

    installed = problem_catalog.validate_resume_catalog(run_dir, home)
    assert installed == expected


def test_validate_resume_catalog_env_override_rejects_path_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "scbench-home"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    old_root = tmp_path / "old-problems"
    old_problem = old_root / "alpha"
    old_problem.mkdir(parents=True)
    (old_problem / "config.yaml").write_text("name: alpha\n", encoding="utf-8")
    new_root = tmp_path / "new-problems"
    new_problem = new_root / "alpha"
    new_problem.mkdir(parents=True)
    (new_problem / "config.yaml").write_text("name: alpha\n", encoding="utf-8")
    monkeypatch.setenv("SCBENCH_PROBLEMS_PATH", str(new_root))

    expected = problem_catalog.CatalogManifest(
        version="env-override", commit=str(old_root.resolve())
    )
    problem_catalog.save_run_catalog_manifest(run_dir, expected)

    with pytest.raises(problem_catalog.CatalogError) as exc:
        problem_catalog.validate_resume_catalog(run_dir, home)
    message = str(exc.value)
    assert (
        f"Run expects version env-override at commit {old_root.resolve()}"
        in message
    )
    assert (
        f"Installed catalog is version env-override at commit {new_root.resolve()}"
        in message
    )
