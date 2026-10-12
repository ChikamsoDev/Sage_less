"""Versioned, validated Drive state snapshots, including small trained checkpoints.

The large pretrained cache lives outside this state directory and is not copied.

SQLite databases use their online backup API. Other writers must be quiescent for
an application-consistent snapshot across checkpoints and databases. Drive FUSE
is not a POSIX transactional store: manifests, final commit markers, checksums,
and fallback to previous versions handle interrupted copies conservatively.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import time
import uuid


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid snapshot path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..", "") for part in path.parts):
        raise ValueError("Snapshot path escapes state directory")
    if path.as_posix() != value:
        raise ValueError("Noncanonical snapshot path")
    return path


class DrivePersistence:
    def __init__(self, local_dir, drive_dir):
        self.local_dir = Path(local_dir).absolute()
        self.drive_dir = Path(drive_dir).absolute()
        # The source is state only. A caller must pass a distinct workspace to its
        # executor; never point this object at a project root or home directory.
        left, right = self.local_dir.resolve(), self.drive_dir.resolve()
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("Local state and Drive backup directories must not overlap")
        if self.local_dir.is_symlink() or self.drive_dir.is_symlink():
            raise ValueError("Snapshot roots may not be symbolic links")
        self.snapshots = self.drive_dir / "snapshots"

    def _validate(self, snapshot):
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ValueError("Snapshot is not a real directory")
        manifest_path = snapshot / "manifest.json"
        marker = snapshot / "COMMITTED"
        if not marker.is_file() or marker.is_symlink() or not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("Snapshot is incomplete")
        if marker.read_text().strip() != _hash(manifest_path):
            raise ValueError("Manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
            raise ValueError("Unsupported snapshot schema")
        if len(manifest["files"]) > 100000:
            raise ValueError("Snapshot file count exceeds limit")
        names = set()
        for entry in manifest["files"]:
            relative = _relative_path(entry["path"])
            if str(relative) in names:
                raise ValueError("Duplicate snapshot path")
            names.add(str(relative))
            path = snapshot / "data" / relative
            for parent in (path, *path.parents):
                if parent == snapshot:
                    break
                if parent.is_symlink():
                    raise ValueError("Snapshot contains a symbolic link")
            if not path.is_file() or path.stat().st_size != entry["size"] or _hash(path) != entry["sha256"]:
                raise ValueError("Snapshot file checksum mismatch: " + str(relative))
            if entry.get("sqlite"):
                connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                try:
                    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("SQLite integrity failure")
                finally:
                    connection.close()
        return manifest

    def sync(self):
        if not self.local_dir.is_dir():
            raise ValueError("Local state does not exist")
        self.snapshots.mkdir(parents=True, exist_ok=True)
        if self.snapshots.is_symlink():
            raise ValueError("Snapshot directory may not be a symbolic link")
        snapshot_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
        staging = self.snapshots / (".pending-" + snapshot_id)
        final = self.snapshots / snapshot_id
        (staging / "data").mkdir(parents=True)
        entries = []
        try:
            for source in sorted(self.local_dir.rglob("*")):
                if source.is_symlink():
                    raise ValueError("Refusing state symlink: " + str(source.relative_to(self.local_dir)))
                if not source.is_file():
                    if not source.is_dir():
                        raise ValueError("Refusing special state file")
                    continue
                relative = source.relative_to(self.local_dir)
                # SQLite sidecars are superseded by an online backup; in-progress
                # atomic files are never valid persisted state.
                if source.name.endswith(("-wal", "-shm", ".lock")) or ".tmp-" in source.name:
                    continue
                target = staging / "data" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(source, "rb") as stream:
                    is_sqlite = stream.read(16) == b"SQLite format 3\x00"
                if is_sqlite:
                    original = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
                    copy = sqlite3.connect(target)
                    try:
                        original.backup(copy)
                        if copy.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                            raise ValueError("SQLite backup failed integrity check")
                    finally:
                        copy.close()
                        original.close()
                else:
                    before = source.stat()
                    shutil.copyfile(source, target)
                    after = source.stat()
                    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                        raise RuntimeError("State changed during snapshot; retry while idle")
                os.chmod(target, 0o600)
                entries.append({"path": relative.as_posix(), "size": target.stat().st_size, "sha256": _hash(target), "sqlite": is_sqlite})
            manifest = {"schema_version": 1, "snapshot_id": snapshot_id, "created": time.time(), "files": entries,
                        "consistency": "SQLite online backup; other writers must be idle", "bytes": sum(item["size"] for item in entries)}
            _write_json(staging / "manifest.json", manifest)
            # Validate only committed snapshots; writing the marker is the last
            # mutation of snapshot content before its final directory name.
            (staging / "COMMITTED").write_text(_hash(staging / "manifest.json") + "\n")
            self._validate(staging)
            os.replace(staging, final)
            _write_json(self.drive_dir / "LATEST.json", {"snapshot_id": snapshot_id})
            return {"status": "saved", "snapshot_id": snapshot_id, "files": len(entries), "bytes": manifest["bytes"], "path": str(final)}
        except BaseException:
            # A killed process may leave .pending dirs; restore ignores them.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def available(self):
        if not self.snapshots.exists():
            return []
        candidates = []
        for path in self.snapshots.iterdir():
            if not path.is_dir() or path.is_symlink() or path.name.startswith("."):
                continue
            try:
                created = float(json.loads((path / "manifest.json").read_text()).get("created", 0))
            except (OSError, ValueError, TypeError):
                created = 0
            candidates.append((created, path.name))
        return [name for _, name in sorted(candidates, reverse=True)]

    def restore(self, *, overwrite=False, snapshot_id=None):
        if self.local_dir.exists() and any(self.local_dir.iterdir()) and not overwrite:
            return {"status": "skipped", "reason": "Local state is nonempty; automatic restore never overwrites it"}
        candidates = self.available()
        if snapshot_id is not None:
            if _relative_path(snapshot_id).name != snapshot_id:
                raise ValueError("Invalid snapshot identifier")
            candidates = [snapshot_id]
        rejected = []
        for candidate in candidates:
            snapshot = self.snapshots / candidate
            try:
                manifest = self._validate(snapshot)
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
                rejected.append({"snapshot_id": candidate, "reason": str(error)})
                continue
            self.local_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = self.local_dir.with_name(self.local_dir.name + ".restore-" + uuid.uuid4().hex)
            previous = self.local_dir.with_name(self.local_dir.name + ".previous-" + uuid.uuid4().hex)
            staging.mkdir()
            try:
                for entry in manifest["files"]:
                    relative = _relative_path(entry["path"])
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(snapshot / "data" / relative, target)
                    os.chmod(target, 0o600)
                    if _hash(target) != entry["sha256"]:
                        raise ValueError("Restored file checksum mismatch")
                moved = False
                if self.local_dir.exists():
                    os.replace(self.local_dir, previous)
                    moved = True
                try:
                    os.replace(staging, self.local_dir)
                except BaseException:
                    if moved:
                        os.replace(previous, self.local_dir)
                    raise
                if moved:
                    shutil.rmtree(previous)
                return {"status": "restored", "snapshot_id": candidate, "files": len(manifest["files"]), "rejected": rejected}
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return {"status": "unavailable", "reason": "No valid committed snapshot", "rejected": rejected}

    def delete_snapshots(self, *, approved=False):
        """Explicit owner operation for privacy deletion; never called by cognition."""
        if approved is not True:
            raise PermissionError("Owner approval required to delete all backup history")
        count = len(self.available())
        if self.snapshots.exists():
            shutil.rmtree(self.snapshots)
        (self.drive_dir / "LATEST.json").unlink(missing_ok=True)
        return {"deleted_snapshots": count}
