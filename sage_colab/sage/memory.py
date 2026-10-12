"""Auditable evidence memory. Similarity is a retrieval aid, never a truth label."""
from __future__ import annotations

import array
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid
from typing import Any

ENCODER_VERSION = "signed-token-hash-v1-384"
DIMENSIONS = 384
EVIDENCE_CLASSES = {"OBSERVED", "INFERRED", "IMAGINED"}
_SECRET_KEYS = {"password", "secret", "token", "api_key", "access_token", "authorization", "grant", "credentials", "private_key"}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _vector(text: str) -> array.array:
    counts = [0.0] * DIMENSIONS
    for token in re.findall(r"[\w]+", text.casefold()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % DIMENSIONS
        counts[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in counts)) or 1
    return array.array("f", (v / norm for v in counts))


def _without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_secrets(item) for key, item in value.items()
                if key.casefold() not in _SECRET_KEYS and not key.casefold().endswith("_secret")}
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


def _verified(payload: dict) -> bool:
    for value in (payload.get("status"), payload.get("verification_result"), payload.get("verification"), payload.get("receipt")):
        if value == "CONFIRMED_SUCCESS":
            return True
        if isinstance(value, dict) and value.get("status") == "CONFIRMED_SUCCESS":
            return True
    return False


class MemoryStore:
    """Append-first SQLite store, with owner deletion and separate training consent.

    Scope is an exact goal id. Optional provenance/time/evidence filters are applied
    in SQL before ranking. The deterministic hashing encoder needs no download.
    """

    def __init__(self, state_dir, encoder=None, encoder_version=None):
        self.encoder = encoder
        self.encoder_version = str(encoder_version or ENCODER_VERSION)
        if encoder is not None and not encoder_version:
            raise ValueError("A custom encoder requires a stable version identifier")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "memory.sqlite3"
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
            goal_id TEXT NOT NULL, episode_id TEXT NOT NULL, evidence_class TEXT NOT NULL,
            provenance TEXT NOT NULL, training_consent INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL, vector BLOB NOT NULL);
        CREATE INDEX IF NOT EXISTS events_scope ON events(goal_id,timestamp);
        CREATE INDEX IF NOT EXISTS events_training ON events(kind,training_consent,evidence_class);
        CREATE TABLE IF NOT EXISTS claims(
            id TEXT PRIMARY KEY,text TEXT NOT NULL,created REAL NOT NULL,
            supersedes TEXT, superseded_by TEXT, vector BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS claim_evidence(
            claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
            event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            relation TEXT NOT NULL CHECK(relation IN ('supports','contradicts')),
            PRIMARY KEY(claim_id,event_id,relation));
        CREATE TABLE IF NOT EXISTS skills(id TEXT PRIMARY KEY,payload TEXT NOT NULL,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS skill_evidence(
            skill_id TEXT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            PRIMARY KEY(skill_id,event_id));
        """)
        with self.db:
            prior = self.db.execute("SELECT value FROM metadata WHERE key='encoder'").fetchone()
            if prior and prior[0] != self.encoder_version:
                for row in self.db.execute("SELECT id,payload FROM events").fetchall():
                    self.db.execute("UPDATE events SET vector=? WHERE id=?", (self._encode(row["payload"]).tobytes(), row["id"]))
                for row in self.db.execute("SELECT id,text FROM claims").fetchall():
                    self.db.execute("UPDATE claims SET vector=? WHERE id=?", (self._encode(row["text"]).tobytes(), row["id"]))
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('encoder',?)", (self.encoder_version,))
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('schema_version','1')")

    def _encode(self, text):
        if self.encoder is None:
            return _vector(text)
        function = self.encoder.encode if hasattr(self.encoder, "encode") else self.encoder
        values = [float(item) for item in function(text)]
        if not values or len(values) > 65536 or not all(math.isfinite(item) for item in values):
            raise ValueError("Encoder must return a finite one-dimensional vector")
        norm = math.sqrt(sum(item * item for item in values)) or 1.0
        return array.array("f", (item / norm for item in values))

    def set_encoder(self, encoder, version):
        """Owner installs a local callable; no implicit network access or downloads.

        Every original record is retained. Reindexing and version update form one
        transaction, so incompatible vector spaces are never mixed.
        """
        if not version or (encoder is None and version != ENCODER_VERSION):
            raise ValueError("Supply the actual stable encoder version")
        with self._lock:
            prior_encoder, prior_version = self.encoder, self.encoder_version
            self.encoder, self.encoder_version = encoder, str(version)
            try:
                with self.db:
                    for row in self.db.execute("SELECT id,payload FROM events").fetchall():
                        self.db.execute("UPDATE events SET vector=? WHERE id=?", (self._encode(row["payload"]).tobytes(), row["id"]))
                    for row in self.db.execute("SELECT id,text FROM claims").fetchall():
                        self.db.execute("UPDATE claims SET vector=? WHERE id=?", (self._encode(row["text"]).tobytes(), row["id"]))
                    self.db.execute("INSERT OR REPLACE INTO metadata VALUES('encoder',?)", (self.encoder_version,))
            except BaseException:
                self.encoder, self.encoder_version = prior_encoder, prior_version
                raise
        return {"encoder_version": self.encoder_version, "status": "reindexed"}

    def append(self, kind, payload, goal_id="", episode_id="", evidence_class="OBSERVED", provenance="live", training_consent=False):
        if evidence_class not in EVIDENCE_CLASSES:
            raise ValueError("Unknown evidence class")
        if not isinstance(payload, dict) or not isinstance(kind, str) or not kind:
            raise ValueError("Events require a kind and a JSON object")
        if type(training_consent) is not bool:
            raise ValueError("training_consent must be an explicit boolean")
        serialized = _json(payload)
        if len(serialized.encode()) > 2_000_000:
            raise ValueError("Event exceeds 2 MB; store an evidence reference")
        event_id = str(uuid.uuid4())
        with self._lock, self.db:
            self.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", (
                event_id, kind, serialized, str(goal_id), str(episode_id), evidence_class,
                str(provenance), int(training_consent), time.time(), self._encode(serialized).tobytes()))
        return event_id

    @staticmethod
    def _event(row):
        return {key: (json.loads(row[key]) if key == "payload" else bool(row[key]) if key == "training_consent" else row[key])
                for key in row.keys() if key != "vector"}

    def events(self, goal_id="", limit=50):
        limit = max(0, min(int(limit), 10000))
        with self._lock:
            rows = self.db.execute("SELECT * FROM events" + (" WHERE goal_id=?" if goal_id else "") + " ORDER BY timestamp DESC,id DESC LIMIT ?",
                                   (goal_id, limit) if goal_id else (limit,)).fetchall()
        return [self._event(row) for row in rows]

    def retrieve(self, query, scope="", limit=5, *, provenance=None, evidence_classes=None, since=None, until=None, kinds=None, include_claims=True):
        where, params = [], []
        if scope:
            where.append("goal_id=?")
            params.append(str(scope))
        for column, values in (("provenance", provenance), ("evidence_class", evidence_classes), ("kind", kinds)):
            if values is not None:
                values = [values] if isinstance(values, str) else list(values)
                if not values:
                    return []
                where.append(column + " IN (" + ",".join("?" for _ in values) + ")")
                params.extend(values)
        for operator, bound in ((">=", since), ("<=", until)):
            if bound is not None:
                where.append("timestamp" + operator + "?")
                params.append(float(bound))
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self._lock:
            rows = self.db.execute("SELECT * FROM events" + clause, params).fetchall()
            allowed = {row["id"] for row in rows}
            candidates = []
            query_vector = self._encode(str(query))
            def similarity(blob):
                vector = array.array("f")
                vector.frombytes(blob)
                return sum(a * b for a, b in zip(query_vector, vector))
            for row in rows:
                item = self._event(row)
                item["similarity"] = similarity(row["vector"])
                item["score"] = item["similarity"] + (0.06 if row["evidence_class"] == "OBSERVED" else 0)
                item["encoder_version"] = self.encoder_version
                candidates.append(item)
            if include_claims and kinds is None:
                for row in self.db.execute("SELECT * FROM claims WHERE superseded_by IS NULL").fetchall():
                    refs = self.db.execute("SELECT event_id,relation FROM claim_evidence WHERE claim_id=?", (row["id"],)).fetchall()
                    # Claims must not expose out-of-scope/private evidence through their text.
                    if not refs or any(ref["event_id"] not in allowed for ref in refs):
                        continue
                    item = self._claim_record(row, refs)
                    item.update(kind="claim", evidence_class="INFERRED", similarity=similarity(row["vector"]), encoder_version=self.encoder_version)
                    item["score"] = item["similarity"]
                    candidates.append(item)
        candidates.sort(key=lambda item: (item["score"], item.get("timestamp", 0)), reverse=True)
        return candidates[:max(0, min(int(limit), 1000))]

    def _validate_ids(self, event_ids):
        result = []
        for event_id in dict.fromkeys(event_ids):
            row = self.db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
            if row is None:
                raise ValueError("Evidence reference does not exist: " + str(event_id))
            result.append(row)
        return result

    def claim(self, text, evidence_ids, contradictions=None, supersedes=None):
        if not str(text).strip() or not evidence_ids:
            raise ValueError("A claim requires text and supporting evidence")
        claim_id = str(uuid.uuid4())
        with self._lock, self.db:
            self._validate_ids(evidence_ids)
            self._validate_ids(contradictions or [])
            if supersedes:
                old = self.db.execute("SELECT * FROM claims WHERE id=?", (supersedes,)).fetchone()
                if old is None or old["superseded_by"]:
                    raise ValueError("Superseded claim must exist and be current")
            self.db.execute("INSERT INTO claims VALUES(?,?,?,?,?,?)", (claim_id, str(text), time.time(), supersedes, None, self._encode(str(text)).tobytes()))
            for relation, ids in (("supports", evidence_ids), ("contradicts", contradictions or [])):
                for event_id in dict.fromkeys(ids):
                    self.db.execute("INSERT INTO claim_evidence VALUES(?,?,?)", (claim_id, event_id, relation))
            if supersedes:
                self.db.execute("UPDATE claims SET superseded_by=? WHERE id=?", (claim_id, supersedes))
        return claim_id

    def _claim_record(self, row, refs=None):
        if refs is None:
            refs = self.db.execute("SELECT event_id,relation FROM claim_evidence WHERE claim_id=?", (row["id"],)).fetchall()
        record = {key: row[key] for key in row.keys() if key != "vector"}
        record["evidence_ids"] = [ref["event_id"] for ref in refs if ref["relation"] == "supports"]
        record["contradictions"] = [ref["event_id"] for ref in refs if ref["relation"] == "contradicts"]
        # Empty episode ids are one unknown episode, not independent evidence.
        episodes = self.db.execute("SELECT DISTINCT episode_id FROM events JOIN claim_evidence ON events.id=claim_evidence.event_id WHERE claim_id=? AND relation='supports'", (row["id"],)).fetchall()
        record["independent_episodes"] = sum(bool(item[0]) for item in episodes)
        record["evidence_class"] = "INFERRED"
        record["status"] = "superseded" if row["superseded_by"] else "contested" if record["contradictions"] else "supported"
        return record

    def claims(self, include_superseded=False):
        with self._lock:
            return [self._claim_record(row) for row in self.db.execute("SELECT * FROM claims" + ("" if include_superseded else " WHERE superseded_by IS NULL")).fetchall()]

    def register_skill(self, skill):
        if not isinstance(skill, dict):
            raise ValueError("Skill must be a JSON object")
        required = {"goal_pattern", "procedure", "verification", "evidence_ids"}
        if not required.issubset(skill) or not isinstance(skill["procedure"], list) or not skill["procedure"]:
            raise ValueError("Skill needs goal_pattern, nonempty procedure, verification and evidence_ids")
        if any(not isinstance(action, dict) or not isinstance(action.get("tool"), str) or not isinstance(action.get("args", {}), dict) for action in skill["procedure"]):
            raise ValueError("Skill procedures must contain typed tool actions")
        skill_id = str(uuid.uuid4())
        with self._lock, self.db:
            rows = self._validate_ids(skill["evidence_ids"])
            successes = [row for row in rows if row["evidence_class"] == "OBSERVED" and _verified(json.loads(row["payload"]))]
            if not successes:
                raise ValueError("A reusable skill requires observed, verified successful evidence")
            record = dict(skill)
            record.update(id=skill_id, preconditions=skill.get("preconditions", []), permitted_effects=skill.get("permitted_effects", []),
                          termination=skill.get("termination", "verification reports CONFIRMED_SUCCESS"),
                          provenance=skill.get("provenance", "extracted_from_verified_episodes"),
                          independent_success_episodes=len({row["episode_id"] for row in successes if row["episode_id"]}),
                          reliability={"verified_successes": len(successes), "evidence_count": len(rows), "scope": "recorded evidence only"})
            self.db.execute("INSERT INTO skills VALUES(?,?,?)", (skill_id, _json(record), time.time()))
            for row in rows:
                self.db.execute("INSERT INTO skill_evidence VALUES(?,?)", (skill_id, row["id"]))
        return skill_id

    def skills(self, query=""):
        with self._lock:
            records = [json.loads(row[0]) for row in self.db.execute("SELECT payload FROM skills").fetchall()]
        if query:
            vector = self._encode(str(query))
            records.sort(key=lambda value: sum(a * b for a, b in zip(vector, self._encode(_json(value)))), reverse=True)
        return records

    def training_experiences(self, limit=256):
        with self._lock:
            rows = self.db.execute("SELECT * FROM events WHERE kind='experience' AND training_consent=1 AND evidence_class='OBSERVED' AND provenance IN ('live','validated_simulation') ORDER BY timestamp DESC LIMIT ?", (max(0, min(int(limit), 100000)),)).fetchall()
        result = []
        for row in rows:
            payload = _without_secrets(json.loads(row["payload"]))
            payload["memory_id"] = row["id"]
            payload["episode_id"] = row["episode_id"]
            payload["provenance"] = row["provenance"]
            payload["training_consent"] = True
            result.append(payload)
        return result

    def set_training_consent(self, goal_id, consent):
        """Owner operation: revoke or grant corpus use separately from retrieval."""
        if not goal_id or type(consent) is not bool:
            raise ValueError("Specify goal_id and explicit boolean consent")
        with self._lock, self.db:
            count = self.db.execute("UPDATE events SET training_consent=? WHERE goal_id=?", (int(consent), goal_id)).rowcount
        return {"updated": count, "consent": consent, "note": "Existing trained parameters are not unlearned; reset/retrain them to remove influence."}

    def delete(self, *, goal_id=None, event_ids=None, before=None, all_data=False):
        """Owner deletion cascades derived claims/skills; snapshots need separate removal."""
        if not (goal_id is not None or event_ids is not None or before is not None or all_data is True):
            raise ValueError("Explicit deletion selector required")
        where, params = [], []
        if goal_id is not None:
            where.append("goal_id=?")
            params.append(goal_id)
        if event_ids is not None:
            event_ids = list(event_ids)
            if not event_ids:
                return {"deleted": 0}
            where.append("id IN (" + ",".join("?" for _ in event_ids) + ")")
            params.extend(event_ids)
        if before is not None:
            where.append("timestamp<?")
            params.append(float(before))
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self._lock:
            with self.db:
                ids = [row[0] for row in self.db.execute("SELECT id FROM events" + clause, params).fetchall()]
                for event_id in ids:
                    self.db.execute("DELETE FROM claims WHERE id IN (SELECT claim_id FROM claim_evidence WHERE event_id=?)", (event_id,))
                    self.db.execute("DELETE FROM skills WHERE id IN (SELECT skill_id FROM skill_evidence WHERE event_id=?)", (event_id,))
                    self.db.execute("DELETE FROM events WHERE id=?", (event_id,))
                self.db.execute("UPDATE claims SET supersedes=NULL WHERE supersedes IS NOT NULL AND supersedes NOT IN (SELECT id FROM claims)")
                self.db.execute("UPDATE claims SET superseded_by=NULL WHERE superseded_by IS NOT NULL AND superseded_by NOT IN (SELECT id FROM claims)")
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.execute("VACUUM")
        return {"deleted": len(ids), "note": "Remove older Drive snapshots separately. Deletion does not unlearn trained parameters."}

    def close(self):
        with self._lock:
            self.db.close()
