"""Внешняя граница переносимого candidate v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_PLATFORM = "linux/amd64"
CANDIDATE_TRANSPORT = "docker-save"
CORE_IMAGE_TAG = "distil3d:cuda124-sm80-core"
MOGE_IMAGE_TAG = "distil3d:cuda124-sm80-moge"
CANDIDATE_LOCK_PATHS = (
    "containers/diffusion/requirements/core.conda.lock",
    "containers/diffusion/requirements/core.pip.lock",
    "containers/diffusion/requirements/gen3c-eval.conda.lock",
    "containers/diffusion/requirements/gen3c-eval.pip.lock",
    "containers/diffusion/requirements/vggt.conda.lock",
    "containers/diffusion/requirements/vggt.pip.lock",
    "containers/diffusion/requirements/moge-gen3c.pip.lock",
)

_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CandidateLock:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CandidateImage:
    tag: str
    image_id: str


@dataclass(frozen=True, slots=True)
class CandidateImages:
    core: CandidateImage
    moge: CandidateImage


@dataclass(frozen=True, slots=True)
class Candidate:
    schema_version: int
    candidate_id: str
    source_revision: str
    source_dirty: bool
    platform: str
    lock_revision: str
    locks: tuple[CandidateLock, ...]
    transport: str
    images: CandidateImages


def parse_lock_inventory(inventory: str) -> tuple[CandidateLock, ...]:
    """Разобрать канонический TSV от ``images.sh``."""

    lines = inventory.splitlines(keepends=True)
    if len(lines) != len(CANDIDATE_LOCK_PATHS):
        raise ValueError("lock inventory must contain exactly seven lines")

    locks: list[CandidateLock] = []
    for expected_path, line in zip(CANDIDATE_LOCK_PATHS, lines, strict=True):
        fields = line.removesuffix("\n").split("\t")
        if len(fields) != 2 or not line.endswith("\n"):
            raise ValueError("lock inventory must use canonical TSV lines")
        path, digest = fields
        if path != expected_path:
            raise ValueError("lock inventory paths are not in canonical order")
        locks.append(CandidateLock(path=path, sha256=_sha256(digest, "lock")))

    parsed = tuple(locks)
    if render_lock_inventory(parsed) != inventory:
        raise ValueError("lock inventory is not canonical")
    return parsed


def render_lock_inventory(locks: tuple[CandidateLock, ...]) -> str:
    """Собрать TSV из доверенного внутреннего объекта."""

    return "".join(f"{lock.path}\tsha256:{lock.sha256}\n" for lock in locks)


def candidate_lock_revision(locks: tuple[CandidateLock, ...]) -> str:
    """Вычислить aggregate SHA канонического TSV."""

    inventory = render_lock_inventory(locks).encode("utf-8")
    return f"sha256:{hashlib.sha256(inventory).hexdigest()}"


def build_candidate(
    *,
    source_revision: str,
    lock_revision: str,
    core_image_id: str,
    moge_image_id: str,
    lock_inventory: str,
) -> Candidate:
    """Создать candidate из внешних данных сборки."""

    if _SOURCE_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source revision must be 40 lowercase hex characters")
    core_digest = _sha256(core_image_id, "core image id")
    moge_digest = _sha256(moge_image_id, "moge image id")
    _sha256(lock_revision, "lock revision")
    locks = parse_lock_inventory(lock_inventory)
    if candidate_lock_revision(locks) != lock_revision:
        raise ValueError("lock revision does not match lock inventory")

    return Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id=(f"{source_revision[:12]}-{core_digest[:12]}-{moge_digest[:12]}"),
        source_revision=source_revision,
        source_dirty=False,
        platform=CANDIDATE_PLATFORM,
        lock_revision=lock_revision,
        locks=locks,
        transport=CANDIDATE_TRANSPORT,
        images=CandidateImages(
            core=CandidateImage(tag=CORE_IMAGE_TAG, image_id=core_image_id),
            moge=CandidateImage(tag=MOGE_IMAGE_TAG, image_id=moge_image_id),
        ),
    )


def candidate_to_mapping(candidate: Candidate) -> dict[str, object]:
    """Представить доверенный candidate как JSON-словарь."""

    return {
        "schema_version": candidate.schema_version,
        "candidate_id": candidate.candidate_id,
        "source_revision": candidate.source_revision,
        "source_dirty": candidate.source_dirty,
        "platform": candidate.platform,
        "lock_revision": candidate.lock_revision,
        "locks": [
            {"path": lock.path, "sha256": lock.sha256} for lock in candidate.locks
        ],
        "transport": candidate.transport,
        "images": {
            "core": {
                "tag": candidate.images.core.tag,
                "id": candidate.images.core.image_id,
            },
            "moge": {
                "tag": candidate.images.moge.tag,
                "id": candidate.images.moge.image_id,
            },
        },
    }


def candidate_from_mapping(value: object) -> Candidate:
    """Разобрать и проверить внешний JSON-словарь."""

    if not isinstance(value, dict):
        raise ValueError("candidate must be a JSON object")
    expected_fields = {
        "schema_version",
        "candidate_id",
        "source_revision",
        "source_dirty",
        "platform",
        "lock_revision",
        "locks",
        "transport",
        "images",
    }
    if set(value) != expected_fields:
        raise ValueError("candidate has missing or unknown fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("candidate schema version must be 1")
    source_revision = value["source_revision"]
    if (
        not isinstance(source_revision, str)
        or _SOURCE_REVISION.fullmatch(source_revision) is None
    ):
        raise ValueError("source revision must be 40 lowercase hex characters")
    if value["source_dirty"] is not False:
        raise ValueError("candidate source must be clean")
    if value["platform"] != CANDIDATE_PLATFORM:
        raise ValueError("candidate platform must be linux/amd64")
    if value["transport"] != CANDIDATE_TRANSPORT:
        raise ValueError("candidate transport must be docker-save")

    locks_value = value["locks"]
    if not isinstance(locks_value, list) or len(locks_value) != len(
        CANDIDATE_LOCK_PATHS
    ):
        raise ValueError("candidate must contain exactly seven locks")
    locks: list[CandidateLock] = []
    for expected_path, lock_value in zip(
        CANDIDATE_LOCK_PATHS,
        locks_value,
        strict=True,
    ):
        if not isinstance(lock_value, dict) or set(lock_value) != {
            "path",
            "sha256",
        }:
            raise ValueError("candidate lock has missing or unknown fields")
        if lock_value["path"] != expected_path:
            raise ValueError("candidate locks are not in canonical order")
        digest = lock_value["sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(
                "candidate lock sha256 must be 64 lowercase hex characters"
            )
        locks.append(CandidateLock(path=expected_path, sha256=digest))
    parsed_locks = tuple(locks)

    lock_revision = value["lock_revision"]
    if not isinstance(lock_revision, str):
        raise ValueError("lock revision must be a string")
    _sha256(lock_revision, "lock revision")
    if candidate_lock_revision(parsed_locks) != lock_revision:
        raise ValueError("lock revision does not match candidate locks")

    images_value = value["images"]
    if not isinstance(images_value, dict) or set(images_value) != {"core", "moge"}:
        raise ValueError("candidate images must contain exactly core and moge")
    core = _image_from_mapping(images_value["core"], "core", CORE_IMAGE_TAG)
    moge = _image_from_mapping(images_value["moge"], "moge", MOGE_IMAGE_TAG)

    candidate_id = value["candidate_id"]
    expected_id = f"{source_revision[:12]}-{core.image_id[7:19]}-{moge.image_id[7:19]}"
    if candidate_id != expected_id:
        raise ValueError("candidate id does not match source and image identities")

    return Candidate(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        candidate_id=expected_id,
        source_revision=source_revision,
        source_dirty=False,
        platform=CANDIDATE_PLATFORM,
        lock_revision=lock_revision,
        locks=parsed_locks,
        transport=CANDIDATE_TRANSPORT,
        images=CandidateImages(core=core, moge=moge),
    )


def read_candidate(candidate_file: Path) -> Candidate:
    """Прочитать и проверить внешний ``candidate.json``."""

    value = json.loads(candidate_file.read_text(encoding="utf-8"))
    return candidate_from_mapping(value)


def write_candidate(candidate_file: Path, candidate: Candidate) -> None:
    """Записать ``candidate.json`` с режимом 0644."""

    payload = json.dumps(candidate_to_mapping(candidate), indent=2, sort_keys=True)
    candidate_file.write_text(f"{payload}\n", encoding="utf-8")
    candidate_file.chmod(0o644)


def _sha256(value: str, field: str) -> str:
    prefix = "sha256:"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"{field} must use sha256:<64-lowercase-hex>")
    digest = value.removeprefix(prefix)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{field} must use sha256:<64-lowercase-hex>")
    return digest


def _image_from_mapping(value: object, name: str, tag: str) -> CandidateImage:
    if not isinstance(value, dict) or set(value) != {"tag", "id"}:
        raise ValueError(f"candidate {name} image has missing or unknown fields")
    if value["tag"] != tag:
        raise ValueError(f"candidate {name} image tag is not fixed")
    image_id = value["id"]
    if not isinstance(image_id, str):
        raise ValueError(f"candidate {name} image id must be a string")
    _sha256(image_id, f"{name} image id")
    return CandidateImage(tag=tag, image_id=image_id)
