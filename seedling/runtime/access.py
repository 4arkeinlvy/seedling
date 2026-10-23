"""M10: who may join a run. Join tokens, caps on issued ids, TLS contexts.

Without tokens anyone who can reach the coordinator joins, and a fresh id costs only its
probation (audit.py). With tokens, the operator hands each participant a secret:

    tokens.txt      one "name:secret" per line, "#" starts a comment
                    (make secrets with: python -c "import secrets; print(secrets.token_hex(16))")

A worker joins with its secret; every id it gets is tied to the token's name. A worker caught
lying revokes its token, which shuts out every id issued under it. The coordinator keeps
only SHA-256 hashes of the secrets in memory.
"""

from __future__ import annotations

import hashlib
import ssl
from dataclasses import dataclass, field
from pathlib import Path


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


@dataclass(frozen=True)
class Access:
    tokens: dict[str, str] = field(default_factory=dict)  # sha256(secret) -> name
    max_ids_per_token: int = 8
    max_ids: int = 100_000

    @classmethod
    def from_file(cls, path: Path | str, **limits: int) -> Access:
        tokens: dict[str, str] = {}
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, sep, secret = line.partition(":")
            if not sep or not name.strip() or len(secret.strip()) < 16:
                raise ValueError(f"{path}:{number}: expected name:secret with a secret of 16+ characters")
            tokens[_hash(secret.strip())] = name.strip()
        if not tokens:
            raise ValueError(f"{path}: no tokens")
        return cls(tokens, **limits)

    @property
    def required(self) -> bool:
        return bool(self.tokens)

    def name_of(self, secret: str) -> str | None:
        """The token's name, or None if the secret is not one of ours."""
        return self.tokens.get(_hash(secret)) if secret else None


def server_tls(cert: str, key: str) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    return context


def client_tls(ca: str | None = None) -> ssl.SSLContext:
    """Verify the coordinator against `ca` (e.g. its self-signed certificate) or the system's CAs."""
    context = ssl.create_default_context(cafile=ca)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context
