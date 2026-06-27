#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _stream_xor(key: bytes, nonce: bytes, data: bytes) -> bytes:
    # NOT: Bu, MFENC2 eski (legacy) artefaktlari icin ev yapimi bir akis
    # sifresi keystream'idir (HMAC-SHA256 sayac modu); standart bir AEAD
    # DEGILDIR. Sadece geriye-uyumlu cozme icin tutulur. Yeni artefaktlar
    # icin MFENC1 (ChaCha20Poly1305) kullanilmalidir.
    import hmac
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        out.extend(hmac.new(key, nonce + counter.to_bytes(8, 'big'), hashlib.sha256).digest())
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[:len(data)]))


def decrypt_mfenc(blob: bytes, password: str, aad_sha256: str) -> bytes:
    import hmac
    if blob.startswith(b'MFENC1'):
        # MFENC1: standart AEAD (ChaCha20Poly1305) — tercih edilen format.
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        salt = blob[6:22]
        nonce = blob[22:34]
        ct = blob[34:]
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 240000, dklen=32)
        return ChaCha20Poly1305(key).decrypt(nonce, ct, aad_sha256.encode('ascii'))
    if blob.startswith(b'MFENC2'):
        # MFENC2: LEGACY (eski) format — standart AEAD degil; ev yapimi
        # HMAC-SHA256 keystream XOR + encrypt-then-MAC (HMAC ciphertext
        # uzerinde, compare_digest ile sabit-zaman kontrol). Yalniz eski
        # artefaktlarin cozulebilmesi icin geriye-uyumlu olarak korunur;
        # yeni sifreleme MFENC1 (ChaCha20Poly1305) ile yapilmalidir.
        salt = blob[6:22]
        nonce = blob[22:38]
        tag = blob[38:70]
        ct = blob[70:]
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 240000, dklen=32)
        expected = hmac.new(key, aad_sha256.encode('ascii') + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError('MFENC2 authentication failed')
        return _stream_xor(key, nonce, ct)
    raise ValueError('not a supported MFENC artifact')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--artifact')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    key = json.loads(Path(args.key).read_text(encoding='utf-8'))
    artifact = Path(args.artifact or key['encrypted_artifact'])
    out = Path(args.out)
    data = decrypt_mfenc(artifact.read_bytes(), key['password'], key['aad_sha256'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    actual = sha(out)
    expected = key['original_package_sha256']
    ok = actual == expected
    print(json.dumps({'ok': ok, 'output': str(out), 'sha256': actual, 'expected_sha256': expected}, indent=2))
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
