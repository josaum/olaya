"""Sign MCPB using an exact Keychain certificate without exporting its private key."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--identity-sha1", required=True, help="Exact existing code-signing certificate SHA1 fingerprint"
    )
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if len(args.identity_sha1) != 40 or any(c not in "0123456789abcdefABCDEF" for c in args.identity_sha1):
        parser.error("Supply a 40-character certificate SHA1 fingerprint")
    original_hash = sha(artifact)
    with tempfile.TemporaryDirectory(prefix="olaya keychain signing ") as folder:
        work = Path(folder)
        helper, signature, signer = work / "signer", work / "signature.der", work / "signer.pem"
        subprocess.run(
            [
                "xcrun",
                "clang",
                "-Wno-deprecated-declarations",
                str(ROOT / "scripts/sign-mcpb-keychain.c"),
                "-framework",
                "Security",
                "-framework",
                "CoreFoundation",
                "-o",
                str(helper),
            ],
            check=True,
        )
        subprocess.run([str(helper), args.identity_sha1, str(artifact), str(signature)], check=True)
        subprocess.run(
            [
                "openssl",
                "cms",
                "-verify",
                "-binary",
                "-inform",
                "DER",
                "-in",
                str(signature),
                "-content",
                str(artifact),
                "-noverify",
                "-out",
                "/dev/null",
                "-signer",
                str(signer),
            ],
            check=True,
        )
        actual = (
            subprocess.check_output(
                ["openssl", "x509", "-in", str(signer), "-noout", "-fingerprint", "-sha1"], text=True
            )
            .strip()
            .split("=")[-1]
            .replace(":", "")
        )
        if actual.lower() != args.identity_sha1.lower():
            raise RuntimeError("Signature signer did not match the requested identity")
        subprocess.run(["security", "verify-cert", "-c", str(signer), "-p", "codeSign"], check=True)
        certificate = subprocess.check_output(
            [
                "openssl",
                "x509",
                "-in",
                str(signer),
                "-noout",
                "-subject",
                "-issuer",
                "-dates",
                "-fingerprint",
                "-sha256",
            ],
            text=True,
        )
        # Negative check: modification must invalidate the detached signature.
        altered = work / "altered.mcpb"
        shutil.copyfile(artifact, altered)
        with altered.open("r+b") as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(bytes([first[0] ^ 1]))
        failed = subprocess.run(
            [
                "openssl",
                "cms",
                "-verify",
                "-binary",
                "-inform",
                "DER",
                "-in",
                str(signature),
                "-content",
                str(altered),
                "-noverify",
                "-out",
                "/dev/null",
            ],
            capture_output=True,
        )
        if failed.returncode == 0:
            raise RuntimeError("Tampered payload unexpectedly verified")
        if sha(artifact) != original_hash:
            raise RuntimeError("Artifact changed during signing")
        target = artifact.with_name(artifact.stem + "-signed.mcpb")
        data = signature.read_bytes()
        shutil.copy2(artifact, target)
        with target.open("ab") as stream:
            stream.write(b"MCPB_SIG_V1" + struct.pack("<I", len(data)) + data + b"MCPB_SIG_END")
        report = {
            "artifact": target.name,
            "artifact_sha256": sha(target),
            "unsigned_payload_sha256": original_hash,
            "unsigned_payload_bytes": artifact.stat().st_size,
            "identity_certificate_sha1": args.identity_sha1.upper(),
            "certificate": certificate.strip(),
            "signature_format": "MCPB_SIG_V1 detached CMS/PKCS#7 SHA256",
            "cms_signature_verified": True,
            "macos_code_signing_certificate_trusted": True,
            "tampered_payload_rejected": True,
            "private_key_exported": False,
            "mcpb_cli_2_1_2_limitation": "Its node-forge PKCS#7 verification function is unimplemented and reports unsigned; native cryptographic verification succeeded.",
        }
        report_path = target.with_suffix(".signature-verification.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        target.with_suffix(".sha256").write_text(report["artifact_sha256"] + "  " + target.name + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
