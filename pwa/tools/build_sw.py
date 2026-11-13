"""Write the list of files to cache, with a content hash per file and one for the whole
app (the version), into pwa/sw.js. The per-file hashes let the service worker reuse
unchanged files from the previous version instead of downloading all ~40 MB again.

Run after changing any file in pwa/ (the GitHub Pages workflow runs it automatically):
    python pwa/tools/build_sw.py
"""
import hashlib
import re
from pathlib import Path

PWA = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {"tools", "__test"}
EXCLUDE_FILES = {"sw.js", "_headers", "README.md"}
EXCLUDE_SUFFIX = {".map", ".txt", ".md", ".py"}


def files():
    """Files of the web app that the service worker precaches (relative path, absolute path)."""
    for p in sorted(PWA.rglob("*")):
        rel = p.relative_to(PWA)
        if p.is_dir() or rel.parts[0] in EXCLUDE_DIRS or p.name in EXCLUDE_FILES or p.suffix in EXCLUDE_SUFFIX:
            continue
        if p.name.startswith("."):
            continue
        yield rel.as_posix(), p


def main():
    digest = hashlib.sha256()
    short = lambda data: hashlib.sha256(data).hexdigest()[:12]  # noqa: E731
    entries = [("./", short((PWA / "index.html").read_bytes()))]
    total = 0
    for rel, p in files():
        data = p.read_bytes()
        digest.update(rel.encode())
        digest.update(data)
        entries.append((rel, short(data)))
        total += len(data)
    version = digest.hexdigest()[:12]
    block = (
        "// <precache>\n"
        f"const VERSION = '{version}';\n"
        "const PRECACHE = [ // [path, first 12 hex digits of its SHA-256]\n"
        + "".join(f"  ['{e}', '{h}'],\n" for e, h in entries)
        + "];\n"
        "// </precache>"
    )
    sw = PWA / "sw.js"
    text = sw.read_text(encoding="utf-8")
    new, n = re.subn(r"// <precache>.*?// </precache>", block, text, flags=re.S)
    if n != 1:
        raise SystemExit("precache markers not found in sw.js")
    sw.write_text(new, encoding="utf-8")
    print(f"sw.js: version {version}, {len(entries)} entries, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
