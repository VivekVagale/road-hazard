"""Download only the India part (527 MB) of the 13 GB RDD2022 archive.

Run:  python -m src.download

The figshare archive is one zip holding one zip per country. Reading the zip's
directory over HTTP range requests lets us fetch just India.zip. If the
connection drops, re-running resumes from the bytes already on disk.
"""

import struct
import zlib
from pathlib import Path

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent.parent
URL = "https://ndownloader.figshare.com/files/38030910"
MEMBER = "RDD2022/India.zip"
DEST = ROOT / "data" / "raw" / "RDD2022" / "India.zip"


def crc32(path: Path) -> int:
    c = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            c = zlib.crc32(chunk, c)
    return c


def main() -> None:
    with RemoteZip(URL) as z:
        info = z.getinfo(MEMBER)
    assert info.compress_type == 0, "inner zip must be stored uncompressed to resume by byte range"

    # the member's bytes start after its 30-byte local header + name + extra field
    head = requests.get(URL, headers={"Range": f"bytes={info.header_offset}-{info.header_offset + 29}"}).content
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + name_len + extra_len

    DEST.parent.mkdir(parents=True, exist_ok=True)
    have = DEST.stat().st_size if DEST.exists() else 0
    while have < info.compress_size:
        print(f"{have / 1e6:.0f} / {info.compress_size / 1e6:.0f} MB")
        rng = f"bytes={start + have}-{start + info.compress_size - 1}"
        with requests.get(URL, headers={"Range": rng}, stream=True, timeout=60) as r, open(DEST, "ab") as f:
            try:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"connection dropped ({e.__class__.__name__}), resuming")
        have = DEST.stat().st_size

    assert crc32(DEST) == info.CRC, "checksum mismatch: delete the file and re-run"
    print(f"ok: {DEST.relative_to(ROOT)} ({have / 1e6:.0f} MB, CRC verified)")


if __name__ == "__main__":
    main()
