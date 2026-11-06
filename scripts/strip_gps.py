"""Remove GPS location from JPEG EXIF, losslessly (image data is not re-encoded).

Other EXIF tags (orientation, focal length, lens) are kept: the pipeline uses them.

  python scripts/strip_gps.py data/toy/images/*.jpeg
"""
import struct
import sys

from PIL import Image

GPS_IFD = 0x8825


def strip(path: str) -> bool:
    exif = Image.open(path).getexif()
    if GPS_IFD not in exif:
        return False
    del exif[GPS_IFD]
    new_app1 = b"Exif\x00\x00" + exif.tobytes()
    data = open(path, "rb").read()
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    out, i = [data[:2]], 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError("bad marker")
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is image data
            out.append(data[i:])
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i:i + 2 + length]
        if marker == 0xE1 and seg[4:10] == b"Exif\x00\x00":
            seg = b"\xff\xe1" + struct.pack(">H", len(new_app1) + 2) + new_app1
        out.append(seg)
        i += 2 + length
    open(path, "wb").write(b"".join(out))
    return True


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(("stripped " if strip(p) else "no GPS   ") + p)
