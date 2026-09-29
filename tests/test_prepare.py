from src.prepare import block_split, voc_to_yolo

XML = """<annotation>
  <size><width>600</width><height>400</height></size>
  <object><name>D40</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>300</xmax><ymax>200</ymax></bndbox></object>
  <object><name>D44</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
</annotation>"""


def test_pothole_box_converts_to_yolo_fractions(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(XML)
    lines, counts = voc_to_yolo(p)
    assert lines == ["3 0.333333 0.375000 0.333333 0.250000"]  # D40 = class 3, centre and size as fractions
    assert counts == {"D40": 1, "D44": 1}  # D44 is counted but not kept


def test_block_split_keeps_neighbouring_frames_together():
    stems = [f"India_{i:06d}" for i in range(1000)]
    splits = block_split(stems)
    assert sum(len(v) for v in splits.values()) == 1000
    assert not set(splits["train"]) & set(splits["test"])
    # every block of 25 consecutive frames lands in exactly one split
    for split in splits.values():
        ids = {int(s.split("_")[1]) // 25 for s in split}
        for other in splits.values():
            if other is not split:
                assert not ids & {int(s.split("_")[1]) // 25 for s in other}
