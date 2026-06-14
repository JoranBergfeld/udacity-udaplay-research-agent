import json

from udaplay import index_builder as ib


def test_baseline_text():
    g = {"Platform": "PS1", "Name": "Gran Turismo", "YearOfRelease": 1997,
         "Description": "Racing sim."}
    assert ib.baseline_text(g) == "[PS1] Gran Turismo (1997) - Racing sim."


def test_clean_metadata_drops_none_and_serializes_lists():
    meta = {"Name": "X", "Publisher": None, "children": ["a", "b"], "level": 1, "type": "summary"}
    cleaned = ib.clean_metadata(meta)
    assert cleaned["Name"] == "X"
    assert cleaned["Publisher"] == "Unknown"        # None -> "Unknown" (Chroma rejects None)
    assert cleaned["children"] == json.dumps(["a", "b"])  # lists -> JSON string
    assert cleaned["level"] == 1


def test_load_games(tmp_path):
    (tmp_path / "001.json").write_text(json.dumps({"Name": "A", "Description": "d",
                                                   "Platform": "P", "YearOfRelease": 2000}))
    (tmp_path / "002.json").write_text(json.dumps({"Name": "B", "Description": "e",
                                                   "Platform": "Q", "YearOfRelease": 2001}))
    (tmp_path / "notes.txt").write_text("ignore me")
    games = ib.load_games(str(tmp_path))
    assert {g["Name"] for g in games} == {"A", "B"}


def raptor_node(id, text, level, type_, children, meta):
    from udaplay.raptor import Node
    return Node(id=id, text=text, level=level, type=type_, children=children,
                metadata={**meta, "type": type_, "level": level, "children": children})


def test_build_collection_from_nodes_adds_all(mocker):
    added = {}

    class FakeCollection:
        def add(self, ids, documents, metadatas):
            added["ids"] = ids
            added["documents"] = documents
            added["metadatas"] = metadatas

    nodes = [
        raptor_node("leaf-0", "doc a", 0, "leaf", [], {"Name": "A", "Publisher": None}),
        raptor_node("summary-l1-c0", "summary", 1, "summary", ["leaf-0"], {}),
    ]
    ib._add_nodes(FakeCollection(), nodes)
    assert added["ids"] == ["leaf-0", "summary-l1-c0"]
    assert added["documents"] == ["doc a", "summary"]
    # metadata cleaned: None -> "Unknown", children list -> JSON string
    assert added["metadatas"][0]["Publisher"] == "Unknown"
    assert added["metadatas"][1]["children"] == '["leaf-0"]'
