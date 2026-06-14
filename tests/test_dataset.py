import json

from udaplay import dataset


def test_year_from_released():
    assert dataset.year_from_released("2021-12-08") == 2021
    assert dataset.year_from_released(None) is None
    assert dataset.year_from_released("") is None


def test_normalize_game_full():
    list_item = {
        "name": "Halo Infinite",
        "released": "2021-12-08",
        "platforms": [{"platform": {"name": "Xbox Series S/X"}}],
        "genres": [{"name": "Shooter"}],
    }
    detail = {
        "publishers": [{"name": "Xbox Game Studios"}],
        "description_raw": "Master Chief returns.",
    }
    g = dataset.normalize_game(list_item, detail)
    assert g["Name"] == "Halo Infinite"
    assert g["YearOfRelease"] == 2021
    assert g["Platform"] == "Xbox Series S/X"
    assert g["Genre"] == "Shooter"
    assert g["Publisher"] == "Xbox Game Studios"
    assert g["Description"] == "Master Chief returns."


def test_normalize_game_missing_fields():
    g = dataset.normalize_game({"name": "Mystery Game"}, None)
    assert g["Name"] == "Mystery Game"
    assert g["Platform"] == "Unknown"
    assert g["Genre"] == "Unknown"
    assert g["Publisher"] == "Unknown"
    assert g["Description"] is None  # signals LLM-fill needed
    assert g["YearOfRelease"] is None


def test_rate_limiter_sleeps_when_called_too_fast(mocker):
    sleeps = []
    mocker.patch("udaplay.dataset.time.sleep", side_effect=lambda s: sleeps.append(s))
    times = iter([100.0, 100.0, 100.05, 100.05])
    mocker.patch("udaplay.dataset.time.monotonic", side_effect=lambda: next(times))
    rl = dataset.RateLimiter(min_interval=0.2)
    rl.wait()  # first call, last=0 -> big delta, no sleep
    rl.wait()  # delta 0.05 < 0.2 -> sleeps ~0.15
    assert sleeps and abs(sleeps[0] - 0.15) < 1e-6


def test_client_caches_responses(tmp_path, mocker):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        resp = mocker.Mock()
        resp.json.return_value = {"results": [{"name": "X"}]}
        resp.raise_for_status.return_value = None
        return resp

    mocker.patch("udaplay.dataset.requests.get", side_effect=fake_get)
    client = dataset.RawgClient(api_key="k", cache_dir=str(tmp_path), min_interval=0.0)
    a = client._get("games", {"page": 1})
    b = client._get("games", {"page": 1})  # served from cache
    assert a == b
    assert calls["n"] == 1


def test_fill_missing_description_only_when_absent(mocker):
    client = mocker.Mock()
    client.chat.completions.create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(content="A generated blurb."))]
    )
    g = {"Name": "X", "Platform": "PC", "Genre": "RPG",
         "Publisher": "Y", "Description": None, "YearOfRelease": 2022}
    out = dataset.fill_missing_description(g, client=client)
    assert out["Description"] == "A generated blurb."

    g2 = {**g, "Description": "Already here."}
    out2 = dataset.fill_missing_description(g2, client=client)
    assert out2["Description"] == "Already here."
    # only one LLM call total (for the first game)
    assert client.chat.completions.create.call_count == 1


def test_build_dataset_no_key_uses_existing(tmp_path, monkeypatch):
    games = tmp_path / "games"
    games.mkdir()
    (games / "001.json").write_text(json.dumps({"Name": "Seed"}))
    monkeypatch.delenv("RAWG_API_KEY", raising=False)
    count = dataset.build_dataset(target_count=1000, out_dir=str(games))
    assert count == 1  # no key -> just counts existing, never raises
