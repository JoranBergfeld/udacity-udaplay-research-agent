from udaplay import sac


def test_build_sac_prompt_includes_all_fields():
    game = {"Name": "Gran Turismo", "Platform": "PlayStation 1", "Genre": "Racing",
            "Publisher": "Sony", "Description": "Racing sim.", "YearOfRelease": 1997}
    p = sac.build_sac_prompt(game)
    for token in ["Gran Turismo", "PlayStation 1", "Racing", "Sony", "1997", "Racing sim."]:
        assert token in p


def test_enrich_uses_llm(mocker):
    client = mocker.Mock()
    client.chat.completions.create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(content="Enriched summary."))]
    )
    game = {"Name": "X", "Platform": "PC", "Genre": "RPG",
            "Publisher": "Y", "Description": "d", "YearOfRelease": 2020}
    assert sac.enrich(game, client=client) == "Enriched summary."


def test_enrich_many_preserves_order(mocker):
    client = mocker.Mock()
    client.chat.completions.create.side_effect = [
        mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(content=f"S{i}"))])
        for i in range(3)
    ]
    games = [{"Name": f"G{i}", "Platform": "PC", "Genre": "x",
              "Publisher": "y", "Description": "d", "YearOfRelease": 2000 + i}
             for i in range(3)]
    assert sac.enrich_many(games, client=client) == ["S0", "S1", "S2"]
