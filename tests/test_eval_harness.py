from udaplay import eval_harness as eh


def test_query_set_nonempty_and_shaped():
    assert len(eh.QUERY_SET) >= 10
    for q in eh.QUERY_SET:
        assert "query" in q and "expected" in q  # expected: list of substrings to match


def test_hit_at_k_true_when_expected_name_present():
    results = [{"Name": "Super Mario 64"}, {"Name": "Other"}]
    assert eh.hit_at_k(results, ["Mario 64"]) is True
    assert eh.hit_at_k(results, ["Halo"]) is False


def test_mean_score():
    results = [{"score": 0.8}, {"score": 0.6}]
    assert abs(eh.mean_score(results) - 0.7) < 1e-9
    assert eh.mean_score([]) == 0.0


def test_run_ablation_builds_dataframe(mocker):
    # Fake retriever: returns the expected name only for the SAC index.
    def fake_retrieve(query, index, n_results=5):
        if index == "udaplay_raptor_sac":
            return [{"Name": "Gran Turismo", "score": 0.9}]
        return [{"Name": "Nope", "score": 0.4}]

    df = eh.run_ablation(query_set=[{"query": "GT", "expected": ["Gran Turismo"]}],
                         retriever=fake_retrieve)
    assert list(df.index) == ["udaplay_baseline", "udaplay_raptor", "udaplay_raptor_sac"]
    assert df.loc["udaplay_raptor_sac", "hit_rate"] == 1.0
    assert df.loc["udaplay_baseline", "hit_rate"] == 0.0
