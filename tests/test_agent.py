from udaplay import agent


def test_evaluation_report_schema():
    r = agent.EvaluationReport(useful=True, description="ok")
    assert r.useful is True and r.description == "ok"


def test_agent_answer_schema():
    a = agent.AgentAnswer(answer="hi", citations=["x"], source="internal", confidence=0.9)
    assert a.source == "internal" and a.confidence == 0.9


def test_format_retrieval_results():
    raw = {
        "documents": [["[PS1] Gran Turismo (1997) - Racing."]],
        "metadatas": [[{"Name": "Gran Turismo", "Platform": "PS1",
                        "YearOfRelease": 1997, "Description": "Racing.",
                        "type": "leaf", "level": 0}]],
        "distances": [[0.12]],
    }
    out = agent.format_retrieval_results(raw)
    assert out[0]["Name"] == "Gran Turismo"
    assert out[0]["level"] == 0
    assert abs(out[0]["score"] - 0.88) < 1e-6  # 1 - distance


def test_long_term_memory_register_and_search(mocker):
    store = {}

    class FakeColl:
        def count(self):
            return len(store)

        def add(self, ids, documents, metadatas):
            for i, d, m in zip(ids, documents, metadatas):
                store[i] = (d, m)

        def query(self, query_texts, n_results):
            docs = [d for d, _ in list(store.values())[:n_results]]
            metas = [m for _, m in list(store.values())[:n_results]]
            return {"documents": [docs], "metadatas": [metas], "distances": [[0.1] * len(docs)]}

    fake_client = mocker.Mock()
    fake_client.get_or_create_collection.return_value = FakeColl()
    mocker.patch("udaplay.agent.ib.get_chroma_client", return_value=fake_client)
    mocker.patch("udaplay.agent.ib.get_embedding_function", return_value=None)

    ltm = agent.LongTermMemory()
    ltm.register("Q: when? A: 1999", source="http://x")
    res = ltm.search("when", k=1)
    assert res and "1999" in res[0]["content"]


def test_evaluate_retrieval_parses_json(mocker):
    client = mocker.Mock()
    client.chat.completions.create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(
            content='{"useful": false, "description": "not enough"}'))])
    report = agent.evaluate_retrieval("q", [{"Name": "X"}], client=client)
    assert report.useful is False and "not enough" in report.description


def test_agent_runs_tool_then_answers(mocker):
    # First LLM response: ask to call retrieve_game. Second: final content.
    def make_tool_call():
        tc = mocker.Mock()
        tc.id = "call_1"
        tc.function.name = "retrieve_game"
        tc.function.arguments = '{"query": "Gran Turismo year"}'
        return tc

    first = mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(
        content=None, tool_calls=[make_tool_call()],
        model_dump=lambda **k: {"role": "assistant", "content": None,
                                "tool_calls": [{"id": "call_1", "type": "function",
                                                "function": {"name": "retrieve_game",
                                                             "arguments": '{"query": "x"}'}}]}))])
    final = mocker.Mock(choices=[mocker.Mock(message=mocker.Mock(
        content="Gran Turismo released in 1997.", tool_calls=None,
        model_dump=lambda **k: {"role": "assistant", "content": "Gran Turismo released in 1997."}))])

    client = mocker.Mock()
    client.chat.completions.create.side_effect = [first, final]

    # Avoid real Chroma / OpenAI in LongTermMemory construction.
    mocker.patch("udaplay.agent.LongTermMemory.__init__", return_value=None)
    mocker.patch("udaplay.agent.retrieve_game", return_value=[{"Name": "Gran Turismo",
                                                               "YearOfRelease": 1997}])
    mocker.patch("udaplay.agent.Agent._format_answer",
                 return_value=agent.AgentAnswer(answer="Gran Turismo released in 1997.",
                                                citations=["Gran Turismo"], source="internal",
                                                confidence=0.9))

    a = agent.Agent(client=client)
    result = a.invoke("When was Gran Turismo released?", session_id="s1")
    assert "1997" in result.answer
    # short-term memory retained the session
    assert "s1" in a.sessions and len(a.sessions["s1"]) > 0
