"""UdaPlay state-machine agent: three tools, short- and long-term memory, structured output.

The agent loops: llm -> (route) -> tool-node -> llm ... -> finalize. Each tool is a
pre-defined node dispatched by name (satisfies the 'tools as state-machine nodes' goal).
"""

import json
import os
from typing import List, Optional

from pydantic import BaseModel, Field

from udaplay import index_builder as ib
from udaplay.config import CHAT_MODEL, get_openai_client


class EvaluationReport(BaseModel):
    useful: bool = Field(description="Whether the documents are enough to answer the question")
    description: str = Field(description="Explanation supporting the verdict")


class AgentAnswer(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list)
    source: str = "internal"  # "internal" | "web" | "memory"
    confidence: float = 0.5


def format_retrieval_results(raw: dict) -> List[dict]:
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        out.append({
            "Name": meta.get("Name"),
            "Platform": meta.get("Platform"),
            "YearOfRelease": meta.get("YearOfRelease"),
            "Description": meta.get("Description") or doc,
            "type": meta.get("type"),
            "level": meta.get("level"),
            "score": round(1 - dist, 6),
        })
    return out


class LongTermMemory:
    """Persistent vector memory of facts learned from web searches."""

    def __init__(self):
        client = ib.get_chroma_client()
        self._coll = client.get_or_create_collection(
            name=ib.LONGTERM, embedding_function=ib.get_embedding_function())
        self._seq = self._coll.count()

    def register(self, content: str, source: str = "", timestamp: int = 0) -> None:
        self._coll.add(
            ids=[f"mem-{self._seq}"],
            documents=[content],
            metadatas=[ib.clean_metadata({"source": source, "timestamp": timestamp})],
        )
        self._seq += 1

    def search(self, query: str, k: int = 3) -> List[dict]:
        res = self._coll.query(query_texts=[query], n_results=k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [{"content": d, "source": (m or {}).get("source"), "score": round(1 - dist, 6)}
                for d, m, dist in zip(docs, metas, dists)]


# ---- Tools (each callable is wired as a state-machine node by name) ----

def retrieve_game(query: str, index: str = ib.INDEX_RAPTOR_SAC, n_results: int = 5) -> List[dict]:
    """Semantic search over the chosen index. Returns Name/Platform/Year/Description/score/level."""
    coll = ib.get_collection(index)
    raw = coll.query(query_texts=[query], n_results=n_results)
    return format_retrieval_results(raw)


def evaluate_retrieval(question: str, retrieved_docs: List[dict], client=None,
                       model: str = CHAT_MODEL) -> EvaluationReport:
    """LLM-as-judge: are the retrieved docs sufficient to answer the question?"""
    client = client or get_openai_client()
    prompt = (
        "Decide if the documents are enough to answer the question. "
        "Respond ONLY with JSON: {\"useful\": bool, \"description\": str}.\n"
        f"Question: {question}\nDocuments: {json.dumps(retrieved_docs)[:6000]}"
    )
    resp = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content[content.find("{"): content.rfind("}") + 1])
        return EvaluationReport(**data)
    except Exception:
        return EvaluationReport(useful=False, description=f"Unparseable judge output: {content[:200]}")


def game_web_search(question: str, memory: Optional["LongTermMemory"] = None,
                    max_results: int = 5) -> dict:
    """Tavily web search fallback. Checks long-term memory first; stores useful answers back."""
    if memory is not None:
        hits = memory.search(question, k=1)
        if hits and hits[0]["score"] > 0.92:
            return {"source": "memory", "answer": hits[0]["content"], "results": []}

    from tavily import TavilyClient

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    res = client.search(question, max_results=max_results, include_answer=True)
    answer = res.get("answer") or ""
    results = [{"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
               for r in res.get("results", [])]
    if memory is not None and answer:
        memory.register(f"Q: {question}\nA: {answer}",
                        source=results[0]["url"] if results else "")
    return {"source": "web", "answer": answer, "results": results}


SYSTEM_INSTRUCTIONS = (
    "You are UdaPlay, a video-game research agent. Workflow: first use retrieve_game to search "
    "the internal database; then use evaluate_retrieval to judge if the results answer the "
    "question; if they are not useful, use game_web_search. Always cite game names or URLs. "
    "When you have enough information, give a clear final answer."
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "retrieve_game",
        "description": "Semantic search of the internal game vector database.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A question about the game industry."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "evaluate_retrieval",
        "description": "Judge whether retrieved documents are enough to answer the question.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
            "retrieved_docs": {"type": "array", "items": {"type": "object"}}},
            "required": ["question", "retrieved_docs"]}}},
    {"type": "function", "function": {
        "name": "game_web_search",
        "description": "Search the web when internal results are insufficient.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"}}, "required": ["question"]}}},
]


class Agent:
    def __init__(self, client=None, model: str = CHAT_MODEL,
                 index: str = ib.INDEX_RAPTOR_SAC, max_steps: int = 6):
        self.client = client or get_openai_client()
        self.model = model
        self.index = index
        self.max_steps = max_steps
        self.memory = LongTermMemory()
        self.sessions: dict[str, list] = {}  # short-term memory: session_id -> messages

    # ---- tool nodes, dispatched by name ----
    def _dispatch_tool(self, name: str, args: dict):
        if name == "retrieve_game":
            return retrieve_game(args["query"], index=self.index)
        if name == "evaluate_retrieval":
            return evaluate_retrieval(args["question"], args.get("retrieved_docs", []),
                                      client=self.client).model_dump()
        if name == "game_web_search":
            return game_web_search(args["question"], memory=self.memory)
        return {"error": f"unknown tool {name}"}

    def _llm_node(self, messages: list):
        return self.client.chat.completions.create(
            model=self.model, temperature=0.2, messages=messages,
            tools=TOOL_SCHEMAS, tool_choice="auto")

    def _format_answer(self, question: str, content: str) -> AgentAnswer:
        prompt = (
            "Convert the assistant's answer into JSON with keys answer, citations (list of "
            "strings), source ('internal'|'web'|'memory'), confidence (0-1). Respond ONLY JSON.\n"
            f"Question: {question}\nAnswer: {content}")
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0.0,
            messages=[{"role": "user", "content": prompt}])
        text = resp.choices[0].message.content.strip()
        try:
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return AgentAnswer(**data)
        except Exception:
            return AgentAnswer(answer=content, source="internal", confidence=0.5)

    def invoke(self, query: str, session_id: str = "default") -> AgentAnswer:
        messages = self.sessions.get(session_id, [])
        if not messages:
            messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
        messages.append({"role": "user", "content": query})

        for _ in range(self.max_steps):
            response = self._llm_node(messages)
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            tool_calls = msg.tool_calls
            if not tool_calls:
                break
            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = self._dispatch_tool(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "name": tc.function.name,
                                 "content": json.dumps(result)})

        final_content = messages[-1].get("content") or ""
        answer = self._format_answer(query, final_content)
        self.sessions[session_id] = messages  # persist short-term memory
        return answer
