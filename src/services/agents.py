from __future__ import annotations

import json
import re
from typing import Dict, Any, Optional, List, AsyncIterator

from crewai import Agent, Task, Crew
from crewai.types.streaming import StreamChunkType
from crewai_tools import TavilySearchTool
from langfuse import observe, get_client
from litellm.caching.redis_semantic_cache import RedisSemanticCache
from nemoguardrails import LLMRails

from src.settings import SETTINGS
from src.utils.decorators import agent_response_time
from src.services.redis_cache import redis_cache
from src.services.retrieval import RetrievalService
from src.services.memory import ShortTermMemoryService
from src.services.memory import LongTermMemoryService
from src.services.litellm_client import get_qwen_llm, get_gpt_api_llm
from src.services.tools.retrieval_tool import RetrievalTool
from src.services.tools.long_term_memory_tool import LongTermMemoryTool
from src.services.tools.short_term_memory_tool import ShortTermMemoryTool



class LearningAgents:
    """Agents for the learning-assistant pipeline:

    User question → create_rag_decision_agent (decide: use RAG or answer directly)
        → If use_rag=False: create_direct_answer_agent (short direct answer, no groundedness check)
        → If use_rag=True: create_retrieval_agent → create_check_groundedness_agent
            → If SUPPORTED: return retrieval answer
            → If UNSUPPORTED: create_search_agent → return answer
    """
    def __init__(self):
        self.retrieval_tool = RetrievalTool()
        self.long_term_memory_tool = LongTermMemoryTool()
        self.short_term_memory_tool = ShortTermMemoryTool()
        # Qwen is used for routing / direct answers, OpenAI (gpt-api) for RAG & checks.
        self.qwen_llm = get_qwen_llm()
        self.gpt_llm = get_gpt_api_llm()
        
        

    def create_rag_decision_agent(self) -> Agent:
        """Create an agent that decides whether the question needs RAG (course/lecture) or not.

        This agent only returns JSON with a use_rag flag; it does not generate the final answer.
        """
        return Agent(
            role="RAG Decision Agent",
            goal="Classify whether a question requires retrieval over course materials (RAG). Output JSON only.",
            backstory="""You are a query router for an AI learning assistant.
Your job is to decide if a user question requires retrieving
information from course materials (RAG).""",
            tools=[],
            verbose=True,
            max_iterations=1,
            llm=self.qwen_llm,
        )

    def create_direct_answer_agent(self) -> Agent:
        """Create an agent that gives short, direct answers when RAG is not needed."""
        return Agent(
            role="Direct Answer Agent",
            goal=(
                "Answer questions based on popular YouTube course lectures (e.g., Stanford CS336, CS229) and also handle technical questions beyond those courses."
                "Use short, clear, easy-to-understand language."
            ),
            backstory=(
                """You are a helpful learning assistant created by Tony Bui. You can answer questions based on popular YouTube course lectures (e.g., Stanford CS336, CS229) and also handle technical questions beyond those courses."""
            ),
            tools=[],
            verbose=True,
            max_iterations=1,
            llm=self.qwen_llm,
        )

    def create_retrieval_agent(self) -> Agent:
        """Create an agent that retrieves information from the lecture materials (first step in pipeline)."""
        return Agent(
            role="Lecture Retrieval Agent",
            goal="Answer the user's question using only information retrieved from the lecture knowledge base; always cite course, lecture number, and video timestamps.",
            backstory="""You are the first step in a learning-assistant pipeline. Your job is to answer the user's question using ONLY the lecture materials.

Process:
1. Use the retrieval tool with the user's question to get relevant lecture segments from the knowledge base.
2. From the tool results, synthesize a clear answer. Use ONLY information that appears in the retrieved segments—do not add facts from general knowledge.
3. For every claim or fact in your answer, cite the source:
   - Course name and lecture number (e.g. "Course X, Lecture 3")
   - Video segment timestamps (start and end) from the retrieved metadata when available.

Rules:
- If the retrieved segments do not contain enough information to answer the question, respond with "I don't know" and do not guess or invent content.
- Do not make up course names, lecture numbers, or timestamps; only use values that appear in the retrieval results.
- Keep your answer focused and grounded in the retrieved context so the next step (groundedness check) can verify it.""",
            tools=[self.retrieval_tool, self.short_term_memory_tool],
            verbose=True,
            max_iterations=2,
            llm=self.gpt_llm,
        )

    def create_check_groundedness_agent(self) -> Agent:
        """Create an agent that checks if the retrieval answer is fully supported by the context (second step in pipeline)."""
        return Agent(
            role="Groundedness Check Agent",
            goal="Decide whether the given answer is fully supported by the retrieved context; if not, return UNSUPPORTED. Output only valid JSON.",
            backstory="""You are the groundedness gate in the learning-assistant pipeline. You receive:
- The user's question
- The answer produced by the retrieval agent
- The retrieved context (the raw lecture segments that were used)

Your task: Answer two questions.
1. Is the answer fully supported by the context?
2. If not supported, return UNSUPPORTED.

Return SUPPORTED only if:
- All factual claims in the answer can be traced to the retrieved context.
- Cited course, lecture number, and timestamps match the context.
- No extra or unsupported information was added.

Return UNSUPPORTED if ANY of the following is true (both "I don't know" and hallucination fall into UNSUPPORTED):
- The answer is "I don't know" or indicates the lecture had no answer.
- A factual claim in the answer is NOT present or not supported in the retrieved context.
- Course name, lecture number, or timestamps do not match the context or were invented.
- The answer adds details, examples, or conclusions not in the context.
- The answer contradicts the context.

You MUST respond with exactly one JSON object, no other text or markdown:
{"status": "SUPPORTED", "reason": "brief explanation"}
or
{"status": "UNSUPPORTED", "reason": "brief explanation"}

Use exactly "SUPPORTED" or "UNSUPPORTED" for status. Your output will be parsed to decide whether to return this answer to the user or to call the search agent.""",
            verbose=True,
            max_iterations=2,
            llm=self.gpt_llm,
        )

    def create_search_agent(self) -> Agent:
        """Create an agent that searches the web when the retrieval answer was UNSUPPORTED (fallback in pipeline)."""
        return Agent(
            role="Web Search Fallback Agent",
            goal="Answer the user's question using web search when the lecture-based answer was UNSUPPORTED; cite every source with its URL.",
            backstory="""You are the fallback step in the learning-assistant pipeline. You are called only when the retrieval agent's answer was UNSUPPORTED (not grounded, "I don't know", or hallucination), so the user still needs an answer.

Process:
1. Use the short-term memory tool to get the chat history.
2. Use the search tool to find relevant, up-to-date information for the user's question.
3. Synthesize a clear answer from the search results. Prefer authoritative or educational sources when possible.
4. Always cite your sources in a structured way: for each fact or claim, include the source title and the full URL. Format example: "[Source: Title (URL)]" or a short "Sources:" list with URLs at the end.

Rules:
- Consider the chat history to create the appropriate search queries.
- Base your answer only on what you found in the search results; do not invent facts or URLs.
- If search results do not contain enough to answer the question, say "I don't know" and do not guess.
- Your final response should be the answer to the user plus a clear list or inline citations with URLs so the user can verify.""",
            tools=[TavilySearchTool(), self.short_term_memory_tool],
            verbose=True,
            max_iterations=2,
            llm=self.gpt_llm,
        )

def _format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Format message list as readable chat history for prompts."""
    if not messages:
        return "No previous conversation."
    lines = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "No previous conversation."


def _format_retrieved_context(chunks: list[Dict[str, Any]]) -> str:
    """Format retrieved chunks for the groundedness checker."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        doc = chunk.get("document", "")
        meta = chunk.get("metadata", {}) or {}
        parts.append(
            f"[Segment {i}]\n"
            f"Content: {doc}\n"
            f"Metadata: {json.dumps(meta, default=str)}\n"
        )
    return "\n".join(parts) if parts else "(No segments retrieved)"


def _parse_rag_decision_result(raw: str) -> Dict[str, Any]:
    """Extract JSON from RAG decision agent output. Default to use_rag=True if unparseable (safe: run RAG)."""
    text = raw.strip()
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        out = json.loads(text[start : i + 1])
                        use_rag = out.get("use_rag", True)
                        if not isinstance(use_rag, bool):
                            use_rag = str(use_rag).strip().lower() in ("true", "1", "yes")
                        return {"use_rag": use_rag}
                    except json.JSONDecodeError:
                        break
    # Fallback: look for use_rag false
    lower = text.lower()
    if '"use_rag": false' in lower or "'use_rag': false" in lower:
        return {"use_rag": False}
    return {"use_rag": True}


def _parse_groundedness_result(raw: str) -> Dict[str, str]:
    """Extract JSON from groundedness checker output; default to UNSUPPORTED if unparseable."""
    text = raw.strip()
    # Try to find a JSON object: from first { to matching }
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    # Fallback: look for SUPPORTED or UNSUPPORTED in text
    u = text.upper()
    if "UNSUPPORTED" in u:
        return {"status": "UNSUPPORTED", "reason": text[:200] or "Could not parse checker output"}
    if "SUPPORTED" in u:
        return {"status": "SUPPORTED", "reason": text[:200]}
    return {"status": "UNSUPPORTED", "reason": text[:200] or "Could not parse checker output"}


def _normalize_semantic_cache_hit(raw: Any) -> Optional[Dict[str, Any]]:
    """Turn Redis semantic-cache values into ``{response, use_rag}`` for the orchestrator.

    LiteLLM may return our ``json.dumps`` payload as a dict, or (if mixed with other
    writers) OpenAI-style ``choices`` / ``content`` shapes. Returns ``None`` if no
    usable assistant text is found so callers can fall through to a real generation.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            raw = json.loads(s)
        except json.JSONDecodeError:
            return {"response": raw, "use_rag": None}
    if not isinstance(raw, dict):
        return {"response": str(raw), "use_rag": None}

    use_rag = raw.get("use_rag")

    resp = raw.get("response")
    if resp is not None:
        text = resp if isinstance(resp, str) else str(resp)
        if text.strip():
            return {"response": text, "use_rag": use_rag}

    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return {"response": content, "use_rag": use_rag}

    content = raw.get("content")
    if isinstance(content, str) and content.strip():
        return {"response": content, "use_rag": use_rag}

    return None


class LearningOrchestrator:
    """Runs the pipeline:

    RAG decision → if use_rag then retrieval → groundedness check → return answer or search fallback;
    else direct answer (no RAG) with use_rag=False.
    """

    def __init__(self):
        self.agents = LearningAgents()
        self.rag_decision_agent = self.agents.create_rag_decision_agent()
        self.direct_answer_agent = self.agents.create_direct_answer_agent()
        self.retrieval_agent = self.agents.create_retrieval_agent()
        self.check_groundedness_agent = self.agents.create_check_groundedness_agent()
        self.search_agent = self.agents.create_search_agent()
        self._retrieval_service = RetrievalService()
        self._short_term_memory = ShortTermMemoryService()
        self._long_term_memory = LongTermMemoryService()
        self.langfuse = get_client()
        # Semantic cache for question → answer, using same Redis instance as app
        redis_password = (
            SETTINGS.REDIS_PASSWORD.get_secret_value()
            if SETTINGS.REDIS_PASSWORD is not None
            else ""
        )
        redis_url = f"redis://:{redis_password}@{SETTINGS.REDIS_HOST}:{SETTINGS.REDIS_PORT}"
        # Slightly relaxed threshold so similar phrasings still hit.
        self.semantic_cache = RedisSemanticCache(
            similarity_threshold=0.6,
            redis_url=redis_url,
            embedding_model="text-embedding-3-small",
        )

    async def _get_cached_answer(
        self,
        question: str,
        user_id: str,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Try to fetch a semantically similar cached answer for this question."""
        if not self.semantic_cache:
            print("Semantic cache is not initialized; skipping cache lookup.")
            return None

        messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]
        try:
            cached = await self.semantic_cache.async_get_cache(
                key="learning-session",
                messages=messages,
                metadata={"user_id": user_id, "session_id": session_id},
            )
        except Exception as e:
            print(f"Semantic cache lookup failed with error: {e}")
            return None

        if cached is None:
            print(f"Semantic cache MISS for question: {question}")
            return None

        print(f"Semantic cache RAW HIT for question: {question}, type={type(cached).__name__}")

        normalized = _normalize_semantic_cache_hit(cached)
        if normalized is not None:
            return normalized

        print(
            "Semantic cache HIT but payload had no usable assistant text; treating as MISS."
        )
        return None

    @redis_cache.cache(ttl=10)
    async def _compute_answer(
        self,
        question: str,
        session_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Run the pipeline (guardrails or orchestrator) and return the answer. Cached by redis_cache (exact key)."""
        rails_service: Optional[LLMRails] = getattr(
            self, "_current_rails_service", None
        )
        if rails_service:
            messages: List[Dict[str, Any]] = [
                {
                    "role": "context",
                    "content": {"user_id": user_id, "session_id": session_id},
                },
                {"role": "user", "content": question},
            ]
            guardrails_result = await rails_service.generate_async(messages=messages)
            
            if isinstance(guardrails_result, dict):
                response = guardrails_result.get("content", "") or ""
            elif isinstance(guardrails_result, str):
                response = guardrails_result
            else:
                response = str(guardrails_result) if guardrails_result else ""
            print(f"Guardrails result: {response}")
            # await self._append_to_chat_history(session_id, user_id, question, response)
            return {"response": response, "use_rag": None}
        print("No guardrails service provided, proceeding with LLM generation.")
        result = await self.answer_question(question, session_id, user_id)
        print(f"Response: {result.get('response', '')}")
        return result

    async def _get_chat_history_str(
        self, session_id: Optional[str], user_id: Optional[str]
    ) -> str:
        """Load chat history from short-term memory and format for prompts."""
        if not session_id or not user_id:
            return "No previous conversation."
        ctx = await self._short_term_memory.get_conversation_context(
            session_id, user_id
        )
        if not ctx:
            return "No previous conversation."
        messages = (ctx.get("context") or {}).get("messages") or []
        return _format_chat_history(messages)

    async def _append_to_chat_history(
        self,
        session_id: Optional[str],
        user_id: Optional[str],
        user_message: str,
        assistant_message: str,
        ttl_minutes: int = 60,
        max_messages: int = 20,
    ) -> None:
        """Append a user/assistant turn to short-term memory (Redis)."""
        if not session_id or not user_id:
            return
        try:
            print(f"Appending to chat history for session {session_id} and user {user_id}")
            existing = await self._short_term_memory.get_conversation_context(
                session_id, user_id
            )
            messages = (existing or {}).get("context", {}).get("messages") or []
            if not isinstance(messages, list):
                messages = []

            u = (user_message or "").strip()
            a = (assistant_message or "").strip()
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})

            if max_messages and len(messages) > max_messages:
                messages = messages[-max_messages:]

            await self._short_term_memory.store_conversation_context(
                session_id=session_id,
                user_id=user_id,
                context={"messages": messages},
                ttl_minutes=ttl_minutes,
            )
            print(f"Chat history updated for session {session_id} and user {user_id}: {messages}")
        except Exception as e:
            print(f"Error updating conversation context: {e}")

    @observe(name="learning-orchestrator")
    async def answer_question(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        langfuse_client = get_client()
        if user_id and session_id:
            chat_history_str = await self._get_chat_history_str(session_id, user_id)

        print(f"Chat history: {chat_history_str}")

        # 0. RAG decision: is this question about technical/course content (use RAG) or not?
        rag_decision_task = Task(
            description="""You are a query router for an AI learning assistant.

Your job is to decide if a user question requires retrieving
information from course materials (RAG).
Consider the chat history to make the decision.
If the question is about these technical topics,
the system should retrieve course material.

Return JSON only.

Output format:
{"use_rag": true}
or
{"use_rag": false}

Decision rules:

Return {"use_rag": true} if the question:
- asks about AI, ML, DL, NLP, or LLM concepts
- asks about technical implementation of AI systems
- asks about system architecture or system design
- asks about DevOps, MLOps, or AI infrastructure
- asks about programming or technical explanations related to these topics
- asks about lecture content or course material

Return {"use_rag": false} if the question:
- is greeting or small talk
- asks what the assistant can do
- asks about the assistant itself
- is unrelated to technology or AI
- is general conversation

Examples:

User Question:
"Explain what a large language model is"

Output:
{"use_rag": true}

User Question:
"What is the difference between CNN and RNN?"

Output:
{"use_rag": true}

User Question:
"What is Kubernetes used for?"

Output:
{"use_rag": true}

User Question:
"How does a RAG system work?"

Output:
{"use_rag": true}

User Question:
"How should we design an AI system architecture?"

Output:
{"use_rag": true}

User Question:
"What is the capital of France?"

Output:
{"use_rag": false}

User Question:
"How are you today?"

Output:
{"use_rag": false}

Now classify the following question.

Chat history (recent conversation with the user):
{{chat_history}}

User Question:
"{{question}}"

Output:
""",
            expected_output='A single JSON object: {"use_rag": true} or {"use_rag": false}.',
            agent=self.rag_decision_agent,
        )
        rag_decision_crew = Crew(agents=[self.rag_decision_agent], tasks=[rag_decision_task])
        with langfuse_client.start_as_current_observation(
            name="rag_decision_agent",
            as_type="agent",
            input={"question": question, "chat_history": chat_history_str},
        ) as obs:
            rag_result = await rag_decision_crew.kickoff_async(
                inputs={"question": question, "chat_history": chat_history_str}
            )
            decision = _parse_rag_decision_result(rag_result.raw)
            obs.update(output=decision)

        if not decision.get("use_rag", True):
            # No RAG: get a short, direct answer from the direct-answer agent; no groundedness check.
            direct_answer_task = Task(
                description="""You are a helpful learning assistant created by Tony Bui. You can answer questions based on popular YouTube course lectures (e.g., Stanford CS336, CS229) and also handle technical questions beyond those courses.

Answer the user question using general knowledge.

Rules:

- Keep answers short, clear, and easy to understand.
- Only answer questions that are general conversation or about the assistant itself.
- Do not attempt to answer questions about course material or lecture content.
- Consider the chat history to answer the question.

Examples:

User Question:
"Hello, how can you help me?"

Answer:
Hello! I'm Tony. I can help you with your learning journey by answering questions about the course materials. How can I help you today?

User Question:
"What is the capital of France?"

Answer:
The capital of France is Paris.

Now answer the following question.

Chat history (recent conversation with the user):
{{chat_history}}

User Question:
"{{question}}"

Answer:
""",
                expected_output="A short, clear, easy-to-understand answer in natural language.",
                agent=self.direct_answer_agent,
            )
            direct_answer_crew = Crew(
                agents=[self.direct_answer_agent], tasks=[direct_answer_task]
            )
            with langfuse_client.start_as_current_observation(
                name="direct_answer_agent",
                as_type="agent",
                input={"question": question, "chat_history": chat_history_str},
            ) as obs:
                direct_result = await direct_answer_crew.kickoff_async(
                    inputs={"question": question, "chat_history": chat_history_str}
                )
                response = direct_result.raw
                obs.update(output={"direct_response": response})

            print(f"RAG decision: use_rag=False, direct response: {response[:80]}...")
            await self._append_to_chat_history(session_id, user_id, question, response)
            return {"response": response, "use_rag": False}

        # 1. Retrieval: answer from lecture materials
        retrieval_task = Task(
            description="""Answer the following question using the retrieval tool. Use ONLY the retrieved lecture segments. Cite course name, lecture number, and video timestamps. If the retrieved content does not contain enough information, say 'I don't know'.
You should use the chat history to resolve what the user is referring to and rewrite the RAG query accordingly before retrieving if the question is vague (e.g., "What is it?", "Explain this", "How does that work?").
Chat history (recent conversation with the user):
{{chat_history}}

User question:
{{question}}""",
            expected_output="A clear answer grounded in the retrieved segments, with course/lecture and timestamp citations, or 'I don't know' if not answerable.",
            agent=self.retrieval_agent,
        )
        retrieval_crew = Crew(agents=[self.retrieval_agent], tasks=[retrieval_task])
        with langfuse_client.start_as_current_observation(
            name="retrieval_agent",
            as_type="agent",
            input={"question": question, "chat_history": chat_history_str},
        ) as obs:
            retrieval_result = await retrieval_crew.kickoff_async(
                inputs={"question": question, "chat_history": chat_history_str}
            )
            retrieval_answer = retrieval_result.raw
            obs.update(output={"retrieval_answer": retrieval_answer})

        print(f"Retrieval answer: {retrieval_answer}")

        # 2. Get same context used by retrieval (for groundedness check)
        try:
            chunks = self._retrieval_service.retrieve_vector(question)
            context_str = _format_retrieved_context(chunks)
        except Exception:
            context_str = "(Retrieval context unavailable)"

        # 3. Check groundedness: is the answer fully supported by the context?
        groundedness_task = Task(
            description="""You are given:
- User question: {{question}}
- Answer to check: {{retrieval_answer}}
- Retrieved context (raw segments): {{context}}

1. Is the answer fully supported by the context?
2. If not supported, return UNSUPPORTED (this includes "I don't know" and hallucination).

Respond with exactly one JSON object: {"status": "SUPPORTED", "reason": "brief explanation"} or {"status": "UNSUPPORTED", "reason": "brief explanation"}.""",
            expected_output="A single JSON object with keys status (SUPPORTED or UNSUPPORTED) and reason.",
            agent=self.check_groundedness_agent,
        )
        groundedness_crew = Crew(
            agents=[self.check_groundedness_agent],
            tasks=[groundedness_task],
        )
        with langfuse_client.start_as_current_observation(
            name="groundedness_check",
            as_type="guardrail",
            input={
                "question": question,
                "retrieval_answer": retrieval_answer,
                # Keep context reasonably sized in Langfuse
                "context_preview": context_str[:4000],
            },
        ) as obs:
            check_result = await groundedness_crew.kickoff_async(
                inputs={
                    "question": question,
                    "retrieval_answer": retrieval_answer,
                    "context": context_str,
                }
            )
            check_raw = check_result.raw
            verdict = _parse_groundedness_result(check_raw)
            is_supported = (
                verdict.get("status", "UNSUPPORTED").strip().upper() == "SUPPORTED"
            )
            obs.update(
                output={
                    "verdict": verdict,
                    "is_supported": is_supported,
                }
            )

        print(f"Groundedness check result: {verdict}")

        # 4. Return retrieval only when SUPPORTED; otherwise run search (UNSUPPORTED covers "I don't know" and hallucination)
        if is_supported:
            await self._append_to_chat_history(
                session_id, user_id, question, retrieval_answer
            )
            return {"response": retrieval_answer, "use_rag": True}

        search_task = Task(
            description="""The lecture-based answer was unreliable. Answer the user's question using web search. Cite sources with URLs.
If the question is vague (e.g., "What is it?", "Explain this", "How does that work?"), you MUST use the chat history to resolve what the user is referring to and rewrite the search query accordingly before searching.
Chat history (recent conversation with the user):
{{chat_history}}

User question:
{{question}}""",
            expected_output="An answer based on search results with cited sources (title and URL). Say 'I don't know' if you cannot find enough information.",
            agent=self.search_agent,
        )
        search_crew = Crew(agents=[self.search_agent], tasks=[search_task])
        with langfuse_client.start_as_current_observation(
            name="search_agent",
            as_type="agent",
            input={"question": question, "reason": verdict, "chat_history": chat_history_str},
        ) as obs:
            search_result = await search_crew.kickoff_async(inputs={"question": question, "chat_history": chat_history_str})
            response = search_result.raw
            obs.update(output={"search_answer": response})

        print("Using search tool to answer the question.")
        print(f"Search answer: {response}")
        await self._append_to_chat_history(session_id, user_id, question, response)
        return {"response": response, "use_rag": True}


    @agent_response_time
    async def generate(
        self,
        question: str,
        user_id: str,
        session_id: str,
        rails_service: Optional[LLMRails] = None,
    ) -> Dict[str, Any]:
        """Get answer from the agent pipeline. Response cached by redis_cache (ttl=10) via _compute_answer.
        Chat history is loaded from short-term memory (session_id/user_id); chat_history arg is accepted for API compatibility but not used."""
        langfuse_client = self.langfuse
        with langfuse_client.start_as_current_observation(
            name="learning-session",
            as_type="trace",
            input={"question": question, "user_id": user_id, "session_id": session_id},
        ) as obs:
            cached = await self._get_cached_answer(
                question=question, user_id=user_id, session_id=session_id
            )
            if cached is not None:
                obs.update(output={**cached, "from_cache": True})
                return cached

            if session_id or user_id:
                langfuse_client.update_current_trace(
                    session_id=session_id, user_id=user_id
                )

            self._current_rails_service = rails_service
            result = await self._compute_answer(question, session_id, user_id)

            # Save to semantic cache
            if self.semantic_cache:
                messages = [{"role": "user", "content": question}]
                metadata = {"user_id": user_id, "session_id": session_id}
                value_to_store = json.dumps(result)
                try:
                    await self.semantic_cache.async_set_cache(
                        key="learning-session",
                        value=value_to_store,
                        messages=messages,
                        metadata=metadata,
                        ttl=100,
                    )
                    print("Saved answer to semantic cache (ttl=10).")
                except Exception as e:
                    print(f"ERROR during semantic cache SET: {e}")
                    raise

            obs.update(output=result)
            return result

    
    async def generate_stream(
        self,
        *,
        question: str,
        user_id: str,
        session_id: str,
        rails_service: Optional[LLMRails] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming version of `generate()`.

        Yields events token-by-token during generation using
        ``Crew(stream=True)`` + ``await crew.kickoff_async()``:

        - ``{"type": "chunk",  "content": "..."}``   — one TEXT token/fragment
        - ``{"type": "final",  "response": "...", "use_rag": bool|None, "from_cache": bool}``
        """
        langfuse_client = self.langfuse
        with langfuse_client.start_as_current_observation(
            name="learning-session-stream",
            as_type="trace",
            input={"question": question, "user_id": user_id, "session_id": session_id},
        ) as obs:
            # ── Semantic cache hit → emit final immediately, no streaming needed ──
            cached = await self._get_cached_answer(
                question=question, user_id=user_id, session_id=session_id
            )
            if cached is not None:
                obs.update(output={**cached, "from_cache": True})
                yield {
                    "type": "final",
                    "response": cached.get("response", ""),
                    "use_rag": cached.get("use_rag", None),
                    "from_cache": True,
                }
                return

            if session_id or user_id:
                langfuse_client.update_current_trace(session_id=session_id, user_id=user_id)

            # ── Guardrails path (non-streaming; rails produces a complete response) ──
            self._current_rails_service = rails_service
            if rails_service:
                messages: List[Dict[str, Any]] = [
                    {"role": "context", "content": {"user_id": user_id, "session_id": session_id}},
                    {"role": "user", "content": question},
                ]
                guardrails_result = await rails_service.generate_async(messages=messages)
                if isinstance(guardrails_result, dict):
                    response = guardrails_result.get("content", "") or ""
                elif isinstance(guardrails_result, str):
                    response = guardrails_result
                else:
                    response = str(guardrails_result) if guardrails_result else ""
                await self._append_to_chat_history(session_id, user_id, question, response)
                result = {"response": response, "use_rag": None}
                obs.update(output=result)
                yield {"type": "final", "response": response, "use_rag": None, "from_cache": False}
                return

            # ── Normal pipeline ──────────────────────────────────────────────────
            chat_history_str = await self._get_chat_history_str(session_id, user_id)

            # 0) RAG decision — routing only, no streaming needed
            rag_decision_task = Task(
                description="""You are a query router for an AI learning assistant.

Your job is to decide if a user question requires retrieving
information from course materials (RAG).
Consider the chat history to make the decision.
If the question is about these technical topics,
the system should retrieve course material.

Return JSON only.

Output format:
{"use_rag": true}
or
{"use_rag": false}

Chat history (recent conversation with the user):
{{chat_history}}

User Question:
"{{question}}"

Output:
""",
                expected_output='A single JSON object: {"use_rag": true} or {"use_rag": false}.',
                agent=self.rag_decision_agent,
            )
            rag_decision_crew = Crew(agents=[self.rag_decision_agent], tasks=[rag_decision_task])
            rag_result = await rag_decision_crew.kickoff_async(
                inputs={"question": question, "chat_history": chat_history_str}
            )
            decision = _parse_rag_decision_result(getattr(rag_result, "raw", "") or "")

            final_response: str = ""
            use_rag: Optional[bool] = None

            if not decision.get("use_rag", True):
                # ── Branch A: direct answer, no RAG ─────────────────────────────
                direct_answer_task = Task(
                    description="""Answer the user question using general knowledge.

Keep answers short, clear, and easy to understand.
Consider the chat history to answer the question.

Chat history (recent conversation with the user):
{{chat_history}}

User Question:
"{{question}}"

Answer:
""",
                    expected_output="A short, clear, easy-to-understand answer in natural language.",
                    agent=self.direct_answer_agent,
                )
                direct_answer_crew = Crew(
                    agents=[self.direct_answer_agent],
                    tasks=[direct_answer_task],
                    stream=True,
                )
                # kickoff_async with stream=True returns a CrewStreamingOutput;
                # async-iterate it to receive TEXT tokens as they are generated.
                streaming = await direct_answer_crew.kickoff_async(
                    inputs={"question": question, "chat_history": chat_history_str}
                )
                async for chunk in streaming:
                    if chunk.chunk_type == StreamChunkType.TEXT and chunk.content:
                        yield {"type": "chunk", "content": chunk.content}
                # Must exhaust the iterator before accessing .result
                final_response = streaming.result.raw or ""
                use_rag = False

            else:
                # ── Branch B-1: retrieval answer (stream) ────────────────────────
                retrieval_task = Task(
                    description="""Answer the following question using the retrieval tool. Use ONLY the retrieved lecture segments. Cite course name, lecture number, and video timestamps. If the retrieved content does not contain enough information, say 'I don't know'.
You should use the chat history to resolve what the user is referring to and rewrite the RAG query accordingly before retrieving if the question is vague.

Chat history (recent conversation with the user):
{{chat_history}}

User question:
{{question}}""",
                    expected_output="A clear answer grounded in the retrieved segments, with course/lecture and timestamp citations, or 'I don't know' if not answerable.",
                    agent=self.retrieval_agent,
                )
                retrieval_crew = Crew(
                    agents=[self.retrieval_agent],
                    tasks=[retrieval_task],
                    stream=True,
                )
                streaming = await retrieval_crew.kickoff_async(
                    inputs={"question": question, "chat_history": chat_history_str}
                )
                # Buffer retrieval tokens until groundedness passes. If we streamed them
                # immediately and then fell back to search, the client would concatenate
                # two answers (retrieval + search).
                retrieval_stream_parts: List[str] = []
                async for chunk in streaming:
                    if chunk.chunk_type == StreamChunkType.TEXT and chunk.content:
                        retrieval_stream_parts.append(chunk.content)
                retrieval_answer: str = streaming.result.raw or ""

                # ── Branch B-2: groundedness check (non-stream, routing only) ────
                try:
                    chunks = self._retrieval_service.retrieve_vector(question)
                    context_str = _format_retrieved_context(chunks)
                except Exception:
                    context_str = "(Retrieval context unavailable)"

                groundedness_task = Task(
                    description="""You are given:
- User question: {{question}}
- Answer to check: {{retrieval_answer}}
- Retrieved context (raw segments): {{context}}

Respond with exactly one JSON object: {"status": "SUPPORTED", "reason": "brief explanation"} or {"status": "UNSUPPORTED", "reason": "brief explanation"}.""",
                    expected_output="A single JSON object with keys status (SUPPORTED or UNSUPPORTED) and reason.",
                    agent=self.check_groundedness_agent,
                )
                groundedness_crew = Crew(
                    agents=[self.check_groundedness_agent],
                    tasks=[groundedness_task],
                )
                check_result = await groundedness_crew.kickoff_async(
                    inputs={
                        "question": question,
                        "retrieval_answer": retrieval_answer,
                        "context": context_str,
                    }
                )
                verdict = _parse_groundedness_result(getattr(check_result, "raw", "") or "")
                is_supported = verdict.get("status", "UNSUPPORTED").strip().upper() == "SUPPORTED"

                if is_supported:
                    if retrieval_stream_parts:
                        for part in retrieval_stream_parts:
                            yield {"type": "chunk", "content": part}
                    elif retrieval_answer:
                        yield {"type": "chunk", "content": retrieval_answer}
                    final_response = retrieval_answer
                    use_rag = True
                else:
                    # ── Branch B-3: web search fallback (stream) ─────────────────
                    search_task = Task(
                        description="""The lecture-based answer was unreliable. Answer the user's question using web search. Cite sources with URLs.
If the question is vague, you MUST use the chat history to resolve what the user is referring to and rewrite the search query accordingly before searching.

Chat history (recent conversation with the user):
{{chat_history}}

User question:
{{question}}""",
                        expected_output="An answer based on search results with cited sources (title and URL). Say 'I don't know' if you cannot find enough information.",
                        agent=self.search_agent,
                    )
                    search_crew = Crew(
                        agents=[self.search_agent],
                        tasks=[search_task],
                        stream=True,
                    )
                    streaming = await search_crew.kickoff_async(
                        inputs={"question": question, "chat_history": chat_history_str}
                    )
                    async for chunk in streaming:
                        if chunk.chunk_type == StreamChunkType.TEXT and chunk.content:
                            yield {"type": "chunk", "content": chunk.content}
                    final_response = streaming.result.raw or ""
                    use_rag = True

            # ── Persist history & cache, then emit final event ───────────────────
            final_response = final_response or ""
            await self._append_to_chat_history(session_id, user_id, question, final_response)

            result = {"response": final_response, "use_rag": use_rag}

            if self.semantic_cache:
                messages = [{"role": "user", "content": question}]
                metadata = {"user_id": user_id, "session_id": session_id}
                try:
                    await self.semantic_cache.async_set_cache(
                        key="learning-session",
                        value=json.dumps(result),
                        messages=messages,
                        metadata=metadata,
                        ttl=100,
                    )
                except Exception as e:
                    print(f"ERROR during semantic cache SET (stream): {e}")

            obs.update(output=result)
            yield {"type": "final", "response": final_response, "use_rag": use_rag, "from_cache": False}

