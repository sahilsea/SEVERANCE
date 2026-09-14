"""Real LLM agent adapter supporting high-speed Groq and sovereign local Ollama.

NON-NEGOTIABLE DESIGN PRINCIPLES:
1. The Agent Protocol takes NO principal, NO grade, and NO compartments.
   An agent never learns who is asking — it only ever receives already-gated passages.
2. Structured output: The model returns citations as structured JSON, NEVER scraped
   or parsed from prose.
3. Every citation must be an exact verbatim substring copied directly from a passage.
4. If citation verification fails in the runner, the failure reason is fed back
   into the next draft attempt prompt.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Sequence
import httpx
from contracts import Citation, Draft, Passage


class RealLlmAgent:
    """Real LLM Agent utilizing Groq (remote fast) or Ollama (air-gapped local)."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        ollama_api_base: Optional[str] = None,
        groq_model: Optional[str] = None,
        ollama_model: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.ollama_api_base = (ollama_api_base or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")).rstrip("/")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        self.backend = (backend or os.getenv("AGENT_BACKEND", "groq")).lower()

    def draft(
        self,
        question: str,
        passages: Sequence[Passage],
        feedback: Optional[str] = None,
    ) -> Draft:
        """Generate structured draft answer and verbatim citations from gated passages."""
        if not passages:
            return Draft(
                answer="No readable documents available to address this inquiry.",
                citations=[],
            )

        # Build prompt containing ONLY allowed passages
        passages_text = ""
        for p in passages:
            passages_text += f"\n--- Document: {p.doc_id} | Page: {p.page} ---\n{p.text}\n"

        system_prompt = (
            "You are SEVERANCE, the sovereign document intelligence synthesis agent for MRPL.\n"
            "You must answer the inquiry using strictly the provided readable passages.\n\n"
            "ANSWER DEPTH AND FORMATTING RULES:\n"
            "1. The 'answer' field should be a thorough, well-organized synthesis, generally 150-300 words when\n"
            "   the passages support it. Depth must come from concrete, specific detail actually present in the\n"
            "   passages (named items, quantities, procedures, thresholds, clause numbers, equipment) — never from\n"
            "   restating the question or repeating a templated sentence for each item.\n"
            "2. NEVER reuse the same sentence pattern for multiple items (e.g. do not write 'X: the document\n"
            "   mentions the need to identify X' once per hazard). Each point must add a distinct, specific fact\n"
            "   drawn from the passage text, not a paraphrase of the item's name.\n"
            "3. Write the 'answer' field as GitHub-Flavored Markdown: use '##'/'###' headings to break up\n"
            "   distinct topics, '-' bullet lists for enumerable items, and '**bold**' for key terms, so it\n"
            "   reads well when rendered.\n"
            "4. Only state facts that are supported by the provided passages. If the passages are thin or only\n"
            "   partially cover the question, say plainly and briefly what is and is not covered — a short,\n"
            "   specific answer is always better than a long one padded with filler or repetition.\n\n"
            "CRITICAL CITATION RULES:\n"
            "5. You must back your answer with exact verbatim citations from the passages.\n"
            "6. Every citation 'quote' must be an EXACT 100% VERBATIM substring copied directly from the text.\n"
            "   Do not alter, summarize, or paraphrase the quote in any way.\n"
            "7. Quotes must be between 20 and 300 characters, and contain at least 5 words.\n"
            "8. Specify the exact doc_id and integer page number matching where the quote appears.\n"
            "   (IMPORTANT: Read the page number from '--- Document: <doc_id> | Page: <page> ---'. Do NOT mistake numbered clauses or section numbers inside the text for page numbers).\n"
            "9. Respond STRICTLY in valid JSON with this exact schema:\n"
            "{\n"
            '  "answer": "Comprehensive, Markdown-formatted synthesis grounded in specific passage detail, no repeated sentence templates.",\n'
            '  "citations": [\n'
            '    {"doc_id": "document-id", "page": 1, "quote": "exact verbatim text copied from passage"}\n'
            "  ]\n"
            "}"
        )

        user_prompt = (
            f"Available Readable Passages:\n{passages_text}\n\n"
            f"User Question: {question}\n\n"
            "RESPONSE FORMAT REQUIREMENT:\n"
            "Respond STRICTLY in valid JSON with top-level keys 'answer' (a comprehensive, Markdown-formatted\n"
            "answer built from specific details in the passages — not a repeated sentence template per item —\n"
            "using headings/bullets/bold where it helps readability) and 'citations' (a list of verbatim\n"
            "quotation objects from the passages above). Do NOT output custom\n"
            "top-level keys.\n"
        )
        if feedback:
            user_prompt += (
                f"\nIMPORTANT PREVIOUS ATTEMPT FEEDBACK:\n{feedback}\n"
                "The previous citation failed verification. Make sure your quote is a literal, character-for-character "
                "verbatim copy of text in the specified document and page.\n"
            )

        # Attempt primary backend (Groq if key available, else Ollama)
        raw_response = None
        if (self.backend == "groq" or self.backend == "real") and self.groq_api_key:
            try:
                raw_response = self._call_groq(system_prompt, user_prompt)
            except Exception as e:
                print(f"[RealLlmAgent] Groq call failed ({e}). Falling back to Ollama.")

        if raw_response is None:
            try:
                raw_response = self._call_ollama(system_prompt, user_prompt)
            except Exception as e:
                print(f"[RealLlmAgent] Ollama call failed ({e}).")

        if not raw_response:
            return Draft(
                answer="Error: All LLM backends (Groq and Ollama) failed to respond.",
                citations=[],
            )

        return self._parse_draft(raw_response, passages)

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Call Groq API using HTTP client."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            # gpt-oss-20b is a reasoning model: it spends a large share of the
            # token budget on hidden reasoning before writing the JSON answer,
            # so this needs headroom well beyond the ~150+ word answer itself.
            # Kept under the account's 8000 tokens/minute rate limit so a
            # single call can't itself trip a 429.
            "max_tokens": 3072,
            "temperature": 0.1,
        }
        with httpx.Client(timeout=25.0) as client:
            res = client.post(url, headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"]

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call local Ollama endpoint."""
        url = f"{self.ollama_api_base}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }
        with httpx.Client(timeout=30.0) as client:
            res = client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            return data["message"]["content"]

    def _parse_draft(self, raw_content: str, passages: Sequence[Passage]) -> Draft:
        """Safely parse and validate structured Draft JSON."""
        clean_content = raw_content.strip()
        # Strip markdown code blocks if present
        if clean_content.startswith("```"):
            clean_content = re.sub(r"^```(?:json)?\n?", "", clean_content)
            clean_content = re.sub(r"\n?```$", "", clean_content)

        try:
            parsed = json.loads(clean_content)
        except Exception:
            # Attempt regex extraction of JSON object
            match = re.search(r"\{.*\}", clean_content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                return Draft(answer=clean_content, citations=[])

        if not isinstance(parsed, dict):
            return Draft(answer=clean_content, citations=[])

        if "answer" in parsed:
            answer = str(parsed["answer"])
        else:
            # If the model returned keys like {"possible_dangers": [...]}:
            # Intelligently convert into clean, human-readable text instead of dumping raw JSON syntax
            lines = []
            for k, v in parsed.items():
                if k == "citations":
                    continue
                title = k.replace("_", " ").title()
                if isinstance(v, list):
                    lines.append(f"{title}:")
                    for item in v:
                        lines.append(f"• {str(item).replace('_', ' ').capitalize()}")
                elif isinstance(v, dict):
                    lines.append(f"{title}:")
                    for sub_k, sub_v in v.items():
                        lines.append(f"• {sub_k.replace('_', ' ').title()}: {sub_v}")
                else:
                    lines.append(f"{title}: {v}")
            answer = "\n".join(lines) if lines else clean_content
        raw_citations = parsed.get("citations", [])

        citations: list[Citation] = []
        if isinstance(raw_citations, list):
            for c in raw_citations:
                if not isinstance(c, dict):
                    continue
                try:
                    doc_id = str(c.get("doc_id", "")).strip()
                    page = int(c.get("page", 1))
                    quote = str(c.get("quote", "")).strip()

                    # Only append if valid Citation shape
                    cit = Citation(doc_id=doc_id, page=page, quote=quote)
                    citations.append(cit)
                except Exception:
                    # Invalid citation dropped
                    pass

        return Draft(answer=answer, citations=citations)
