"""
Post-processor and summarizer for ASR transcription text.

Uses Qwen3 LLM for cleaning up speech recognition results and
generating summaries of transcription history.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PostProcessor:
    """LLM-based post-processor for ASR text using Qwen3 with /no_think mode"""

    # Available post-processing models
    AVAILABLE_MODELS = [
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Qwen3-4B-4bit",
        "mlx-community/Qwen3-1.7B-4bit",
    ]

    DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"

    # System prompt template for post-processing
    SYSTEM_PROMPT = """/no_think
You are an assistant that cleans up speech recognition results while preserving the speaker's intended meaning.

## Your Task
Remove verbal noise while keeping the speaker's message intact:

1. **Remove filler words** - These add no meaning:
   - English: um, uh, like, you know, well, so, I mean, kind of, sort of, basically, actually, literally, right?, anyway
   - Japanese: ええと, えーと, あの, まあ, なんか, その, うーん, ちょっと, やっぱ

2. **Handle self-corrections** - When someone corrects themselves mid-sentence, keep only their final intent:
   - "I'll be there at 3, no 4 o'clock" → "I'll be there at 4 o'clock"
   - "Send it to Tom, I mean Jerry" → "Send it to Jerry"
   - "The meeting is on Monday, wait, Tuesday" → "The meeting is on Tuesday"
   - "AですあやっぱりBです" → "Bです"
   - "3時に、いや4時に行きます" → "4時に行きます"

3. **Apply user dictionary** - Replace terms as specified

4. **Format for target app** (if specified):
   - Email: Use polite business language
   - Notion/Markdown: Format lists as Markdown

## Output
Output ONLY the cleaned text. No explanations."""

    # Summarization constants
    CONTEXT_LIMIT = 32_768
    MAX_OUTPUT_TOKENS = 8_192
    CHUNK_INPUT_BUDGET = 20_000  # 32K - 8K output - overhead margin

    # System prompt for summarization (uses thinking mode for quality)
    SUMMARIZE_PROMPT = """You are an assistant that creates concise summaries of speech transcriptions.

## Input
You will receive a chronological list of speech transcription segments with timestamps.

## Your Task
1. Identify the main topics and key points discussed
2. Create a well-organized summary that captures the essential information
3. Group related topics together
4. Preserve important details: names, numbers, dates, decisions, action items
5. Output the summary in the same language as the input transcriptions

## Output Format
Write a clear, structured summary. Use bullet points for distinct topics.
Do NOT include timestamps in the summary unless they are semantically important (e.g., "meeting at 3pm")."""

    # Prompt for merging partial summaries in multi-stage summarization
    MERGE_PROMPT = """You are an assistant that combines multiple partial summaries into a single coherent summary.

## Input
You will receive several partial summaries from different segments of a longer conversation or speech session. Each partial summary covers a contiguous time range.

## Your Task
1. Read all partial summaries carefully
2. Identify overlapping themes and connect related topics across parts
3. Produce ONE unified summary that covers all parts without redundancy
4. Preserve all important details: names, numbers, dates, decisions, action items
5. Output the summary in the same language as the input summaries

## Output Format
Write a clear, structured summary. Use bullet points for distinct topics.
Do NOT refer to "Part 1", "Part 2", etc. - the output should read as a single unified document."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._loading = False
        self._load_error: Optional[str] = None

    def get_status(self) -> dict:
        """Get current post-processor status"""
        return {
            "model_name": self.model_name,
            "loaded": self._loaded,
            "loading": self._loading,
            "error": self._load_error,
            "available_models": self.AVAILABLE_MODELS,
        }

    def set_model(self, model_name: str) -> dict:
        """Set a new post-processor model (requires reload)"""
        if model_name not in self.AVAILABLE_MODELS:
            return {
                "success": False,
                "error": f"Unknown model: {model_name}. Available: {self.AVAILABLE_MODELS}"
            }

        # Unload current model if loaded
        if self._loaded:
            self.unload_model()

        self.model_name = model_name
        logger.info(f"Post-processor model set to: {model_name}")

        return {"success": True, "model_name": model_name}

    def is_model_cached(self, model_name: Optional[str] = None) -> dict:
        """Check if the post-processor model is already cached locally."""
        target_model = model_name or self.model_name
        try:
            from huggingface_hub import try_to_load_from_cache

            result = try_to_load_from_cache(target_model, "config.json")
            is_cached = result is not None

            logger.info(f"PostProcessor model cache check: {target_model} -> cached={is_cached}")
            return {"cached": is_cached, "model_name": target_model}
        except Exception as e:
            logger.warning(f"Failed to check cache for {target_model}: {e}")
            return {"cached": False, "model_name": target_model}

    def load_model(self) -> dict:
        """Load the post-processor LLM model (Qwen3)"""
        if self._loaded:
            return {"success": True, "model_name": self.model_name, "already_loaded": True}

        if self._loading:
            return {"success": False, "error": "Model is already loading"}

        self._loading = True
        self._load_error = None

        try:
            logger.info(f"Loading post-processor model: {self.model_name}")

            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_name)
            self._loaded = True
            self._loading = False

            logger.info(f"Post-processor model loaded: {self.model_name}")
            return {"success": True, "model_name": self.model_name}

        except ImportError as e:
            self._loading = False
            self._load_error = f"mlx-lm is not installed: {e}"
            logger.error(self._load_error)
            return {"success": False, "error": self._load_error}

        except Exception as e:
            self._loading = False
            self._load_error = f"Failed to load post-processor model: {e}"
            logger.error(self._load_error)
            return {"success": False, "error": self._load_error}

    def unload_model(self) -> dict:
        """Unload the post-processor model to free memory"""
        if not self._loaded:
            return {"success": True, "already_unloaded": True}

        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info("Post-processor model unloaded")
        return {"success": True}

    def process(
        self,
        text: str,
        app_name: Optional[str] = None,
        app_bundle_id: Optional[str] = None,
        dictionary: Optional[dict] = None,
        custom_prompt: Optional[str] = None,
    ) -> dict:
        """
        Post-process ASR text using the LLM.

        Args:
            text: Raw ASR transcription text
            app_name: Name of the target application (e.g., "Mail", "Notion")
            app_bundle_id: Bundle ID of the target application
            dictionary: User dictionary for word replacements {from: to}
            custom_prompt: Custom system prompt (uses default if None)

        Returns:
            dict with success, processed_text, and processing_time_ms
        """
        if not self._loaded:
            return {"success": False, "error": "Post-processor model not loaded"}

        if not text or not text.strip():
            return {"success": True, "processed_text": "", "processing_time_ms": 0}

        try:
            import time
            from mlx_lm import generate

            start_time = time.time()

            # Build user message with context
            user_content = self._build_user_message(text, app_name, app_bundle_id, dictionary)

            # Use custom prompt if provided, otherwise use default
            system_prompt = custom_prompt if custom_prompt else self.SYSTEM_PROMPT

            # Build conversation for chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # Apply chat template with enable_thinking=False for faster response
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # Generate response
            response = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=len(text) + 100,  # Allow some expansion for keigo
                verbose=False,
            )

            # Clean up response
            processed_text = response.strip()

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Post-processing complete in {processing_time_ms:.0f}ms: '{text[:50]}...' -> '{processed_text[:50]}...'")

            return {
                "success": True,
                "processed_text": processed_text,
                "processing_time_ms": processing_time_ms,
            }

        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Return original text on error (fallback)
            return {
                "success": False,
                "processed_text": text,
                "error": str(e),
            }

    # --- Summarization ---

    def _count_prompt_tokens(self, messages: list, enable_thinking: bool = True) -> int:
        """Count tokens for a fully-rendered prompt including chat template overhead."""
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return len(self._tokenizer.encode(rendered))

    def _summarize_chunk(self, entries: list, system_prompt: str,
                         language_hint: Optional[str],
                         chunk_index: int, total_chunks: int) -> str:
        """Run a single summarization pass on a subset of entries. Returns summary text."""
        from mlx_lm import generate

        lines = [f"[{e.get('created_at', '')}] {e.get('text', '')}" for e in entries]
        user_content = "\n".join(lines)
        if language_hint:
            user_content += f"\n\n(Primary language: {language_hint})"
        if total_chunks > 1:
            user_content += f"\n\n(Part {chunk_index + 1} of {total_chunks})"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

        response = generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=self.MAX_OUTPUT_TOKENS, verbose=False,
        )

        summary = response.strip()
        if "</think>" in summary:
            summary = summary.split("</think>", 1)[-1].strip()
        return summary

    def _split_into_chunks(self, entries: list, system_prompt: str,
                           language_hint: Optional[str]) -> list:
        """Split entries into chunks that each fit within CHUNK_INPUT_BUDGET tokens."""
        chunks = []
        current_chunk = []

        for entry in entries:
            candidate = current_chunk + [entry]
            lines = [f"[{e.get('created_at', '')}] {e.get('text', '')}" for e in candidate]
            user_content = "\n".join(lines)
            if language_hint:
                user_content += f"\n\n(Primary language: {language_hint})"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            token_count = self._count_prompt_tokens(messages, enable_thinking=True)

            if token_count > self.CHUNK_INPUT_BUDGET and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [entry]
            else:
                current_chunk = candidate

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _merge_summaries(self, partial_summaries: list, language_hint: Optional[str]) -> str:
        """Merge multiple partial summaries into a single unified summary."""
        from mlx_lm import generate

        parts_text = "\n\n".join(
            f"--- Part {i + 1} ---\n{s}" for i, s in enumerate(partial_summaries)
        )
        user_content = parts_text
        if language_hint:
            user_content += f"\n\n(Primary language: {language_hint})"

        messages = [
            {"role": "system", "content": self.MERGE_PROMPT},
            {"role": "user", "content": user_content},
        ]
        token_count = self._count_prompt_tokens(messages, enable_thinking=True)

        if (token_count + self.MAX_OUTPUT_TOKENS) > self.CONTEXT_LIMIT:
            # Recursive case: merge input itself too large
            logger.warning(
                f"Merge prompt too large ({token_count} tokens). "
                f"Recursively merging {len(partial_summaries)} summaries."
            )
            mid = len(partial_summaries) // 2
            left = self._merge_summaries(partial_summaries[:mid], language_hint)
            right = self._merge_summaries(partial_summaries[mid:], language_hint)
            return self._merge_summaries([left, right], language_hint)

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

        response = generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=self.MAX_OUTPUT_TOKENS, verbose=False,
        )

        summary = response.strip()
        if "</think>" in summary:
            summary = summary.split("</think>", 1)[-1].strip()
        return summary

    def summarize(self, texts: list, language_hint: Optional[str] = None, custom_prompt: Optional[str] = None) -> dict:
        """
        Summarize a list of transcription entries.
        Automatically splits into multiple chunks and merges if input exceeds context window.

        Args:
            texts: List of dicts with 'text' and 'created_at' keys
            language_hint: Dominant language to guide output language
            custom_prompt: Custom system prompt (uses default if None)

        Returns:
            dict with success, summary, and processing_time_ms
        """
        if not self._loaded:
            return {"success": False, "error": "Post-processor model not loaded"}

        if not texts:
            return {"success": True, "summary": "", "processing_time_ms": 0}

        try:
            import time

            start_time = time.time()
            system_prompt = custom_prompt if custom_prompt else self.SUMMARIZE_PROMPT

            # Count tokens for the full prompt
            lines = [f"[{e.get('created_at', '')}] {e.get('text', '')}" for e in texts]
            user_content = "\n".join(lines)
            if language_hint:
                user_content += f"\n\n(Primary language: {language_hint})"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            total_tokens = self._count_prompt_tokens(messages, enable_thinking=True)
            fits_in_context = (total_tokens + self.MAX_OUTPUT_TOKENS) <= self.CONTEXT_LIMIT

            if fits_in_context:
                # Single-pass summarization
                logger.info(f"Summarizing {len(texts)} entries in single pass ({total_tokens} input tokens)")
                summary = self._summarize_chunk(texts, system_prompt, language_hint, 0, 1)
            else:
                # Multi-stage: chunk → partial summaries → merge
                chunks = self._split_into_chunks(texts, system_prompt, language_hint)
                logger.info(
                    f"Input too large ({total_tokens} tokens). "
                    f"Splitting {len(texts)} entries into {len(chunks)} chunks."
                )
                partial_summaries = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Summarizing chunk {i + 1}/{len(chunks)} ({len(chunk)} entries)")
                    partial = self._summarize_chunk(chunk, system_prompt, language_hint, i, len(chunks))
                    partial_summaries.append(partial)

                logger.info(f"Merging {len(partial_summaries)} partial summaries")
                summary = self._merge_summaries(partial_summaries, language_hint)

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Summarization complete in {processing_time_ms:.0f}ms: {len(texts)} entries -> {len(summary)} chars")

            return {
                "success": True,
                "summary": summary,
                "processing_time_ms": processing_time_ms,
            }

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "summary": "",
                "error": str(e),
            }

    # --- Helpers ---

    def _build_user_message(
        self,
        text: str,
        app_name: Optional[str],
        app_bundle_id: Optional[str],
        dictionary: Optional[dict],
    ) -> str:
        """Build the user message with context for the LLM"""
        parts = [f"Speech recognition text: {text}"]

        if app_name or app_bundle_id:
            app_info = app_name or ""
            if app_bundle_id:
                app_info += f" ({app_bundle_id})" if app_info else app_bundle_id
            parts.append(f"Target app: {app_info}")

        if dictionary:
            dict_str = ", ".join([f'"{k}"→"{v}"' for k, v in dictionary.items()])
            parts.append(f"User dictionary: {dict_str}")

        return "\n".join(parts)
