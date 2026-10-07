"""Minutes generators (streaming).

* ``LlamaLocal`` - Meta-Llama-3.1-8B-Instruct, 4-bit NF4 via bitsandbytes (CUDA only).
* ``ChatAPI``    - OpenAI chat completions streaming (no GPU).

Both expose ``stream(transcript) -> Iterator[str]`` yielding the *accumulated* markdown.
"""

from __future__ import annotations

from threading import Thread
from typing import Iterator, Protocol

from openai import OpenAI

from meeting_minutes.config import Settings
from meeting_minutes.prompts import build_messages


class MinutesGenerator(Protocol):
    def stream(self, transcript: str, meeting_description: str = ...) -> Iterator[str]: ...


class ChatAPI:
    def __init__(self, client: OpenAI | None = None, model: str | None = None):
        self._client = client
        self.model = model or Settings().openai_chat_model

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def stream(self, transcript: str, meeting_description: str = "a Denver City council meeting") -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model, messages=build_messages(transcript, meeting_description), stream=True
        )
        acc = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                acc += delta
                yield acc


class LlamaLocal:
    def __init__(self, model: str | None = None, max_new_tokens: int | None = None):
        s = Settings()
        self.model_name = model or s.llama_model
        self.max_new_tokens = max_new_tokens or s.max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Local backend needs: pip install -e '.[local]'") from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit bitsandbytes quantisation needs a CUDA GPU (Colab T4 is enough). "
                "Use --backend api for the CPU-friendly fallback."
            )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=quant_config
        )

    def stream(self, transcript: str, meeting_description: str = "a Denver City council meeting") -> Iterator[str]:
        from transformers import TextIteratorStreamer  # noqa: PLC0415

        if self._model is None:
            self._load()
        inputs = self._tokenizer.apply_chat_template(
            build_messages(transcript, meeting_description), return_tensors="pt", add_generation_prompt=True,
            return_dict=True,
        ).to("cuda")
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(
            target=self._model.generate,
            kwargs={**inputs, "max_new_tokens": self.max_new_tokens, "streamer": streamer},
        )
        thread.start()
        acc = ""
        for piece in streamer:
            acc += piece
            yield acc
        thread.join()
