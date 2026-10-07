# Background concepts and the visualizer

This page covers running **open-source models yourself**, mostly on a GPU in Google Colab, and distils the ideas
that matter for the minutes creator.

## Google Colab and GPUs
- Colab gives a hosted Jupyter runtime. Pick the hardware via *Runtime -> Change runtime type*: CPU, **T4 GPU**
  (free/low cost, 16 GB VRAM - enough for everything in this project), or A100/L4 on paid tiers.
- Secrets live in the key icon (`userdata.get("HF_TOKEN")`), never in cells. Google Drive is mounted with
  `drive.mount(...)` to load audio or save outputs.
- `!nvidia-smi` checks the GPU; `torch.cuda.is_available()` in code. Runtimes are ephemeral: reinstall packages and
  re-download models each session (Drive caching of `HF_HOME` can help).
- Rule of thumb: VRAM needed ~ parameters x bytes per weight (8B x 2 bytes = 16 GB in fp16; ~5 GB in 4-bit).

## Hugging Face pipelines (high-level API)
`transformers.pipeline(task, ...)` bundles tokenizer + model + pre/post-processing. Tasks shown include
sentiment-analysis, NER, question-answering, summarization, translation, zero-shot-classification, text-generation,
image generation (diffusers), and text-to-speech/audio. For this project the relevant one is
`pipeline("automatic-speech-recognition", model="openai/whisper-medium", dtype=torch.float16, device="cuda",
return_timestamps=True)`; `return_timestamps=True` enables long-form (>30 s) transcription by chunking.
Needs a Hugging Face token (`HF_TOKEN`) for gated models.

## Tokenizers
- A tokenizer maps text <-> integer token ids (`AutoTokenizer.from_pretrained(name)`; `.encode`, `.decode`,
  `.batch_decode`, `.vocab`, `.get_added_vocab`). Different model families have different vocabularies and special
  tokens (Llama 3.1, Phi-3, Qwen2, StarCoder2 were compared).
- **Chat templates**: `tokenizer.apply_chat_template(messages, ...)` turns `[{"role": ..., "content": ...}]` into
  the exact special-token format each instruct model was trained with. Using the wrong template degrades output.
  The minutes creator relies on it for Llama 3.1; also set `tokenizer.pad_token = tokenizer.eos_token`.
- Token counts drive cost and context limits; long meeting transcripts can exceed a model's context window.

## Models (the "heart" of transformers)
- `AutoModelForCausalLM.from_pretrained(name, device_map="auto", quantization_config=...)` loads weights and spreads
  them over available devices. `model.get_memory_footprint()` shows size; printing the model shows the
  architecture (embedding, N decoder blocks of attention + MLP, LM head).
- **Quantisation** with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")` shrinks an 8B model from ~16 GB to ~5.5 GB
  with modest quality loss. It is what lets Llama 3.1 8B fit a T4 next to Whisper. bitsandbytes needs CUDA.
- Generation: `model.generate(inputs, max_new_tokens=..., streamer=TextStreamer(tokenizer))` prints tokens as they
  are produced; the Gradio app uses `TextIteratorStreamer` on a background thread so the UI can stream.

## The visualizer (`visualizer.py`)
`meeting_minutes/visualizer.py` provides `TokenPredictor`: it requests an OpenAI
completion with `logprobs=True, top_logprobs=3`, builds a networkx graph of each chosen token with its probability and
two alternatives, and draws it with matplotlib. It illustrates that an LLM is a next-token probability machine;
temperature/seed choose among those probabilities. It is optional and unrelated to the minutes flow.

## Pipeline mapping
audio -> **ASR pipeline** -> transcript -> **chat template + tokenizer** -> **4-bit model** ->
streamed markdown minutes, all runnable on a **Colab T4**.
