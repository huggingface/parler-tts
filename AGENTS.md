# AGENTS.md

Notes for AI coding agents (and humans) running Parler-TTS inference, verified end-to-end on a
real **Tesla T4 (16GB, Turing, sm_75)**. Nothing here changes any default behavior — it only
documents things the README/INFERENCE.md don't say, discovered by actually running the code.

## Install: pin `torch` and `torchaudio` explicitly — `pip install git+https://github.com/huggingface/parler-tts.git` alone can resolve incompatible versions today

`setup.py` declares `"torch"` with no version bound, and its `descript-audiotools` dependency
similarly declares an unpinned `torchaudio`. On 2026-07-08, a plain
`pip install git+https://github.com/huggingface/parler-tts.git` into a fresh venv resolved
**torch 2.12.1**, which then fails on any GPU whose driver predates its CUDA build:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12040). Please update
your GPU driver by downloading and installing a new version...
```

(Driver here was 550.163.01 / CUDA 12.4 — plenty modern, just older than what torch 2.12.1 was
built against.) After installing `torch==2.5.1+cu124` first, re-running the same install command
pulled in **torchaudio 2.11.0** (unpinned, latest available) as a side effect of the
`descript-audiotools` dependency chain, which then fails at import time because it needs a newer
CUDA runtime than the pinned torch build ships:

```
OSError: libcudart.so.13: cannot open shared object file: No such file or directory
```

Fix that worked (verified end-to-end): install torch and torchaudio from the **same** pinned
build *before* installing parler-tts, matching your driver's CUDA support:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/parler-tts.git
```

This is a moving-target problem (today's "latest" torch/torchaudio, not a fixed bad release), so
the actionable takeaway for an agent is: **do not trust an unpinned `pip install` of this repo to
resolve a CUDA-compatible torch+torchaudio pair — pin both explicitly to versions matching your
driver first.**

## Attention backend: do NOT pass `attn_implementation="sdpa"` explicitly — it crashes, even though SDPA *is* the silent default

INFERENCE.md's own "Efficient Attention implementations" example says *"SDPA is used by default"*
and then shows how to *"switch between attention implementations"* by passing
`attn_implementation=attn_implementation` to `from_pretrained(...)`, with `"sdpa"` listed as one
of the three valid values (alongside `"eager"` and `"flash_attention_2"`). This reads as if
explicitly passing `"sdpa"` should reproduce the default — **it does not**. It crashes:

```python
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    attn_implementation="sdpa",
).to("cuda:0", dtype=torch.bfloat16)
```
```
ValueError: T5EncoderModel does not support an attention implementation through
torch.nn.functional.scaled_dot_product_attention yet. ... load your model with the argument
`attn_implementation="eager"` meanwhile.
```

Root cause (confirmed by inspecting the loaded model, not just the traceback): Parler-TTS is a
composite model (`text_encoder` = T5, `decoder` = its own architecture, `audio_encoder` = DAC).
When `attn_implementation` is **omitted** (the actual README quickstart), each submodule resolves
its own default independently — `decoder` gets `sdpa`, but `text_encoder` (T5, which doesn't
support SDPA in `transformers==4.46.1`, the version this repo pins) and `audio_encoder` silently
fall back to `eager`:

```python
>>> model.config._attn_implementation        # top-level / decoder
'sdpa'
>>> model.text_encoder.config._attn_implementation
'eager'
>>> model.audio_encoder.config._attn_implementation
'eager'
```

But **explicitly** passing `attn_implementation="sdpa"` propagates that literal value to every
submodule uniformly, including `text_encoder`, whose `_check_and_enable_sdpa` validation rejects
it outright. `attn_implementation="flash_attention_2"` fails too, but for an unrelated reason —
the `flash_attn` package simply isn't installed (`ImportError`, not attempted further here since
flash-attention-2 requires sm80+ and this box is sm_75 anyway, consistent with this same T4's
general attention-backend behavior established elsewhere in this verification pass).
`attn_implementation="eager"` (the literal value actually assigned in INFERENCE.md's runnable
code sample, as opposed to the two commented-out alternatives) works correctly end-to-end.

This is not new or T4-specific — two independent users hit the identical crash on other platforms
(macOS/mps): [huggingface/parler-tts#191](https://github.com/huggingface/parler-tts/issues/191)
and [#168](https://github.com/huggingface/parler-tts/issues/168), both still open with no
maintainer response as of this writing. **Takeaway for agents: leave `attn_implementation`
unspecified (the working default), or set it explicitly to `"eager"`. Do not pass `"sdpa"` or
`"flash_attention_2"` explicitly — both fail on this transformers pin, regardless of GPU.**

## dtype: INFERENCE.md's `torch.bfloat16` recommendation is a real slowdown on T4 — the implicit fp32 default is faster

Every example in INFERENCE.md (attention-backend switching, `torch.compile`, streaming, batch
generation) casts the model with `torch_dtype = torch.bfloat16`. The model checkpoint's own
`config.json` (both `parler-tts-mini-v1` and `parler-tts-large-v1`) declares
`"torch_dtype": "float32"` — i.e. the README's basic quickstart snippet, which passes **no**
`dtype` at all, already matches the checkpoint's native precision.

On this T4 (Turing, no bf16 Tensor Core path), forcing `torch.bfloat16` per INFERENCE.md's advice
measurably regresses generation time versus doing nothing:

| dtype | effective dtype (confirmed via `next(model.parameters()).dtype`) | generation time (2 runs) | peak allocated |
|---|---|---|---|
| unspecified (repo default) | `torch.float32` | 3.86s / 3.76s | ~3.65 GB |
| `torch.bfloat16` (INFERENCE.md's recommendation) | `torch.bfloat16` | 6.10s / 5.51s | ~1.90 GB |
| `torch.float16` | `torch.float16` | 4.39s / 4.51s | ~1.91 GB |

Unlike some bf16-*trained* models, forcing fp16 here does **not** crash or produce NaN/Inf (this
checkpoint was trained/saved in fp32, not bf16, so there's no bf16→fp16 range-overflow risk on
cast) — but it's still slower than simply leaving `dtype` unset. **Takeaway for agents: on a T4,
do not add `torch_dtype=torch.bfloat16` (or fp16) "to optimize for the GPU" — the repo's implicit
fp32 default is both the fastest and the numerically-native option on this hardware generation.**
(Also note: casting to bf16/fp16 and then calling `.cpu().numpy()`, as the README's save-to-disk
snippet does, raises `TypeError: Got unsupported ScalarType BFloat16` — numpy has no bf16 dtype.
Anyone who *does* force bf16/fp16 needs `.float().cpu().numpy()` instead.)

## Quantization / CUDA Graphs — not present anywhere in this repo

- `grep -rn "bitsandbytes\|bnb\|int8" --include="*.py" .` → **zero matches** in the entire repo
  (inference and training code alike). No int8 path exists at all.
- `grep -rn "torch\.cuda\.graph\|CUDAGraph\|make_graphed_callables" --include="*.py" .` → **zero
  matches**. CUDA Graphs are not used anywhere.

## Reproducibility note

The published `generation_config.json` for `parler-tts-mini-v1` sets `"do_sample": true` by
default, and the README's main quickstart snippets (random voice / specific speaker) call
`model.generate(...)` with no `do_sample` override and no seed. Running the exact same documented
command repeatedly therefore produces **different output lengths every time** (observed output
sizes of 179244, 193580, 142380, 186412 bytes across four identical runs of the literal README
snippet). This is expected sampling behavior, not a bug — but agents scripting benchmarks or
regression tests need `transformers.set_seed(...)` (as shown only in README's separate "batch
generation" example, not in the primary quickstart) to get reproducible output length/content.

## Verified working environment (Tesla T4)

- Python 3.10.12, `torch==2.5.1+cu124`, `torchaudio==2.5.1+cu124`, `transformers==4.46.1` (the
  exact version `setup.py` pins: `transformers>=4.46.1,<=4.46.1`).
- Peak VRAM for one `parler-tts-mini-v1` inference call (fp32 default): **~3.7-3.9 GB**
  (`nvidia-smi --query-gpu=memory.used --format=csv -l 1` sampled in the background, cross-checked
  with `torch.cuda.max_memory_allocated()` ≈ 3.65 GB). Comfortably fits a 16GB T4 with plenty of
  headroom for the large (2.3B) checkpoint too.
- Full documented quickstart command (`pip install` + checkpoint download + `model.generate(...)`
  + save to wav) succeeds as-is once torch/torchaudio are pinned per above — no other changes
  needed.
