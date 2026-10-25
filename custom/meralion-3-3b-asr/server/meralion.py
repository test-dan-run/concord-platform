"""MERaLiON-3-3B-ASR on transformers (the model's own remote code), following the `meralion-3-asr` 0.1.0 package's
transformers backend: its fixed prompt, greedy decoding with a 6-gram repeat guard, 30 s non-overlapping chunks,
and removal of the speaker tag the model sometimes emits. The package itself is not installed: it hard-pins
vllm==0.16.0.

Attention is "eager": the Gemma-2 decoder's logit softcapping is not applied under sdpa (the package's note).

`transcribe_batch` runs many chunks in one `generate` call. The processor pads every chunk to the Whisper encoder's
30 s window and gives each the same prompt with the same number of speech tokens, so a batch needs no extra padding
and greedy decoding gives each chunk the text it gets alone (up to bf16 rounding in batched kernels).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

SR = 16000
MAX_CHUNK_S = 30.0  # the model is trained on <= 30 s windows
MIN_TAIL_S = 0.1  # a shorter last chunk is dropped: no speech, yet a whole padded 30 s window
CHUNK_MS = int(MAX_CHUNK_S * 1000)  # the batch length of every chunk: the encoder always sees a padded 30 s window
# from meralion_3_asr/prompts.py (0.1.0)
ASR_CONTENT = ("Instruction: Please transcribe this speech. \n"
               "Follow the text instruction based on the following audio: <SpeechHere>")
GENERATION = {"max_new_tokens": 512, "do_sample": False, "no_repeat_ngram_size": 6}
SPEAKER_TAGS = ("<Speaker1>:", "<Speaker1>", "Speaker 1:")


def decode_audio(data: bytes) -> np.ndarray:
    """Any libsndfile format -> mono float32 at 16 kHz. Raises ValueError on undecodable input."""
    import soundfile as sf

    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - libsndfile raises several types
        raise ValueError(f"could not decode audio file: {exc}") from None
    x = x.mean(axis=1)
    if len(x) == 0:
        raise ValueError("audio file has no samples")
    if sr != SR:
        import soxr

        x = soxr.resample(x, sr, SR).astype(np.float32)
    return x.astype(np.float32)


def chunks(x: np.ndarray, max_s: float = MAX_CHUNK_S) -> list[np.ndarray]:
    n = int(max_s * SR)
    if len(x) <= n:
        return [x]
    out = [x[s:s + n] for s in range(0, len(x), n)]
    return out if len(out[-1]) >= int(MIN_TAIL_S * SR) else out[:-1]


def strip_speaker_tag(text: str) -> str:
    text = text.strip()
    for tag in SPEAKER_TAGS:
        if text.startswith(tag):
            text = text[len(tag):].lstrip()
    return text


def join(texts) -> str:
    """The chunks of one file joined with spaces, as the package does."""
    return " ".join(t for t in texts if t)


class Meralion3Asr:
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.model = None
        self.processor = None
        self.prompt = ""
        self.device = "cpu"
        self.dtype = None
        self._torch = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if not (self.weights_dir / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"weights missing in {self.weights_dir}; run `just fetch-weights meralion-3-3b-asr`")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            from aisvc.gpu import limit_torch_memory

            limit_torch_memory(torch)
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._torch = torch
        path = str(self.weights_dir)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                                               dtype=self.dtype, attn_implementation="eager")
        self.model = self.model.to(self.device).eval()
        # the model's own chat template adds <bos> and the turn markers (Gemma-2 degenerates without <bos>)
        self.prompt = self.processor.tokenizer.apply_chat_template([{"role": "user", "content": ASR_CONTENT}],
                                                                   tokenize=False, add_generation_prompt=True)

    def transcribe_batch(self, pieces: list[np.ndarray]) -> list[str]:
        """Text of each chunk (at most 30 s) from one `generate` call. Callers serialise calls (aisvc MicroBatcher)."""
        if self.model is None:
            raise RuntimeError("model not loaded")
        torch = self._torch
        inputs = self.processor(text=[self.prompt] * len(pieces), audios=list(pieces), return_tensors="pt",
                                padding=True).to(self.device)
        for k, v in inputs.items():
            if v.dtype.is_floating_point:
                inputs[k] = v.to(self.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **GENERATION)
        new = out[:, inputs["input_ids"].shape[1]:]  # rows that stop early are filled with pad, which decoding skips
        return [strip_speaker_tag(t) for t in self.processor.batch_decode(new, skip_special_tokens=True)]

    def transcribe(self, data: bytes) -> tuple[str, float]:
        """One file, chunk by chunk (the app batches the chunks of concurrent requests instead)."""
        x = decode_audio(data)
        return join(self.transcribe_batch([c])[0] for c in chunks(x)), round(len(x) / SR, 3)
