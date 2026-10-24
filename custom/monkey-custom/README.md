# monkey-custom

ASR transcript normaliser: [transcript-normer](https://github.com/test-dan-run/transcript-normer) at commit
e98d4ae, served as an `openai-inspired` `text.transform` model service on CPU. It removes verbal fillers (`um`,
`you know`, `I mean`…) and Singlish discourse particles (`lah`, `lor`, `leh`…), stitches spelled-out acronyms
(`U S A`, `U.S.A.` → `USA`), and never resolves self-corrections (`five no wait ten` stays).

```python
from aicore import client

c = client("monkey-custom")
c.transform("I went to the U S A last year lah spent twenty dollars you know")
# 'i went to the usa last year spent $20'
c.transform(["um okay lah see you tomorrow how are you doing"], mode="punctuated")
# ['Okay, see you tomorrow, how are you doing?']
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `canonical` \| `verbatim` \| `punctuated` | `canonical` | `canonical`: whisper-normalizer (lowercase, digits and currency, no punctuation), for WER and search. `verbatim`: casing and number words kept. `punctuated`: verbatim plus restored punctuation and casing. |
| `remove_fillers` | bool | `true` | Remove fillers |
| `remove_particles` | bool | `true` | Remove Singlish particles |

Unknown options are a 400 `invalid_options`. At most 1,000 inputs of 100,000 characters each (400 beyond).

- **Weights:** `just fetch-weights monkey-custom` (210 MB punctuation model, sha256-pinned) into
  `${AI_ASSETS}/custom-models/monkey-custom/b26fd1c`. Without it the service never becomes ready.
- **Updating:** re-copy `app/services/{normalizer,punctuator}.py` from the project at a new commit into
  `server/normer/` (keep the relative import), and update the commit in the headers, this file and `WHY.md`.
