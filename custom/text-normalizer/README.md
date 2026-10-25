# text-normalizer

A **utility** (no model) showing the utility pattern. Contract **openai-inspired**, API type
**`text.transform`**: text in, text out, and no OpenAI route fits.

Rules (`server/normalize.py`):

1. Unicode NFKC normalisation
2. collapse runs of whitespace, trim
3. capitalise the first letter of each sentence, in any script that has capitals, using its titlecase form (`élan` becomes `Élan`, `ß` becomes `Ss`)
4. add terminal punctuation when missing: `?` when the last sentence opens with a question word
   or auxiliary (`how`, `what`, `are`, `can`, ...), otherwise `.`

Options (`options` object; unknown keys are a 400):

| option | default | |
|---|---|---|
| `terminal_punctuation` | `true` | add `.`/`?` when the text ends without punctuation |
| `capitalize` | `true` | capitalise sentence starts |

```bash
curl -s localhost:8501/v1/text/transform -H 'content-type: application/json' \
  -d '{"model": "text-normalizer", "input": ["hello   world.  how are you"]}'
# {"object":"list","model":"text-normalizer","created":...,
#  "data":[{"object":"text.transform","index":0,"output":"Hello world. How are you?"}], "usage": {...}}
```

```python
from aicore import client
norm = client("default-normalizer")
norm.transform("hello world how are you")       # "Hello world how are you."
norm.transform("how are you")                   # "How are you?"
```

**Replacing the internals.** This is a deliberate placeholder: it cannot insert commas or split
run-on sentences (`hello world how are you` becomes `Hello world how are you.`, not
`Hello world, how are you?`). A punctuation-restoration model
(for example a token-classification model that inserts commas and sentence breaks) can replace
`normalize()` without changing the contract: same route, request, response and options. If the
model needs a GPU, set `gpu.vram_gb` and `build.base: py-cuda` in the manifest; apps and the
conformance suite are unaffected. At that point it becomes `kind: model` with a `custom_reason`.
