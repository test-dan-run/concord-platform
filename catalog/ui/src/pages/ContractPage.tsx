// What a model service must speak and how the verifier decides what it can do. Moved here from the
// Concord apps; the platform docs remain the source of truth.

import { Link } from "react-router-dom";
import { PageHeader, Panel } from "../components/ui";

const RULES = [
  ["openai / audio.transcriptions", "verbose_json + timestamp_granularities[]=word, then verbose_json, then json", "words[] → ASR with word times · segments[] only → segment times · text only → text only · far from the clip's transcript (WER > 50 %) → warning"],
  ["openai / chat.completions with an audio task", "user message with a text instruction and an input_audio part (base64 WAV)", "plausible transcript → ASR, text only · 4xx on the audio part, or an answer unrelated to the clip → fail"],
  ["oip-v2 with AUDIO + SAMPLE_RATE in, SEGMENTS out", "infer on the clip", "SEGMENTS only → VAD · SEGMENTS + SPEAKERS → diarisation (2 speakers → pass, otherwise warning) · segments outside the clip → fail"],
  ["oip-v2 with AUDIO + SAMPLE_RATE in, AUDIO out", "infer on the clip mixed with babble noise at 5 dB SNR (for dereverberation, the clip with a synthetic room response); compare the output with the clean clip", "AUDIO out → speech enhancement · SI-SDR improves on the degraded input → pass · no improvement → warning · length off by more than 10 ms, wrong rate, NaN or clipping → fail"],
  ["openai-inspired / text.itn", "spoken-form phrases such as “twenty one” and “three thirty p m”, with return_spans", "expected digits in every output (“21”, “3:30”) → ITN · in some outputs → warning · in none, or invalid spans → fail"],
  ["anything else", "the contract's conformance checks only", "verified on its protocol; no audio capability"],
];

export function ContractPage() {
  return (
    <>
      <PageHeader crumbs="Reference" title="Model contract" subtitle="What a model service speaks, what apps send through the gateway, and how verification decides what a build can do." />
      <Panel title="Source of truth">
        <p className="m-0 text-[13px] text-muted leading-relaxed">
          Model services, their protocols and per-model config are defined by the platform: <span className="num text-fg">dev/README.md</span>,{" "}
          <span className="num text-fg">platform/contracts/README.md</span> and <span className="num text-fg">platform/services/*.yaml</span>. The catalog reads them
          and adds only registration, verification, scheduling and routing (<span className="num text-fg">platform/catalog/README.md</span>). If anything here
          disagrees with the platform docs, the platform docs win.
        </p>
      </Panel>
      <div className="grid gap-5 lg:grid-cols-2 2xl:grid-cols-3">
        <Panel title="ASR · openai / audio.transcriptions">
          <pre className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{`POST /m/<model>/v1/audio/transcriptions  (multipart)
file=<16 kHz mono WAV>
model=<service name>
response_format=verbose_json | json
timestamp_granularities[]=word
language=<e.g. en>`}</pre>
          <span className="text-xs text-muted">Returns <span className="num">text</span>, and <span className="num">words[]</span> / <span className="num">segments[]</span> when the model has them.</span>
        </Panel>
        <Panel title="Audio LLM · openai / chat.completions">
          <pre className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{`POST /m/<model>/v1/chat/completions
{"model": "<service>", "temperature": 0,
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "Transcribe …"},
   {"type": "input_audio",
    "input_audio": {"data": "<base64 WAV>", "format": "wav"}}]}]}`}</pre>
          <span className="text-xs text-muted">Text only. In pipelines the VAD or diarisation stage provides timing.</span>
        </Panel>
        <Panel title="VAD / diarisation · oip-v2 / infer">
          <pre className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{`POST /v2/models/<model>/infer
inputs:  AUDIO        FP32  [-1]    16 kHz mono (VAD)
                            [-1,-1] clips padded to one
                                    length (diarisers)
         LENGTHS      INT32 [-1]    each clip's length
         SAMPLE_RATE  INT32 [1]     16000
outputs: SEGMENTS     FP32  [-1,2]  start, end (s)
         SPEAKERS     INT32 [-1]    diarisers only
         ROWS         INT32 [-1]    each segment's clip`}</pre>
          <span className="text-xs text-muted">
            A model that returns <span className="num">SPEAKERS</span> is a diariser; without it, a VAD. Diarisers take a batch with <span className="num">LENGTHS</span> and return <span className="num">ROWS</span>; one clip as <span className="num">AUDIO [-1]</span> without <span className="num">LENGTHS</span> still works.
          </span>
        </Panel>
        <Panel title="Speech enhancement · oip-v2 / infer">
          <pre className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{`POST /v2/models/<model>/infer
inputs:  AUDIO        FP32  [-1,-1] clips padded to one
                                    length, −1…1
         LENGTHS      INT32 [-1]    each clip's length
         SAMPLE_RATE  INT32 [1]     one rate for every
                                    clip, e.g. 16000
outputs: AUDIO        FP32  [-1,-1] enhanced, same rate
                                    and lengths as input
         LENGTHS      INT32 [-1]
         SAMPLE_RATE  INT32 [1]     the input's rate`}</pre>
          <span className="text-xs text-muted">
            Audio in, audio out: denoising, dereverberation, speech separation to one target. One clip as <span className="num">AUDIO [-1]</span> without <span className="num">LENGTHS</span> still works. Tensors are sent as JSON number arrays, as the platform contract requires; the binary tensor extension is not implemented, so send files of minutes in pieces. <span className="num">deepfilternet3</span> and <span className="num">wpe-dereverb</span> take any rate from 8 kHz. <span className="num">nemotronlabs-re-use</span> takes 8, 16, 22.05, 24, 32, 44.1 or 48 kHz, and enhances clips over 15 s in 5 s chunks itself.
          </span>
        </Panel>
        <Panel title="Text normalisation (ITN) · openai-inspired / text.itn">
          <pre className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{`POST /m/<model>/v1/text/itn
{"model": "<service>",
 "input": ["meet at three thirty p m"],
 "options": {"language": "en-SG",
             "return_spans": true,
             "classes": "TIME,DATE"}}
→ data[].output  "meet at 3:30pm"
  data[].spans   input and output offsets, class`}</pre>
          <span className="text-xs text-muted">
            Spoken form in, written form out, one output per input. Options are scalar, so <span className="num">classes</span> is a comma-separated string, and unknown options are rejected. Spans are half-open offsets in code points. At most 256 inputs of 10,000 characters. Schema: <span className="num">platform/contracts/openapi/text.itn.yaml</span>.
          </span>
        </Panel>
      </div>
      <Panel title="How a build is verified" subtitle="The platform's conformance checks first, then a capability probe: a 10.56 s synthesised two-speaker clip with a known transcript and turns for audio models, spoken-form phrases for ITN">
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="text-[11px] text-muted uppercase tracking-wider text-left">
                <th scope="col" className="font-medium px-3 py-2 border-b border-line">Service</th>
                <th scope="col" className="font-medium px-3 py-2 border-b border-line">Probe</th>
                <th scope="col" className="font-medium px-3 py-2 border-b border-line">Result</th>
              </tr>
            </thead>
            <tbody>
              {RULES.map(([e, p, r]) => (
                <tr key={e} className="border-b border-track align-top">
                  <td className="px-3 py-2 num text-xs">{e}</td>
                  <td className="px-3 py-2 text-xs text-muted">{p}</td>
                  <td className="px-3 py-2 text-xs">{r}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <span className="text-xs text-muted">
          Every record belongs to one build: image id + config hash + weights hash, or contract + api + base URL + model name for a hosted model. A rebuilt
          image, new weights or a change to a manifest field in the config hash is a new build and is verified again automatically. <Link to="/add">Add a model</Link>.
        </span>
      </Panel>
    </>
  );
}
