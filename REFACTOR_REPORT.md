# Deep refactor of the platform, 2026-10-26

This report covers the deep refactor on branch `refactor/deep-refactor`. The branch has 185 commits on top of `main` at `1cdd5e0`, this report being the last, and it is not merged into `main`. The work followed the deep-refactor workflow: audit the codebase, write the findings down with their evidence, cut them into tickets, and land each ticket as one commit while the original test suite stays unchanged.

Most of the commits fix defects that the audit proved with a failing input. The rest are behaviour-preserving refactors, docs that misstated the code, and guard tests. Every commit message ends with a `Ticket:` line that ties it to the ledger below.

## Results

| What | At `1cdd5e0` | On the branch | Command |
|---|---|---|---|
| Python tests | 645 passed, 29 skipped | 1314 passed, 29 skipped | `uv run pytest` |
| Catalog UI tests | 29 passed (5 files) | 59 passed (19 files) | `cd catalog/ui && npm test` |
| Test files | 43 | 186 (143 new) | `git ls-files` |
| Original test files changed | | 1 (`custom/utmos/tests/test_utmos.py`, ticket T1) | `git diff --diff-filter=MDR 1cdd5e0..HEAD` |
| Functions graded E or F | 12 | 2 | `uvx radon cc -s -n E` over production code |
| Functions graded D or worse | 30 | 20 | `uvx radon cc -s -n D` over production code |
| Duplicated production lines | 469 (1.76%) | 447 (1.68%) | `npx jscpd@4 --min-lines 8 --min-tokens 70`, tests excluded |
| Concord Label suite against the branch's aicore | 243 passed | 243 passed | app venv with `PYTHONPATH=<branch>/clients/aicore/src` |
| Concord Bench suite | 299 passed | 299 passed | same |
| Synthetics suite | 241 passed, 1 skipped | 241 passed, 1 skipped | same |

The app suites were run read-only after each change to aicore, and `git status` in each app was the same before and after every run. The two functions still graded E are `size_model.main` in the register-model skill (a linear print script) and `Exporter._run`.

## What happens when you merge

Nothing on this branch takes effect until you merge it. The main checkout was left on `main` the whole time, because the catalog container mounts it.

1. The aicore changes reach Concord Bench, Concord Label and Synthetics as soon as `main` has them, because the apps install aicore from the main checkout in editable mode. Their suites pass against the branch's aicore with the same counts as before.
2. The catalog runs the new code after `just catalog-up`.
3. The image fixes take effect only when you rebuild the images, and each rebuild gives the image a new id, so its verification runs again and its batch profile needs measuring again.
   - Images whose own server code changed: ark-asr-0-6b, meralion-3-3b-asr, wavlm-sv, redimnet2-sv, dnsmos, qwen3-forced-aligner, utmos, pyannote-diarization, deepfilternet3, wpe-dereverb, silero-vad, text-normalizer.
   - Every image on base 0.2.0 also gets the aisvc fixes when it is rebuilt, because the base is rebuilt once its inputs change. text-normalizer, monkey-custom, silero-vad, kokoro-tts and voxcpm2 stay on base 0.1.1, so they do not get the aisvc fixes (decision 5 below).
   - ark-asr-0-6b and meralion-3-3b-asr already needed a rebuild for `63d86dc`.
4. Batch profiles that were measured with an out-of-memory split at one length were fitted from lowered limits (W1-1), and the ark-asr-0-6b profile includes points past the server's concurrency (W1-2). Press "Measure again" on the Batching panel for those models after the catalog restarts.

## Decisions that need you

The branch leaves these alone because each one either needs an edit to an original test, changes behaviour that apps or people rely on, or is outside this repository.

| ID | Question | What the branch does now |
|---|---|---|
| CF2 | A PATCH of `/api/hosted/{id}` that moves `base_url` to another host without a key keeps the stored key, so the gateway and verification send it to the new host. Requiring the key on a host change needs an edit of `catalog/tests/test_hosted.py:163` and a change to the hosted detail page, which saves `base_url` without a key. | `POST /api/hosted/test` refuses to send the stored key to another host (B2). PATCH is unchanged. |
| CR12 | A stop sent while a model is starting answers 202 and is not kept, and `?force=1` on a model with an active lease stops it and the lease starts it again at once. Should a stop queue behind the start (or answer 409), and should a force stop end the lease? | The README describes what happens today (`8748dd5`). |
| CI1 | aisvc's `DynamicBatcher` gives every caller in a failed group the error, which `tests/test_aisvc_batch.py:175-184` pins. | ark-asr and meralion were fixed in their own code (G1, X1-1), so a bad file fails only its own request there. The aisvc behaviour is unchanged. |
| P2U4 | `just datasets-gc` deletes what clean-up finds when you answer "y", which can be more than its dry run showed. The CLI call list is pinned by `catalog/tests/test_dataset_store.py:192`. | The API takes `expect_files` and the Storage page sends it (W6-2, E2-4). The CLI is unchanged. |
| P3I5 | Five images are on base 0.1.1, which ships aisvc 0.1.0: a handler error has code null and there are no `/platform/*` routes. Moving them to 0.2.0 needs edits of `tests/test_images.py` (the BATCHED set) and `tests/test_render.py:75`, and rebuilds. kokoro-tts and voxcpm2 are marked "must not change". | contracts/README.md names the exception (X4-4). |
| W3-3 | The daily re-verification of hosted endpoints runs only when `CATALOG_VERIFY_NEW_BUILDS` is on, so turning off verification of new platform builds also turns off the provider re-checks. | Documented, not changed. |
| CR19 | `test_a_model_waits_rather_than_loading_partly` takes 30 s, because the scheduler has no GPU reading until its first tick and the start waits `gpu_free_wait_s`. The same 30 s wait happens to the first GPU request after a catalog restart. A reading on the first request would fix both. | Unchanged. |
| W6-6 | A references speaker that is not a string (a list, a number) used to publish and then break `GET .../files` with a 500. The branch now refuses it with 422, because both apps store speakers with `str()`. A third-party package with numeric speakers is refused. | Refused. Revert `39c3682` if you want numeric speakers accepted and shown as text instead. |
| CI9 | `rows.py` is copied into five images, `turns.py` into two, and the ASR and speaker-verification helpers into two each. `tests/test_images.py:103-113` checks that the copies match. Moving them into aisvc would rebuild every image. | Kept as copies. |
| CI10 | The three enhancers handle an empty row differently (two return an empty row, nemotronlabs-re-use refuses the batch), and each behaviour is pinned by its image's tests. | Unchanged. |
| P3I7 | The vendored transcript-normer in monkey-custom is quadratic for guarded fillers (5.2 s for a 100k-character input of "you know"). Its WHY.md says the copy changes only its imports. | Not changed. The fix belongs upstream (search only the 64 characters before each match). |
| Outside | `/home/user/code/dev/README.md` around line 232 still says GPU services run together when their shares add up to the limit, and the Synthetics `orchestrator/lease.py` `check_vram` docstring says "over util_limit". Both predate the one-GPU-service policy. | Not edited: both are outside this repository. |

Three choices on the branch change behaviour in small ways that you may want to check.
- A PATCH of an OIP endpoint that changes its build now answers 422 when the provider's metadata cannot be read, instead of keeping the old signature (W2-3, X2-6).
- In catalog mode, `stack.py up --holder ID --ttl N` with N over 3600 is refused before the catalog is asked, with a message that says how to keep a lease longer (W4-2).
- The batch sweep for an OpenAI transcription server sends at most 32 requests at once, because the aisvc route runs in a thread pool of 40 and larger batches only queued (W1-2).

## How the work was done

The audit ran six passes, each with read-only agents over separate areas, and every claim needed evidence: a repro that failed on the code, an instrumented count, or an exact code path. The first pass covered seven areas (catalog runtime, catalog features, datasets, platform tools and contracts, custom images, the aicore client and UI, and the overall architecture). The second pass covered what the first had only skimmed. The third reviewed the code the fixes had added and the image servers nobody had audited, and ran the real-model tests on CPU with the real weights. The fourth, fifth and sixth passes reviewed only the newest commits. Each pass found less than the one before. The workflow stops at a pass that comes back empty. The sixth pass found only that the swap counter on the Models page can still be one off in rare cases, so the README now says so (`806a930`) instead of adding another field to make it exact.

The test suite at `1cdd5e0` was treated as the contract. No lane could edit or delete an original test, so each defect got a locking test in a new file, written first and seen to fail on the unfixed code. The one exception is ticket T1, which fixed the utmos real-model test: it compared a tone with a noisy tone, which UTMOS scores the same, and it had never run because the dev environment has no torch.

The fixes ran in lanes, each on its own branch and worktree under `/home/user/code/dev/worktrees/`, so parallel work could not mix. Each lane's commits were reviewed and cherry-picked onto `refactor/deep-refactor` one lane at a time, with the full Python suite (and the UI suite when the UI changed) green after each. Nothing touched `main`, the main checkout, Docker, the running catalog on :8700 or the apps' files.

One full run of the suite had one failing test while the three app suites were running on the same machine. It did not happen again in 9 further full runs, 6 runs of the catalog tests under load, or 8 runs of the timing-sensitive tests with every CPU busy, so the failing test is not known.

## Architecture

The architecture is sound, and the debt was in rules written twice and in a few unlocked check-then-act paths. aicore and aisvc are leaf libraries, `tools/` is the base layer, and `catalog/` sits on top of it. Inside the catalog the modules form a directed graph around the scheduler. The Dataset Catalog is isolated from the model catalog, so one process for every catalog role is the right size for a personal platform.

The branch changed the structure in three places.
- The dataset version format had three owners that disagreed. Its rules now live in one standard-library module, `aicore/dataset_format.py`, which the catalog and `FakeDatasetCatalog` both use (C5), so the fake refuses what the service refuses.
- Nine places in `batching.py`, `verifier/` and `app.py` reached into the scheduler's lock and private state. They now go through `Scheduler.hold` and three accessors (R1-4).
- Container inspection, the GPU fit rule and the host weights paths each had two or more copies that had drifted. Each now has one owner: `stack.inspect_project` (R2-1), `vram_check.start_env` for the fit (A7) and `Platform.model_dir` (R2-2).

Structural changes considered and rejected:
- Splitting the catalog into several services.
- Dropping the direct lease mode, which Synthetics and 25 tests rely on.
- Running the tools as `python -m`, which would break the Justfile, the README, the skill and Synthetics.
- Packaging `contracts/`.
- Generating the README tables. A test that checks them is cheaper (F12).

## Defects fixed, by area

The appendix lists every commit with its ticket and audit finding. These are the defects with the most effect.

Security:
- A hosted provider's API key that the provider echoed back was stored in the verification report (B1) and sent to apps in gateway error replies (W2-1). Both are redacted now.
- `POST /api/hosted/test` sent the stored key to any URL in the request (B2).
- The importer trusted bundle model ids and weights paths as file paths (B8).

Scheduler and leases:
- The tick could apply a container listing that went out of date before it took the lock, which wiped what a start was given, started a model twice or marked a stopped model running (A6).
- A lease could be granted while its model was being stopped (A4). An unexpected error in a start left the model "starting" forever and blocked every GPU lease (A1). The GPU fit and the start's own rule disagreed by a few MB, so a request could be granted and refused forever (A7).
- Pins from expired starts survived (A5, X2-3), a failed stop pinned a model for good (A3), and an eviction whose stop kept failing could run two GPU models (X2-2).

Catalog API and features:
- Aliases gave 500s in plans, leases and the batch routes (H1, A9, H2).
- Two exports could run at once and delete each other's bundles (B3), and a setup error wedged every later export (B4).
- The event stream dropped a slow client without ending its stream, so the UI stayed "live" with no events (H4).
- The batch profiler fitted profiles from lowered limits after an out-of-memory split (W1-1) and counted queued requests as batches (W1-2).

Datasets:
- Malformed packages gave 500s, and an import could leave an empty dataset behind (C4).
- The fake accepted what the service refused (C5).
- NaN, infinite and huge numbers broke stored versions (C8, X1-6).
- An upload held the event loop for about a second while another request held the database lock (W6-1). The version comparison was quadratic (W6-3).

Tools:
- `stack.py` drove Docker itself when the catalog answered with an error, which could stop a model the catalog had leased to an app (F4).
- Generated files were written in place, so `docker compose` could read a half-written file (F9).
- The register-model skill's `platform_state.py` crashed, and its docs told agents to write a field that fails validation (F1, F2).
- `base_digest` hashed files the base build never copies (W4-1), and it now refuses COPY forms it cannot read (X3-2).

Images:
- In ark-asr and meralion, one failing file failed every concurrent request, and an out-of-memory error stopped batching across requests for good (G1, X1-1).
- Short or odd audio crashed the speaker verification images, dnsmos and the aligner (G2, G3), and utmos took 12 GB of RAM at an odd sample rate (X4-1).
- Parameters of the wrong type or range answered 500 (G4, X4-2), and aisvc answered some handler errors outside the contract's error shape (G7, X1-3).

## Measured performance changes

| What | Before | After | Ticket |
|---|---|---|---|
| Scheduler tick with the fake runtime (weights.yaml parsed every tick) | 15.8 to 16.0 ms | 0.87 to 0.90 ms | R1-6 |
| uids.jsonl check, 50k files and 50k uid lines | 8.58 s | 0.29 s | C3 |
| Version comparison, one file of 3000 segments | 1.34 s | a few ms | W6-3 |
| Event loop stall while another request held the datasets database lock | 0.93 s | under 0.3 s (test bound) | W6-1 |
| Storage stats on 200k audio and 1M version-file rows | 1.05 s | 0.60 s | W6-5 |
| `FakeCatalog.index_lookup`, 5000 keys against 5000 entries | 2.66 s | 1.6 ms | D4 |
| Peak extra disk for a tarball bundle of 256 MiB of weights | 508.7 MiB | 256.0 MiB | R2-8 |
| ffprobe calls per file when a zip is unpacked | 2 | 1 | W6-4 |

Each figure comes from the command in its commit message, run at that commit.

## Rejected or deliberate

| Candidate | Why it stays |
|---|---|
| An exact swap count | The count is made when a request is granted. It can be one off when a stop keeps failing together with a model started outside the catalog or a catalog restart. Making it exact needs another field on each run, for a figure only the Models page shows. |
| The `sys.path` inserts at the top of the tools | `python tools/x.py` needs them, and the Justfile, the README, the skill and Synthetics run the tools that way. |
| `tools/schema.py` importing the catalog's dataset check | It is lazy and costs 12 ms, and it keeps `just validate` as one gate. The catalog's half of that cycle is gone (C6). |
| Two lease systems, one in files for direct mode and one in the catalog | Direct mode is the fallback when the catalog is down, and 25 tests and Synthetics use it. |
| The config hash including signature example values | Verification runs with those examples, so they belong in the build key. Changing it would re-key every OIP service. |
| One `docker images` call instead of 19 `docker image inspect` calls | Measured at 0.65 s against 0.14 s every 30 s, outside the scheduler lock. Not worth a protocol change. |
| Vendored code in nemotronlabs-re-use and monkey-custom | Changes belong upstream. |
| kokoro-tts and voxcpm2 | Their manifests say the image must not change. |
| `FakeRuntime` living in `catalog/runtime.py` | The tests import it from there. |
| `Catalog.events()` with no consumer | Public aicore API; it costs nothing. |
| Rules written twice across a package or language boundary (BASE_PATHS, the route maps, the probe for each API) | aicore ships to apps without `tools/`, and the UI is TypeScript. A guard test now checks each pair (R3-3). |

## Smaller follow-ups not done

These came up during the work, each is low in effect, and none blocks a merge.

- Scheduler: a start request that expires after it was granted still brings the model up pinned. The profiler's 30-minute limit counts time in the queue, although the README says it waits its turn. A "no build key" failure is usually not stored, because the build key is what is missing. A re-evicted model's stop runs every tick without the stop backoff.
- Gateway and hosted endpoints: `/m/{id}` without a trailing path is not routed, a hosted create writes two rows outside one transaction, and an endpoint can name itself as its parent.
- aisvc: `{"restore": "false"}` on `/platform/batch-limits` restores, a text.itn handler result that is not a pair answers 400 instead of 500, and text.transform does not check that its outputs are strings. silero-vad answers 500 for a 401-digit integer parameter.
- Tools: a misnamed copy of a manifest overwrites the first one's entry in `load_platform`, so the duplicate-name error names the same file twice. A `base_digest` refusal shows as a traceback in `just build`. `size_model.py` keeps its own copy of the model folder rule.
- Model cards: a GitHub card whose README came from a folder above the configured subpath may get wrong relative links (not checked against real data).
- UI: the VAD and diarisation panel on the Model contract page does not list the rates each service accepts.
- Docs: `DATASET_CATALOG_PLAN.md` still describes `refs.py` as the reader and does not mention `expect_files`.

## Worktrees and branches

The branch lives in `/home/user/code/dev/worktrees/platform-deep-refactor`. The lane worktrees and their `dr/*` branches were removed once `git cherry` showed every lane's commits on the branch. The deep-refactor plugin (`deep-refactor@agile-skills` 0.11.3) is installed at user scope, and its skill loads in a new session.

## Appendix: every commit

### First wave: defects from the first audit pass (81 commits)

| Commit | Ticket | Audit finding | What changed |
|---|---|---|---|
| `168673b` | E1 | CU1 | Keep uploads moving after Stop and clear |
| `4011b2a` | E2 | CU8 | Leave hosted endpoints out of Add model's platform list |
| `31706ba` | E3 | CU9 | Drop a hosted connection test that ran on older inputs |
| `e395430` | E4 | CU10 | Refresh the Batching panel when a model starts or stops |
| `a24c065` | E5 | CU13 | Remove UI code nothing uses |
| `bb44c54` | E6 | CU14 | Share the check line glyphs and the form input class |
| `7b8ab77` | H1 | CR8 = CU3 | Accept aliases in job plans and reject a malformed plan body |
| `effbca9` | H2 | CR7 | Resolve aliases in the batch routes |
| `99cf878` | H3 | CR9 | Let PATCH /api/models clear a field with null |
| `a911807` | H4 | CR11 = ARCH8 | End the event stream of a subscriber that was dropped |
| `f99e7ad` | H5 | CR14 + ARCH5 drift | Store "could not start" when a batch profile's model fails to start |
| `0f22eeb` | H6 | CR13 | Read the whole vLLM log for a KV cache profile |
| `4f9f2f4` | G1 | CI1 | Fail only the bad file when a shared ASR batch fails |
| `c2d3288` | G2 | CI2 | Keep a one-sample clip at 44.1 or 48 kHz in the SV and DNSMOS images |
| `14755b0` | G3 | CI3 | Give a forced aligner row under 0.1 s no words without the model |
| `bbcf411` | G4 | CI6 | Refuse request parameters of the wrong type with a 400 |
| `77ce232` | G5 (CI4) | CI4 + CI5 + CI7 | Accept empty INT32 and BOOL tensors in OIP requests |
| `be664e3` | G5 (CI5) | CI4 + CI5 + CI7 | Let out of memory backoff split a batch of zero-length items |
| `b9336e6` | G5 (CI7) | CI4 + CI5 + CI7 | Give a handler's ValueError the invalid_request code |
| `7bf28a5` | G6 | CI8 | Keep host Python caches out of the custom images |
| `c3d9558` | G7 | P2K5 | Answer errors in shaping a handler's result in the contract's shape |
| `71d3cce` | G8 | P2K6 | Correct the error code and the port in the text.itn OpenAPI schema |
| `14a318f` | A1 | CR3 | Fail a start that crashes instead of leaving it starting |
| `d08d767` | A2 | CR10 | Keep the weights hasher running after an error that is not an OSError |
| `c7b40c3` | A3 | CR6 | Keep a model unpinned and try again when its stop fails |
| `e021848` | A4 | CR2 | Treat a model that is being stopped as down when granting a request |
| `7b61799` | A5 | CR4 | Drop the pin of a manual start that ended before its model ran |
| `1799f4d` | A6 | CR1 | Ignore a container listing the starter made out of date during a tick |
| `458a460` | A7 | ARCH1 | Grant GPU memory by the same rule the start uses |
| `992b75d` | A8 | ARCH3, scheduler half | Reload changed manifests on one thread and let the others reuse it |
| `5a926c4` | A9 | CR5 = CU4 | Name each model once in a lease that lists it twice |
| `dab6716` | A3 (follow-up) | CR6 | Log a failed clean-up stop after a failed start as a warning |
| `3ce5953` | D1 | CU7 | Raise CatalogError for an error body that is JSON but not an object |
| `3de4bd5` | D2 | CU2 | Give the gateway's error message as OIPError.message |
| `aedfdba` | D3 | CU11 | Put an IPv6 AI_HOST in brackets in service URLs |
| `eb2e132` | D4 | CU6 | Look up FakeCatalog index keys in one pass over the index |
| `05f9e84` | D5 | CU12 | Close the catalog client that batch_size makes |
| `5204d2a` | D6 | CU5 | Make FakeCatalog answer leases, the index and plans like the server |
| `dc3a44b` | D6 (follow-up) | CU5 | Drop the expected failure from the plan alias parity test |
| `86e598e` | B1 | CF1 | Redact a hosted key that a provider echoes from verification reports |
| `ee90769` | B2 | CF2, /test only | Send a stored hosted key on /test only to the host it was stored for |
| `7e75b5a` | B3 | CF3 + ARCH2 | Run one export when two starts arrive at once |
| `5f306d8` | B4 | CF4 | End an export job that fails while it sets itself up |
| `5265836` | B5 | CF5 | Remove bundles when the export dir is a symlink or a relative path |
| `6ad384b` | B6 | CF6 | Carry SOURCE.txt through export and import |
| `e4e3df8` | B7 | CF7 | Write the live catalog DB on import only when no catalog answers |
| `7c1f58a` | B8 | CF8 | Check bundle model ids and weights paths before the importer writes |
| `ddae8ce` | B9 | CF9 | Start a download's writer thread only when the response is read |
| `4654fd1` | B10 | CF10 | Split the hosted connection test and fix its lines and its 500 |
| `266acf6` | B11 | CF11 | Keep the verification report when a capability probe crashes |
| `99df97d` | B12 | CF12 | Map DER speakers one to one when there are many of them |
| `98cb0e5` | B13 | CF13 | Resolve model card links against their folder instead of stripping dots |
| `b67d496` | B14 | CF14 | Export a model once when it is named together with its alias |
| `df8f9f5` | F1 | PT1 + PT15 | Fix the skill's platform_state.py and share the port bands with it |
| `780958f` | F2 | PT2 | Stop the register-model skill from telling agents to write profiles |
| `42fd7b7` | F3 | PT3 | Show a custom image's batch limits in just batch |
| `ed509bb` | F4 | PT4 + PT18 | Stop stack.py from driving Docker when the catalog fails its health check |
| `2539f2d` | F5 | ARCH4 = PT13 | Show the catalog's lease holders in stack.py status |
| `9c5dee4` | F6 | PT6 | Print wait_healthy progress every 30 s without a terminal |
| `5526bdf` | F7 | PT7 | Give text.itn services an .itn() usage snippet in the static catalog |
| `5b7f0e7` | F8 | PT9 | State the one GPU service policy in the static catalog legend |
| `e15bc5c` | F9 | PT10, render half of ARCH3 | Write the rendered files atomically |
| `62acd6c` | F10 | PT11 | Report a broken engines.yaml entry once, and say so in the manifests |
| `6ab555d` | F11 | PT12 | Print an error instead of a traceback for just fetch-cards nope |
| `2970297` | F12 | ARCH6 = PT8 | Fix the qwen3-8b README row and check the tables against the manifests |
| `7819206` | F13 | P2K4 | Drop the text-normalizer option from the generic transform snippet |
| `fdfdfc5` | F14 | P2K7 | Count the KV cache right for three kinds of model in size_model.py |
| `813af3b` | F15 | P2K8 | Correct manifest comments whose figures do not match the code or disk |
| `008037b` | F16 | P2K9 | Leave no partial files when fetch_weights fails, and print an error |
| `208ff13` | F17 | P2K11 | Label the static catalog's GPU figures as the minimum to load |
| `192e502` | C1 | DS4 | Strip app and ref when a use is removed, as when it is added |
| `e2afa8b` | C2 | DS5 | Refuse a sha256 with a newline after it |
| `ca49af5` | C3 | DS3 | Check uids.jsonl against a set of paths, not a list per line |
| `9c2f9cb` | C4 | DS1 | Refuse malformed versions with 422 instead of a 500 |
| `940decd` | C5 | DS2 | Share the version format between the catalog and FakeDatasetCatalog |
| `69d93e4` | C6 | DS7 + DS8 | Find the sample versions without tools.schema and drop a dead return |
| `2bc01df` | C7 | DS6 | Describe the fake, the digest's JSON and audio versions as they are |
| `377c87f` | C8 | P2K1 | Refuse files.csv durations and sample rates that break a dataset later |
| `55f1d6e` | C9 | P2K2 | Give each uid in uids.jsonl to one segment, the nearest without one |
| `6633514` | C10 | P2K3 | Say in version.md what readers of a version actually enforce |
| `1971416` | D6 (follow-up) | CU5 | Give FakeCatalog the server's rule for integer output ids |

### Second wave: defects from the second pass and follow-ups from the first wave (47 commits)

| Commit | Ticket | Audit finding | What changed |
|---|---|---|---|
| `ff5e57c` | E2-1 | P2U2 | Show a deleted version or dataset as gone on its page |
| `7b8f40b` | E2-2 | P2U3 | Ignore a dropped zip while Import package is busy |
| `5bd2ffd` | E2-3 | P2U5 + P2U6 | Make Docs and Model contract show the calls that work |
| `e6b3c09` | E2-4 | P2U4, UI half | Send the dry run's file count with "Delete N files" |
| `8d8fc66` | E2-5 | P2U10 | Show the first 50 changed files on Compare versions |
| `de578a3` | E2-6 | lane E follow-up | Keep a newer upload of a file when an older one fails |
| `37792b8` | E2-7 | lane H follow-up | Refresh every live query after the event stream reconnects |
| `e75d07a` | W5-1 |  | Show a removed model as gone on its page |
| `37c6607` | W5-2 |  | Cap the lists again when Compare versions changes pairs |
| `89fb4f0` | W5-3 |  | Give the enhancers' real sample rates on Model contract |
| `09c0054` | W2-1 | P2C7, security | Redact the key in a hosted provider's error reply at the gateway |
| `c3b0b3e` | W2-2 | P2C8 | Check a hosted endpoint's parent before storing it |
| `129c77d` | W2-3 | P2C9 | Drop or reread a hosted OIP signature when PATCH changes the contract |
| `a05433d` | W2-4 | P2C10 | Write display_name, tags and parent_id on PATCH /api/hosted/{id} |
| `0a29fa8` | W2-5 | P2C11 | Keep repeated reply headers apart in the gateway |
| `29e7fe2` | W2-6 | P2C12 | Route GET /v2/models/{id} through the gateway for OIP metadata |
| `0be4900` | W1-1 | P2C1 | Lift the batch limits again before each length of the sweep |
| `3c45f16` | W1-2 | P2C2 | Send an OpenAI transcription server at most 32 requests at once |
| `7821e0c` | W1-3 | P2C3 | End a length on a slow batch also when it gave no memory reading |
| `c2b2667` | W1-4 | P2C4 | Refuse batch API values that break a start or the response |
| `cca1213` | W1-5 | P2C5 | Compare a running service's limits with a start at its own memory |
| `ed0eef3` | W1-6 | P2C6 | Say that the torch reading is kept, not the larger one |
| `6c38555` | W4-1 | lane G follow-up | Hash only the files a base build reads in base_digest |
| `165e1bd` | W4-2 | lane D and F follow-ups | Close the catalog clients stack.py and batch.py open |
| `96f70d0` | W4-3 | lane D follow-up | Keep odd batch views and plain error strings from breaking aicore |
| `464e9d3` | W4-4 | lane G follow-up | Refuse true as a speaker count and give zero-row SV requests a clear 400 |
| `0f5e7f0` | W4-5 | docs | Name the invalid_request code and kokoro-tts's real torch cap in docs |
| `3f25246` | W4-1 (follow-up) | lane G follow-up | Refuse a base COPY line that base_digest cannot read |
| `493cc77` | W4-5 (follow-up) | docs | Say what AISVC_VRAM_GB holds in the aisvc gpu docstring |
| `6295833` | W3-1 | P2C13 + lane D follow-up | Refuse non-finite index spans and give output ids back as stored |
| `fede35a` | W3-2 | P2C14 + lane H follow-up | Count a running GPU model as a load once another GPU model loads |
| `b3848f5` | W3-3 | P2C15 | Verify hosted endpoints on their own worker thread |
| `93b6e09` | W3-4 | lane B follow-up | Keep the report when a conformance check raises any error |
| `0ae9aa9` | W3-5 | lane A follow-up | Keep the pin of a manual start that arrives during a stop |
| `a611202` | W3-6 | lane A follow-up | Wait longer between tries of a spin-down stop that keeps failing |
| `02548e3` | W3-7 | lane A follow-up | Load broken manifests once per change, not on every call |
| `a441a5e` | W3-8 | P2C16 | Stop a half-started container after a failed start on every runtime |
| `82f9b2b` | W3-9 | P2C17 | Declare python-multipart as a direct dependency |
| `25b3565` | W3-10 | P2K10 | List every catalog API route in the README and drop /redoc |
| `171492f` | W3-8 (follow-up) | P2C16 | Drop the Docker setup from the failed clean-up stop test |
| `6d370e4` | W6-1 | P2U1 | Run the audio PUT's store calls on a worker thread |
| `5d73b80` | W6-2 | P2U4, server half | Delete nothing when clean-up finds more files than the dry run showed |
| `cc94c8a` | W6-3 | P2U7 | Look segments up by uid and start time when comparing two versions |
| `35942ba` | W6-4 | P2U8 | Probe each file of an unpacked zip once |
| `f46581e` | W6-5 | P2U9 | Count the store's audio for the stats in one pass |
| `39c3682` | W6-6 | lane C follow-up | Refuse a references speaker that is not a string |
| `fd1b029` | W6-7 | lane C follow-up, docs | Name aicore.dataset_format in the aicore docstring |

### Behaviour-preserving refactors (21 commits)

| Commit | Ticket | Audit finding | What changed |
|---|---|---|---|
| `f457c3b` | R3-1 | CF15b + CF15c | Give the pinned README rule one owner in model_cards |
| `6c68bab` | R3-2 | CF16 | Split cards.build into one helper per part of the card |
| `8960292` | R3-3 |  | Test that the copied base paths and route maps match the contract |
| `3e8e11e` | R3-4 | PT20 part | Delete the unused _FULL_SHA pattern from model_cards |
| `677d066` | R2-1 | ARCH9 + PT5 + CR17 | Share container inspection and survive a container removed mid-list |
| `864fed3` | R2-2 | PT14 | Compute a local model's host folder in one place |
| `9cb7f91` | R2-3 | PT16 | Split the manifest rules by concern and drop a duplicate error |
| `242bd44` | R2-4 | PT17 | Break load_platform into one helper per step |
| `421167a` | R2-5 | PT20 | Drop unused Config.host() and the float form of render_command's util |
| `43762e3` | R2-6 | PT21 | Write out check=False, give the env default as text, declare markupsafe |
| `244b5f4` | R2-7 |  | Give the kebab-case pattern a public name |
| `2a9465e` | R2-8 | PT19 | Stream a tarball's weights from where they are instead of staging them |
| `a90b8d3` | R1-1 | CR15 | Remove dead scheduler fields, a forwarder and unused imports |
| `a638427` | R1-2 | CR16 | Let the scheduler refuse hosted starts and stops on its own |
| `f4cc9e4` | R1-3 | ARCH10 | Drop three util_limit checks that can never fire |
| `482b7e9` | R1-4 | ARCH5 | Lease the verifier's and profiler's model through Scheduler.hold |
| `dd0ab29` | R1-5 | CR18 | Split the scheduler's and planner's longest functions into phases |
| `0a96d58` | R1-6 | CR20 | Parse each weights.yaml again only when it changes |
| `bf09a60` | R1-7 | CF15a | Parse stored timestamps with one helper in catalog/db.py |
| `ba0ce0a` | R1-4 (follow-up) | ARCH5 | Keep a batch profile measured before its model crashed |
| `1cc0d71` | R2-8 (follow-up, P4T1) | PT19, P4T1 | Keep the old tarball when a bundle fails while it is written |

### Third wave: defects from the third to sixth passes, the sanctioned test edit and docs (35 commits)

| Commit | Ticket | Audit finding | What changed |
|---|---|---|---|
| `074d24a` | T1 (P3I1) | P3I1 | Score real speech in the utmos real-model test |
| `8748dd5` | CR22 |  | Make the catalog README say what the scheduler and events do |
| `7f2d2ba` | X3-1 | P3T2 | Show the status table when the catalog does not answer |
| `38cd9ed` | X3-2 | P3T4 | Refuse COPY forms and ignore patterns base_digest cannot read |
| `94a6dfb` | X3-3 | P3T6 | Print one error line from just batch when no catalog answers |
| `cc8d740` | X4-1 | P3I2 | Refuse a utmos SAMPLE_RATE that resampling cannot handle |
| `01c2241` | X4-2 | P3I3 | Refuse silero-vad parameters outside their range |
| `a8dbdff` | X4-3 | P3I4 | Accept only whole numbers in POST /platform/batch-limits |
| `eda8e65` | X4-4 | P3I5 | Say which images still answer with aisvc 0.1.0 |
| `62a207b` | X4-5 | P3I6 | Capitalise text-normalizer sentence starts in any script |
| `1e5ebbe` | X4-6 | P3I8 | Point the Dockerfile base defaults at the 0.2.0 bases |
| `963ab25` | X1-1 | P3T1 | Free a failed ASR batch before its files run again one by one |
| `0f53ffc` | X1-2 | P3T3 | Treat AI_HOST=[::1] as the local host in aicore URLs |
| `3484002` | X1-3 | P3T5 | Build openai-inspired responses inside guard |
| `1cdcae5` | X1-4 | P3T7 | Give the RE-USE chunking limit in samples on the contract page |
| `28eb602` | X1-5 | P3T8 | Make FakeCatalog leases and index hits match the server |
| `198f4d7` | X1-6 | P3C7 + lane W6 follow-up | Refuse reference segments whose times are NaN or infinite |
| `d704d22` | X1-6 (follow-up) | P3C7 + lane W6 follow-up | Refuse a references time too large for a float instead of a 500 |
| `eacaf3d` | X4-6 (follow-up) | P3I8 | Point ark-asr and meralion's base defaults at 0.2.0 and guard them |
| `606f174` | X2-1 | P3C1 | Stop a failed start's container before marking the start failed |
| `91f6015` | X2-2 | P3C2 | Wait for an evicted GPU model to stop before starting the next one |
| `3d0b42a` | X2-3 | P3C3 | Drop a pin when the request that was to bring its model up ends first |
| `9fa90ea` | X2-4 | P3C4 | Stop an import whose catalog stops answering partway through |
| `3385781` | X2-5 | P3C5 | Refuse a hosted base_url that cannot be read as a URL with a host |
| `0fda0cc` | X2-6 | P3C6 | Never keep the old model's signature when PATCH changes an OIP build |
| `0e6ec6e` | X2-7 | P3C8 | Pass upstream header bytes through the gateway as they were sent |
| `db63dc8` | X2-5 (follow-up) | P3C5 | List a hosted endpoint whose stored base_url no longer parses |
| `100b542` | X2-2 (follow-up, P4C1) | P3C2, P4C1 | Count a swap after an earlier failed spin-down stop |
| `de076e9` | X2-5 (follow-up, P4C2) | P3C5, P4C2 | Answer 422 for the connection test of a stored bad base_url |
| `d3fbfae` | X2-4 (follow-up, P4C3) | P3C4, P4C3 | Say which step stopped when the catalog stops answering an import |
| `59f4c2e` | X4-5 (follow-up, P4T2) | P3I6, P4T2 | Capitalise sentence starts with the titlecase form in text-normalizer |
| `921f55c` | X3-2 (follow-up, P4T3) | P3T4, P4T3 | Refuse COPY sources with build variables, heredocs and escape directives |
| `282a8b3` | X3-1, R2-7 (follow-ups, P4T4, P4T5) | P3T2, P4T4, P4T5 | Name the commands a wedged catalog stops, and drop an unused alias |
| `e92e2db` | X2-2 (follow-up, P5-1) | P3C2 | Mark an eviction as a swap only when it is one |
| `806a930` | X2-2 (follow-up, P6-1, P6-2) | P3C2 | Say the swap count can be one off, and fix a test docstring |

The commit that adds this report is not in the table. Each commit's message explains the problem, the change and the test that locks it.
