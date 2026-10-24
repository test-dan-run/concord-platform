# Dataset version format

A dataset version in the Dataset Catalog is a folder that holds a reference package without its audio. The
platform owns the format, and the catalog service stores each version under
`${DATASET_STORE}/versions/<dataset_id>/<n>/`. Concord Bench and Concord Label read versions from that folder on
the store's mount. Their CONTRACTS.md §2 defines a reference package zip as a dataset version plus `audio/`, with
a link to this file. The design is in [../../DATASET_CATALOG_PLAN.md](../../DATASET_CATALOG_PLAN.md) §5.

The format is `concord-reference-package`, format version 3. Readers also take versions 1 and 2. Version 1 has no
labels, version 2 adds labels, and version 3 allows a version with audio only (§6). A reader refuses a format
version it doesn't know, with a message that names the versions it does know. Fields a reader doesn't know are
ignored, so a newer writer can add fields without a new format version.

Times are seconds as JSON numbers. Timestamps are ISO 8601 in UTC with a `Z`. Paths are relative POSIX paths
without `..`. JSON files are UTF-8.

The sample versions are in [samples/](samples/), and `just validate` checks them with `check_folder` in
`catalog/datasets/versions.py`. The catalog runs the same check on every version it receives.

| Sample | What it shows |
|---|---|
| `version-labelled` | format version 2: speakers and text (SegLST), uids, and labels on segments, files and speakers |
| `version-asr` | format version 2: text without speakers (`segments.jsonl`), with labels on segments |
| `version-diar-v1` | format version 1: speakers without text (RTTM) |
| `version-audio-only` | format version 3: audio only, so `label_types` is empty and there is no `refs/` folder |

## 1. Folder

```
<n>/
  manifest.json
  files.csv
  refs/<one references file>   required unless label_types is empty
  refs/uids.jsonl              optional
  refs/attributes.json         optional, format version 2 and later
```

A version has no `audio/` and no `outputs/` folder. The audio is in the store, found by its sha256. Model outputs
are shared through the Model Catalog's output index, so a version doesn't carry `outputs/seg.jsonl`.

## 2. `manifest.json`

| Field | Required | Meaning |
|---|---|---|
| `format` | yes | `"concord-reference-package"` |
| `format_version` | yes | `1`, `2` or `3` |
| `name` | yes | the dataset name the writer suggests |
| `created_at` | yes | when the version was published (the catalog sets it) |
| `language`, `description` | no | as on a dataset |
| `tags` | no | a list of strings |
| `source` | in the catalog | `{"app", "install_id", "dataset_uid", "dataset_name"}` |
| `export_version` | with `source` | the version number `n` |
| `scope` | no | `"complete"` (fully reviewed files) or `"any"` (any file with a confirmed segment) |
| `label_types` | yes | a subset of `["asr", "diar"]`, which says what the references file feeds. It may be empty only in format version 3. |
| `counts` | yes | `{"files", "segments", "hours"}`. `files` must equal the number of rows in `files.csv`. |
| `audio_included` | yes | `"none"` in a version. A package zip may say `"all"` or `"partial"` (§7). |
| `review` | no | review statistics, shown in an import preview |
| `models` | no | Model Catalog ids of the models that made the pseudo labels |
| `normaliser` | no | the suggested WER normaliser, as an id (`basic`) or a key (`basic@1`) |

The catalog fills four fields when it stores a version, whatever the writer sent:

- `source` is `{"app": <the publishing app>, "install_id": "dataset-catalog", "dataset_uid": <dataset id>,
  "dataset_name": <dataset name>}`. Versions of one catalog dataset share `install_id` and `dataset_uid`, so Bench
  links the datasets it makes from them in one version chain.
- `export_version` is the version number `n`.
- `created_at` is the time of publishing.
- `audio_included` is `"none"`.

## 3. `files.csv`

The header is `path,sha256,duration,sample_rate`, and there is one row per audio file. `path` is the file's path in
the dataset. `sha256` is the hex SHA-256 of the original bytes, which is the file's key in the store. `duration` is
in seconds, and `sample_rate` is an integer in Hz. File stems (the file name without its extension) are unique,
ignoring case, because references are matched to audio by stem.

## 4. References

When `label_types` isn't empty, `refs/` holds exactly one references file besides `uids.jsonl` and
`attributes.json`. In format version 1, `attributes.json` counts as a references file, so a version 1 folder can't
have one. The references file is matched to audio by stem, and it holds only confirmed segments with their own
(unpadded) times. Label writes one of these files:

| File | When | Each record |
|---|---|---|
| `refs/segments.seglst.json` | speaker and text (`label_types` asr and diar) | SegLST: `{"session_id": stem, "speaker", "start_time", "end_time", "words": text}` |
| `refs/segments.jsonl` | text without speaker (`label_types` asr) | `{"file": path, "start", "end", "text"}` |
| `refs/refs.rttm` | speaker without text (`label_types` diar) | RTTM `SPEAKER <stem> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>` |

A version may carry any other references format that the apps' annotated upload reads, e.g. a `.stm` file from
another tool. The catalog stores such a file unchanged. It shows segments and compares versions for the three
files above and for STM, and it reports the segments of any other format as unknown.

From format version 2, a record in `segments.jsonl` or `segments.seglst.json` may carry
`"attributes": {"<name>": value, ...}`, which are labels of that one segment. A value is a string, a number or
true / false. A nested object is read with dotted names, so `{"accent": {"region": "north"}}` is the label
`accent.region`.

## 5. `refs/uids.jsonl` and `refs/attributes.json`

`refs/uids.jsonl` has one `{"file": path, "start", "end", "uid"}` per reference segment. It gives the writer's
segment uid, so flags can come back to the writer. A uid is matched to a segment by file and by times within 0.01
seconds.

`refs/attributes.json` holds labels of whole files and of speakers, keyed by the file's path in `files.csv`:

```
{"files": {"<path>": {"<name>": value, ...}, ...},
 "speakers": {"<path>": {"<speaker>": {"<name>": value, ...}, ...}, ...}}
```

Both keys are optional, and values follow §4. A file label applies to every segment in the file. A speaker label
is read with the prefix `speaker.`, so `{"gender": "F"}` under speaker `agent` is the label `speaker.gender` on
each of `agent`'s segments in that file. A label on a segment record wins over a file or speaker label of the same
name. A path that isn't in `files.csv` makes the version invalid.

## 6. Versions with audio only (format version 3)

When `label_types` is empty, the version has no references, so `refs/` is empty or missing. A person can open
such a version in Concord Label to start labelling. Concord Bench lists it but can't import it, because Bench
needs references. The catalog makes these versions from its Upload audio page, with `counts.segments` set to 0.

## 7. Reference package zip

A reference package is a version plus the audio, as a folder or as a zip of that folder. The folder may be the
zip's root or its only top level folder.

- `audio/<path>` holds the original bytes of a file in `files.csv`. A package may leave out files the receiver
  already has, and `audio_included` says whether it has `"all"`, `"partial"` or `"none"` of them. Bytes that don't
  match their sha256 make the package invalid.
- `outputs/seg.jsonl` may hold seg model outputs, as Concord Bench's CONTRACTS.md §2.5 describes. The catalog
  doesn't keep them.

The catalog accepts a package zip when an app publishes a version and on its Import package page. It moves the
audio into the store and keeps the rest as a version. A version downloaded from the catalog is a package zip,
with or without its audio.

## 8. Digest

A version's `digest` identifies its content, so publishing the same content again returns the existing version.
It is the hex sha256 over these parts, in this order:

1. First, `manifest.json` as canonical JSON (keys sorted, no spaces) without `source`, `export_version`,
   `created_at` and `audio_included`.
2. Second, the bytes of `files.csv`.
3. Third, the bytes of each file under `refs/`, sorted by file name.

Each part goes into the hash as `<name>\n<length in bytes>\n<bytes>`, where `<name>` is `manifest.json`,
`files.csv` or `refs/<file name>`.
