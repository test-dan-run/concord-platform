# Dataset Catalog: design and plan

Planned 2026-10-24. This plan adds a Dataset Catalog to the platform, next to the model catalog, which is
renamed the Model Catalog. Concord Bench and Concord Label both use the Dataset Catalog. The plan changes
three repositories: the platform, Concord Bench (`../apps/concord-bench`) and Concord Label
(`../apps/concord-label`). Section numbers without a file name are in this document.

The Dataset Catalog keeps versioned datasets that both apps can read. Audio lives once, in a store keyed by
the sha256 of the original bytes, and both apps mount that store read-only. A dataset version is an
immutable record of which audio files a dataset has, plus its references and labels. Each app keeps its
own upload. Label publishes a version when references are ready, which replaces "Send to Bench". Bench
imports a version without copying any audio. The platform gets a standalone UI for the Dataset Catalog, as
a second tab beside the Model Catalog.

The work is split into milestones D0 to D7 (§11). The owner's decisions are in §1. The open questions of
the first draft are settled, and §13 records how.

---

## 1. Decisions

Made by the owner on 2026-10-24.

| Topic | Decision | Section |
|---|---|---|
| Where datasets live | A small dataset service in the platform (option C), not ClearML Data and not DVC. | §2 |
| Uploads | Bench and Label keep their own upload pages. A person never has to go to the platform to add audio. | §8, §9 |
| Standalone UI | The platform has its own UI for the Dataset Catalog, and it accepts raw audio uploads as well as reference package zips. | §6 |
| Naming | The existing catalog is called the Model Catalog. The Dataset Catalog is a separate tab in the same UI. | §3 |
| Branches | All work happens on a branch off `main` in each repository: `dataset-catalog` in the platform, Bench and Label. | §11 |
| Deleting versions | A version can be deleted. | §5.3 |
| Store location | The store defaults to `~/ai-data/datasets`. | §4 |
| Label's Sync | Out of scope. Sync's code doesn't change. | §8.1 |
| Version format | The version format moves into the platform's `contracts/`. | §5.1 |

This plan replaces two lines of MODEL_CATALOG_REDESIGN.md. Line 16 lists datasets as out of scope, and
line 135 says "datasets stay in the apps". D1 changes both lines to point here (§11).

## 2. Why the platform and not the apps

Each app keeps its own copy of every audio file today, in `data/originals/<sha256><ext>`. Label sends a
reference package to Bench, and Bench asks which files it lacks, so only those travel. A dataset in Label
has no versions, and Bench makes a new dataset (`<name> v<N>`) for each package it receives from the same
Label dataset. A plain upload in Bench can't be marked as a new version of an existing dataset at all.

A shared catalog fixes these gaps in one place:

- **One copy of the audio.** Both apps read the same files, so nothing is copied between them.
- **Versions for every dataset.** Every dataset gets numbered versions with a parent link, wherever it
  came from.
- **One list of datasets.** The UI and both apps can show and compare every version.

The platform already runs a shared service that both apps call, the model catalog's output index, which is
also keyed on the audio sha256. The Dataset Catalog follows the same pattern. The apps already depend on
the platform's Python client (`clients/aicore`), so the shared client code goes there. That avoids adding a
module that would be copied by hand between Bench and Label (Label's AGENTS.md lists the modules the apps
copy today).

## 3. Names and the catalog service

The process at port 8700 is called the catalog service in this plan. It serves two catalogs:

- **The Model Catalog** is everything it does today: models, builds, verification, the gateway, leases, the
  output index and export.
- **The Dataset Catalog** is new: datasets, versions and the audio store.

The rename is a change to display text and documentation. Identifiers that the apps depend on keep their
names, because renaming them would break both apps and every deployed `.env`.

| Changes to "Model Catalog" | Stays as it is |
|---|---|
| UI page title and `<title>`, the Nav brand (`catalog/ui/src/app/Nav.tsx:71`), the FastAPI title, the heading of `catalog/README.md`, the title of MODEL_CATALOG_REDESIGN.md, the root README, and the text of both apps that links to it ("Open the Model Catalog") | The `catalog/` package, `python -m catalog`, the `aicore.catalog` module, `AICORE_CATALOG`, `CATALOG_URL`, `CATALOG_UI_URL`, the `catalog` hostname on `ai-net`, port 8700, the `catalog-*` Justfile recipes, the image name, `/api/models`, `/m/` and `/v2/`, and the UI paths `/` and `/models/:id` that both apps link to |

The old static page titled "AI Service Catalog" (`ui/template.html.j2`, opened by `just catalog`) is renamed
"AI service list", so it isn't confused with either catalog.

## 4. Storage

All of the Dataset Catalog's state lives in one folder, so it can be backed up as one unit. The folder is
set by `DATASET_STORE`, which defaults to `${AI_DATA}/datasets`, and `AI_DATA` defaults to `~/ai-data`. It
sits next to `AI_ASSETS` (`~/ai-assets`, the model weights) and outside the platform repository's `run/`
folder, which holds state that can be rebuilt.

```
${DATASET_STORE}/
  datasets.db                 SQLite: datasets, versions, audio, holds, uses (§4.2)
  audio/<sha256[0:2]>/<sha256>  original audio bytes, no extension
  versions/<dataset_id>/<n>/  one folder per version (§5)
  tmp/                        uploads in progress
```

### 4.1 Audio

- **Key.** An audio file is stored under the hex sha256 of its original bytes. The key is the same one both
  apps use today (CONTRACTS.md §2.2) and the one the output index uses.
- **No extension on disk.** The file has no extension, because two uploads with the same bytes can arrive
  as `.wav` and `.WAV`. The extension, size, duration, sample rate and channel count are kept in
  `datasets.db`. ffmpeg detects the format from the content when it reads the file.
- **One writer.** Only the catalog service writes to the store. It writes each file to `tmp/` first and
  checks the sha256. It then sets the mode to 0644 and moves the file into place, so a reader never sees a
  partial file.
- **Read-only for the apps.** The apps read audio straight from the mount. They never write to it.
- **Derived audio stays in each app.** The 16 kHz FLAC, the peaks and the enhanced copies are not in the
  store. Each app derives them, as it does today, and keys them by the original sha256.

The catalog service's compose file mounts the store read-write at the same path on the host and in the
container, the way it mounts `PLATFORM_DIR`. Bench and Label mount it read-only and set `DATASET_STORE` to
the path they see. In `just dev`, the apps run on the host and read the host path directly.

### 4.2 Database

The Dataset Catalog has its own SQLite file (`datasets.db`), kept apart from `run/catalog/catalog.db`. It
uses the catalog's `DB` class (`catalog/db.py`: raw SQL, WAL, one schema string). The model catalog's
database holds state that can be rebuilt, while the dataset records can't be rebuilt, so the two files are
backed up differently.

| Table | Columns | Notes |
|---|---|---|
| `dataset` | `id`, `name`, `language`, `description`, `tags`, `created_by`, `created_at`, `updated_at` | `id` is a uuid4 hex string (32 characters), the same form as Label's `dataset.uid`, so it fits the existing `dataset_uid` fields in both apps |
| `version` | `dataset_id`, `n`, `parent_n`, `digest`, `manifest`, `counts`, `label_types`, `published_by`, `published_at` | `n` is 1, 2, ... per dataset, and a deleted version's number is never used again (§5.3) |
| `version_file` | `dataset_id`, `n`, `path`, `sha256`, `duration`, `sample_rate` | indexed on `sha256` |
| `audio` | `sha256`, `size`, `ext`, `duration`, `sample_rate`, `channels`, `state`, `created_at` | `state` is `partial` (upload in progress) or `ready` |
| `hold` | `app`, `sha256`, `created_at` | audio an app needs that may not be in any version yet (§4.3) |
| `use` | `dataset_id`, `n`, `app`, `ref`, `created_at` | an app's dataset made from a version, e.g. a Bench dataset id (§5.3) |

### 4.3 Holds and clean-up

Audio is uploaded before it belongs to any version. For example, Label stores audio at upload, and a person
may annotate it for weeks before publishing. An app therefore places a hold on each sha256 it keeps in one
of its datasets, and it releases the hold when it deletes that dataset. A dataset an app made from a
catalog version holds its audio too, so deleting the version (§5.3) never removes audio an app still
reads.

Clean-up (`POST /api/store/gc`) deletes an audio file only when all of these are true:

- No version lists it.
- No app holds it.
- It has been in the store for more than 24 hours, so an upload that hasn't been published or held yet is
  safe.

Clean-up always shows a dry run first, with the files and bytes it would delete. It runs only when a person
starts it from the Storage page (§6) or with `just datasets-gc`.

## 5. Versions

### 5.1 Format

A version is a reference package (format version 2, Bench's CONTRACTS.md §2), stored as a folder, without
`audio/` and without `outputs/`:

```
versions/<dataset_id>/<n>/
  manifest.json
  files.csv
  refs/<one references file>     optional when label_types is empty (§5.2)
  refs/uids.jsonl                optional
  refs/attributes.json           optional
```

Keeping the existing format means both apps reuse code they already have. Label already writes it, and
Bench already reads it, including audio found by sha256 instead of under `audio/`. The catalog fills in
`manifest.source` as `{"app": <publisher>, "install_id": "dataset-catalog", "dataset_uid": <dataset id>,
"dataset_name"}` and sets `export_version` to `n`. Bench's existing version chain (`source_key`,
`export_version`, `previous_version_id`) then links the Bench datasets made from one catalog dataset with
no change.

- **Seg outputs are left out.** `outputs/seg.jsonl` is not part of a version, because the output index
  already shares model outputs between the apps. Label records its seg outputs in the index today, and
  Bench looks them up there before it calls a model.
- **The platform owns the version format.** The format moves into the platform's `contracts/`, as
  `contracts/datasets/version.md` with sample folders, and the platform's contract checks validate the
  samples. Bench's and Label's CONTRACTS.md §2 then define a reference package zip as "a dataset version
  plus `audio/`", with a link to the platform's file. People and other tools can still hand a zip to Bench,
  and the Dataset Catalog UI can import one (§6).

### 5.2 Format version 3

Version 3 of the format makes one change. When `label_types` is empty, the folder has no references file.
That allows a version with audio only, which a person can open in Label to start labelling (§8.3). Bench
lists such versions but can't import them, because it needs references (its `NO_REFERENCES` rule). Readers
still take versions 1 and 2.

### 5.3 Rules

- **A version never changes.** Its `digest` is the sha256 of `manifest.json`, `files.csv` and the
  `refs/` files in a fixed order, with the catalog's own fields left out.
- **Publishing the same content again returns the existing version.** If a publish has the same digest as
  the dataset's latest version, the catalog returns that version with status 200, so a double click
  doesn't make two versions.
- **Every audio file must already be in the store.** If any file is missing, the publish fails with 422
  and lists the missing sha256s.
- **The parent must be the latest version.** A publish names its `parent` (or none, for version 1). If the
  dataset's latest version isn't that parent, the catalog answers 409 and names the latest version. The app
  then asks the person whether to publish anyway, which sets `force: true`, and the new version's
  `parent_n` still records the version it was based on. A parent that has been deleted counts as stale in
  the same way.
- **A version can be deleted.** Deleting it removes its row, its `version_file` rows and its folder. Its
  number isn't used again, and a later version whose parent it was shows the parent as "v2 (deleted)".
  When apps have made datasets from the version (`use` rows), the UI lists them and asks the person to
  confirm. Those datasets keep working, because each app keeps its own copy of the references and holds the
  audio (§4.3), and their link to the version shows that it was deleted.
- **A dataset can be deleted** with all its versions, under the same rules.
- **Audio isn't deleted with a version.** Files that no version lists and no app holds become candidates
  for clean-up (§4.3).

## 6. Platform UI

The catalog UI (`catalog/ui`) gets a switch at the top of the Nav, under the brand: "Model Catalog" and
"Dataset Catalog". It uses the existing `Toggle` component. The Nav groups below the switch change with
the tab. The Model Catalog keeps its pages and paths exactly as they are. The Dataset Catalog's pages live
under `/datasets`, which collides with nothing (the gateway owns `/m/` and `/v2/`).

| Page | Path | Shows | Actions |
|---|---|---|---|
| Datasets | `/datasets` | Each dataset's name, language, latest version, files, hours, label types, label names, who published it last and when | Search and filter by language, label type and app |
| Dataset | `/datasets/:id` | The version history (number, date, publisher, parent, counts, and a short summary of changes from the parent), and which app datasets were made from each version | Edit the name, description and tags; download a version as a reference package, with or without audio; add audio (opens Upload audio); delete a version or the dataset (§5.3) |
| Version | `/datasets/:id/v/:n` | The files table (path, duration, sample rate, segments, labels) with a player that reads from the store, the references of one file, and label statistics | Open the same file in Label when Label is set up (§9) |
| Compare | `/datasets/:id/compare?a=&b=` | Files added and removed, files whose references changed (segments added, removed and edited, with a text diff per segment), and label changes | none |
| Upload audio | `/datasets/upload` | Files, a folder or zips of audio, with per file progress, and the files the store already has | Make a new dataset whose version 1 has audio only, or add the files to a dataset whose latest version has audio only (§6.1) |
| Import package | `/datasets/import` | A preview of an uploaded reference package zip, like Bench's import preview | Import it as a new dataset or as a new version of an existing one |
| Storage | `/datasets/storage` | Store size, audio files that are in versions, held only, or in neither, and holds per app | A clean-up dry run, then the clean-up itself (§4.3) |

The Storage widget in the Nav replaces the GPU widget while the Dataset Catalog tab is open. The SSE stream
(`/api/events`) gains a `dataset` event, so pages update when an app publishes.

### 6.1 Raw audio uploads

The Upload audio page takes audio files without references, so a person can add a dataset in the platform
and then label it in Label (§8.3). It works like the apps' upload pages:

1. First, the browser hashes each file and asks `POST /api/store/missing`, so files the store already has
   aren't sent again.
2. Second, it sends each missing file in 16 MB chunks to `PUT /api/store/audio/{sha256}`, which resumes
   after a dropped connection, and commits it. Zips are sent whole and unpacked by the catalog service.
3. Third, the catalog service probes each file with ffprobe for its duration, sample rate and channels.
   Files ffprobe can't read are listed on the page and left out.
4. Fourth, the page makes a version with audio only (§5.2). For a new dataset it is version 1. For an
   existing dataset, it is the next version, with the latest version's files plus the new ones.

The page adds files only to a dataset whose latest version has audio only. A dataset with references gets
new audio through Label, where the new files are labelled and then published, so a version never mixes
files that have references with files that don't. The page reads no annotation files, because the
annotation parsers live in the apps. References come into the catalog through a reference package zip
(Import package) or from an app. The catalog service's image gains ffmpeg for ffprobe.

## 7. API

All paths are under `/api`, like the rest of the catalog service. Dataset routes live in a new package,
`catalog/datasets/` (`store.py`, `versions.py`, `routes.py`, `gc.py`). They are added to the app by one
call in `catalog/app.py`, which keeps the change to that file small (§12).

### 7.1 Audio store

| Call | Body / response |
|---|---|
| `POST /api/store/missing` | `{"sha256": [...]}` -> `{"missing": [...]}`: the files the store doesn't have, or has only partly |
| `PUT /api/store/audio/{sha256}?ext=&offset=` | raw bytes from `offset` -> `{"size"}`; 409 when `offset` isn't the current size, so an upload can resume |
| `POST /api/store/audio/{sha256}/commit` | none -> the audio record; 422 if the bytes don't hash to `sha256`, and the partial file is removed |
| `GET /api/store/audio/{sha256}` | the original bytes, with the content type from `ext` (for the UI player; the apps read the mount) |
| `POST /api/store/holds` | `{"app", "add": [...], "remove": [...]}` -> `{"held"}` |
| `GET /api/store/stats` | sizes and counts for the Storage page |
| `POST /api/store/gc` | `{"dry_run": true}` -> `{"files", "bytes", "sample": [...]}`; the same call with `false` deletes them |

### 7.2 Datasets and versions

| Call | Body / response |
|---|---|
| `GET /api/datasets` | `?q=&language=&label_type=` -> datasets with their latest version |
| `POST /api/datasets` | `{"name", "language"?, "description"?, "tags"?, "app"}` -> 201 dataset |
| `GET /api/datasets/{id}`, `PATCH /api/datasets/{id}`, `DELETE /api/datasets/{id}` | dataset with its versions; PATCH changes name, description and tags; DELETE removes the dataset and all its versions (§5.3) |
| `POST /api/datasets/{id}/versions` | multipart: a zip of the version folder, plus `parent`, `app` and `force` -> 201 version, 200 when the digest matches the latest version, 409 on a stale parent, 422 when the folder is invalid or audio is missing |
| `GET /api/datasets/{id}/versions/{n}` | version record with its manifest and counts |
| `GET /api/datasets/{id}/versions/{n}/files` | `?offset=&limit=` -> files with their segment counts and labels |
| `GET /api/datasets/{id}/versions/{n}/refs?path=` | the reference segments of one file |
| `GET /api/datasets/{id}/versions/{n}/package?audio=true` | the version as a reference package zip, streamed |
| `GET /api/datasets/{id}/compare?a=&b=` | the comparison of §6 |
| `DELETE /api/datasets/{id}/versions/{n}` | none -> 204 (§5.3) |
| `POST /api/datasets/{id}/versions/{n}/uses`, `DELETE` the same with `?app=&ref=` | `{"app", "ref"}` -> 204 |
| `POST /api/datasets/import` | multipart reference package zip, plus `dataset_id`? -> 201 version (for the UI's Import package page) |
| `POST /api/datasets/{id}/versions/audio` | `{"files": [{"path", "sha256"}], "parent"}` -> 201 version with audio only (for the UI's Upload audio page, §6.1); 409 when the latest version has references |
| `POST /api/store/unpack` | `{"sha256"}` of a committed zip -> its audio files, each stored and probed, with their paths in the zip |

Apps read a version's folder from the mount (`${DATASET_STORE}/versions/<id>/<n>/`) rather than over HTTP.

### 7.3 Python client

`aicore.datasets` gets `DatasetCatalog` (same base URL as `aicore.catalog`, from `AICORE_CATALOG`) and
`FakeDatasetCatalog`, which keeps everything in memory and in a temporary folder, the way `FakeCatalog` does
for the Model Catalog. The client has these helpers:

- `upload_audio(path)` uploads a file in 16 MB chunks, resumes on 409 and commits, and it skips files that
  `missing` doesn't list.
- `audio_path(sha256)` returns the file's path in the mounted store.
- `version_dir(id, n)` returns a version's folder in the mounted store.
- `publish(id, folder, parent, force=False)` publishes a version.
- `hold` and `release` place and release holds.

## 8. Changes to Concord Label

### 8.1 Audio at upload

Label's upload page stays as it is. During ingest, each original goes to the store through
`upload_audio` instead of being copied to `data/originals/`, and Label places a hold on it.
`paths.original` returns the store path when `DATASET_STORE` is set, and the derived FLAC and peaks stay in
`data/`. Deleting a Label dataset releases the holds on its files, unless another Label dataset uses the
same audio.

Label's Sync (label packages between Label installs) is out of scope, and its code doesn't change. Audio
that arrives through Sync is still copied to `data/originals/`. So `paths.original` looks in the store
first and then in `data/originals/`, and publishing (§8.2) uploads any file the store doesn't have yet.

### 8.2 Publish replaces Send to Bench

The "Send to Bench" panel becomes "Publish to Dataset Catalog", with the same choices of name, scope and
tags. The "include audio" and "include outputs" choices go away, because the audio is already in the store
and the outputs are in the index.

1. On the first publish, Label creates a catalog dataset and stores its id on the Label dataset, in a new
   `catalog_dataset_id` column with an Alembic migration.
2. Label uploads any of the files the store lacks (§8.1). It then writes the version folder with the code
   that writes reference packages today (`label/refpkg.py`), minus `audio/` and `outputs/`, and publishes
   it with the latest version it knows as `parent`.
3. On a 409, the panel shows the latest version's publisher and date, and it asks whether to publish
   anyway.

The history panel lists the published versions, with links to them in the Dataset Catalog UI. The
`reference_export` table keeps the old sends, so they still show.

### 8.3 New dataset from the Dataset Catalog

The Datasets page gets "New from Dataset Catalog". A person picks a dataset and a version, and Label makes a
working dataset from it:

- The files point at audio in the store, and Label places holds on them.
- The references become confirmed segments, and their uids come from `refs/uids.jsonl` when it is there,
  so flags still match.
- The labels from `refs/attributes.json` go into the files' `meta`.

The working dataset keeps `catalog_dataset_id`, so its next publish becomes version `n + 1` with `n` as its
parent. A version with audio only (§5.2) becomes an `audio_only` dataset.

### 8.4 What doesn't change

Pseudo-labelling, review, live evaluation, lexicons and the output cache work as they do now. Live
evaluation reads the locked score from Bench by the catalog dataset id instead of the Label uid (§9).

## 9. Changes to Concord Bench

### 9.1 Audio at upload

Bench's upload stays as it is. Ingest sends originals to the store and places holds, the same way Label
does (§8.1). Bench's own `paths.original` returns the store path when `DATASET_STORE` is set. Enhanced
copies stay in `data/derived/`.

### 9.2 Import from the Dataset Catalog

"Import references" gets a second source next to the zip upload, called "From Dataset Catalog". A person
picks a dataset and a published version that has references, sees the import preview Bench already shows,
and accepts it.

1. `refpackage.accept` reads the version folder from the mount as the package root. Every file is found by
   sha256 in the store, which the current code already supports for originals Bench has.
2. The new Bench dataset gets `source_key` `dataset-catalog/<dataset id>` and `export_version` `n`. It is
   linked to the Bench dataset made from the previous version, as today.
3. Bench places holds on the version's audio and records a `use` on the version, with its own dataset id
   as `ref`. The holds keep the audio in the store if the version is deleted later (§5.3), and the `use`
   lets the catalog show where a version is used.

A zip from a person or another tool still imports as today, and Bench offers to publish it to the catalog.

### 9.3 Publish from Bench

A Bench dataset that came from an annotated upload or a zip can be published as version 1 of a new catalog
dataset, or as a new version of an existing one. The second choice closes the gap EVALUATE_PLAN.md §7.1
left open, where a dataset with new references had to be uploaded again as an unrelated dataset. Bench
writes the version folder from its own segments and labels. Datasets made from catalog versions are
published from Label, not from Bench, because the references are corrected in Label.

## 10. Contract version 3

The handoff between the apps changes, so CONTRACTS.md goes to contract version 3 in both apps. The apps
still check each other's version through `/api/config`.

| Contract | Version 2 (today) | Version 3 |
|---|---|---|
| §2 Reference package | Label writes it and pushes it to Bench | Defined as a dataset version (platform `contracts/datasets/`) plus `audio/`. Used for zips from people and other tools, and for downloads. Format version 3 allows audio only (§5.2). |
| §3 Flags file | names the dataset by Label `dataset_uid` and `install_id` | names it by the catalog dataset id and version; Label finds the working dataset by `catalog_dataset_id` |
| §4 Bench endpoints | `fingerprints/missing`, `uploads`, `chunk` and `reference-imports` for pushes; `by-source` scores | the push endpoints are removed from the contract (Bench's own UI still uses uploads); scores are at `GET /api/datasets/by-catalog/{dataset_id}/scores` |
| §5 Label endpoints | `files/by-sha` with `dataset_uid` | the same, with `catalog_dataset_id` |

Datasets in Bench that came from Label before the change keep their old `source_key`, so their links back
to Label stop working. They aren't migrated, because the Label datasets they came from can be published to
the catalog and imported again.

## 11. Milestones

| # | Repo | Work | Needs | Exit | Size |
|---|---|---|---|---|---|
| D0 | all | Commit or set aside the uncommitted work on platform `main` (model cards) and Bench `main` (stopping jobs). Merge Label's `reference-package-v2` into Label `main` (4 files conflict: `AGENTS.md`, `Justfile`, `contracts/check.py`, `docs/CONTRACTS.md`). Create the `dataset-catalog` branch in each repository from its `main`. | none | All three `main` branches are clean, and Label `main` speaks contract version 2 | 0.5 to 1 d |
| D1 | Platform | The name "Model Catalog" in the UI and docs, the tab switch with an empty Dataset Catalog page, and the two lines of MODEL_CATALOG_REDESIGN.md (§1, §3, §6) | D0 | Existing UI tests pass; `/` and `/models/:id` still open the same pages | 1 d |
| D2 | Platform | The store, `datasets.db`, the API, holds, clean-up, deleting versions and datasets, zip unpacking and ffprobe, the format in `contracts/datasets/`, the `aicore.datasets` client and fake, the compose mount and `AI_DATA` (§4, §5, §6.1, §7) | D1 | Unit tests for every rule in §4.3 and §5.3; the fake and the real client pass the same tests | 6 to 8 d |
| D3 | Platform | The Dataset Catalog pages, including Upload audio (§6) | D2 | UI tests; each page checked in a browser against the fixture datasets; an upload resumes after the connection drops | 6 to 8 d |
| D4 | Bench | Audio in the store, import from the catalog, publish from Bench, and moving existing originals into the store (§9, §12) | D2 | Tests with `FakeDatasetCatalog`; an imported version scores the same as the same zip imported today | 5 to 7 d |
| D5 | Label | Audio in the store, publish, new dataset from the catalog, and moving existing originals into the store (§8, §12) | D2 | Tests with `FakeDatasetCatalog`; a published version opens in a new Label dataset with the same segments and uids | 6 to 8 d |
| D6 | Bench + Label | Contract version 3: flags, links and scores by catalog id, push removed, CONTRACTS.md and samples in both apps, and `just e2e-handoff` going through the catalog service (§10) | D4, D5 | `just check-contracts` in both apps; the Docker round trip passes: upload in Label, publish, import in Bench, flag back to Label | 3 to 4 d |
| D7 | all | Docs: the catalog README, the Dataset Catalog section, both apps' READMEs and docs pages, and this plan's "what was built" section | all | Docs match the build | 1 to 2 d |

The total is about 5 to 7 weeks. D3, D4 and D5 can run side by side once D2 is done, because each one works
against `FakeDatasetCatalog` until the real service is up.

Each repository's work stays on its `dataset-catalog` branch until D6 passes. Contract version 3 needs both
apps at once, so the two app branches are merged into their `main` branches together. The platform branch
can merge earlier, because the Dataset Catalog adds routes and changes none of the Model Catalog's.

## 12. Moving existing audio into the store

Both apps already hold originals in `data/originals/`. A command in each app (`just move-originals`) moves
them into the store:

1. First, it uploads every original through `upload_audio`, which checks the sha256, and it places a hold
   for each one.
2. Second, it renames `data/originals/` to `data/originals.moved/`, so the app reads from the store from
   then on.
3. Third, the person deletes `data/originals.moved/` when they're satisfied. The command never deletes it.

Files with the same bytes in both apps are stored once, so the store ends up smaller than the two folders
together. Until step 3, the disk holds the audio twice.

## 13. Questions settled after the first draft

The owner answered the first draft's open questions on 2026-10-24. Each answer is in §1 and worked into
the section it shapes.

| Question | Answer |
|---|---|
| Can a version be deleted, or only withdrawn? | It can be deleted. The draft's "withdrawn" state is gone (§5.3). |
| Should the Dataset Catalog UI accept raw audio uploads, beyond reference package zips? | Yes (§6.1) |
| Where should the store live by default? | `~/ai-data/datasets` (§4) |
| Should Label's Sync move to the catalog too? | No. Sync is out of scope, and its code doesn't change (§8.1). |
| Should the version format move from the apps into the platform's `contracts/`? | Yes (§5.1) |

## 14. Risks

| Risk | What reduces it |
|---|---|
| The store becomes the only copy of the audio. | One folder to back up (§4); `just move-originals` never deletes the app's copy (§12) |
| The catalog service is down. | The apps keep reading audio and versions from the mount, so existing datasets still work. Only uploads, publishing and holds wait for the service. |
| Clean-up deletes audio an app still needs. | Holds, a 24 hour grace period and a dry run every time (§4.3) |
| A person deletes a version that apps use. | The UI lists the apps' datasets made from it and asks to confirm; those datasets keep their references and their held audio (§5.3) |
| The two apps derive different 16 kHz audio from the same original, so shared outputs don't match. | No change from today. Both apps already rely on deriving the same audio from the same original sha256. |
| Merge conflicts with the model-cards work on platform `main`. | D0 commits that work first. The dataset routes are added to `catalog/app.py` by one call, and the UI changes are mostly in new files. |
| Contract version 3 lands in one app before the other. | Both app branches merge together after D6 (§11). |
