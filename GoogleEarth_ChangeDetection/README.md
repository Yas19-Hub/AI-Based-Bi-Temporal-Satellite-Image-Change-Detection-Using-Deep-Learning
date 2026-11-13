# Google Earth Historical Image Change Detection

Local toolkit for reviewing **100 paired Google Earth Pro screenshots** (2019 and 2024 by
default). It validates filenames, aligns the later image to the earlier reference, measures
pixel/SSIM differences, renders comparison artefacts, exports CSV/Markdown reports, and provides
a Streamlit review dashboard.

> **Important:** outputs measure differences between two screenshots. They are **not confirmed
> geographic, land-cover or radiometric change**. Alignment failures, different dates, imagery
> artefacts, clouds, labels and UI overlays can all create differences. Review every flagged
> tile against the original imagery.

## What you get

- A 200-row expected manifest (100 tiles x 2 years), including explicit `missing_file` rows.
- Validation for unreadable images, unsupported extensions, wrong year folders, duplicate files,
  unexpected dimensions, oversized files and unrecognized filenames.
- ORB feature alignment with ECC fallback, transform sanity checks and an explicitly flagged
  identity fallback.
- Absolute grayscale difference, SSIM map, thresholded change mask, heatmaps, overlay, annotated
  candidate regions and side-by-side comparison panels.
- Reports: `reports/validation_report.md`, `reports/processing_report.md` and CSV exports for
  missing files, dataset issues, metadata gaps, quality flags and per-tile statistics.
- A dashboard with tile selection, live threshold sliders, side-by-side images, change layers,
  statistics, region tables, metadata and downloads.

All originals in `raw_screenshots/` are treated as immutable inputs.

## Requirements

- Windows 10/11 (tested on Windows 11)
- Python 3.12 (tested with 3.12.6)
- Google Earth Pro installed separately for manual screenshot capture
- No Google Earth account/API or network scraping is used by this project

## Setup (Windows PowerShell)

```powershell
cd D:\projects\images\GoogleEarth_ChangeDetection

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m src.cli init
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You can also call the interpreter directly without activating:

```powershell
.\.venv\Scripts\python.exe -m src.cli init
```

`init` creates the folders, `metadata/tile_manifest.csv`, `metadata/pair_summary.csv`,
`metadata/tile_registry.csv`, `metadata/manual_overrides.csv` and `metadata/review_status.csv`.

## Folder layout

```text
GoogleEarth_ChangeDetection/
  raw_screenshots/2019/T001_2019.png ... T100_2019.png
  raw_screenshots/2024/T001_2024.png ... T100_2024.png
  metadata/                  manifest, overrides, registry, review status
  processed/aligned/         reference and aligned target images
  processed/comparisons/     four-panel comparison images
  processed/change_maps/     diff, heatmaps, masks, overlays, annotations
  processed/results_index.json
  processed/tile_stats.csv
  changes_only/              optional: only changed 2019/2024 pairs
  reports/                   Markdown and CSV reports
  logs/app.log
  src/                       application code
  tests/                     synthetic tests
  app.py                     Streamlit dashboard
  config.yaml
```

## Screenshot workflow (manual and safe)

This project deliberately does **not** automate Google Earth Pro's UI, scrape imagery endpoints,
bypass access controls, or use paid APIs. You choose the imagery and capture it yourself.

1. Open Google Earth Pro and navigate to the tile location (for example from a saved placemark or
   `metadata/tile_registry.csv` coordinates).
2. Open **Historical Imagery** and select imagery whose year matches the target folder (2019 or
   2024). Check the actual imagery date shown by Google Earth Pro; do not assume it.
3. Position the camera consistently between years (same place, zoom, tilt and bearing). Hide
   unnecessary UI chrome/placemarks if possible.
4. Save/export the map image with the exact name `T###_YYYY.png`, for example
   `T001_2019.png`, in `raw_screenshots/2019/` or `raw_screenshots/2024/`.
5. Keep the original file unchanged after capture. Do not crop, resize, annotate or re-save it.
6. Record the imagery date you observed. Dates are **never inferred** from filenames, folder names
   or file timestamps:

   ```powershell
   python -m src.cli metadata --tile T001 --year 2019 --imagery-date 2019-06-15
   python -m src.cli metadata --tile T001 --year 2024 --imagery-date 2024-03-02
   ```

   Optional coordinates and capture time:

   ```powershell
   python -m src.cli metadata --tile T001 --year 2019 `
     --imagery-date 2019-06-15 --capture-date 2026-10-21 `
     --latitude 51.5007 --longitude -0.1246 --notes "same camera as 2024"
   ```

7. Repeat for all 100 tiles and both years. Run validation regularly:

   ```powershell
   python -m src.cli validate
   python -m src.cli status
   ```

### Guided capture helpers

List missing slots and print the manual checklist:

```powershell
python -m src.cli capture-plan --year 2019 --missing-only
```

Optional countdown screen capture after **you** have positioned Google Earth Pro:

```powershell
python -m src.cli capture --tile T001 --year 2019 --countdown 10
```

This only calls Pillow `ImageGrab` after a countdown. It does not control Google Earth or select a
date. It saves the entire screen to `raw_screenshots/2019/T001_2019.png` and refuses to overwrite
an existing original unless `--overwrite` is passed. For clean results, prefer Google Earth Pro's
own map-only image export if available; screen capture may include UI overlays and private window
content.

### Save an already-captured 2019/2024 screenshot

If you already exported or snipped an image, save it directly into the correct year folder with
the expected name:

```powershell
# Save a file you already have
python -m src.cli save-screenshot --tile T001 --year 2019 --source "C:\Users\You\Pictures\T001_2019.png"
python -m src.cli save-screenshot --tile T001 --year 2024 --source "C:\Users\You\Pictures\T001_2024.png"

# Save a screenshot currently on the Windows clipboard (Win+Shift+S or Print Screen)
python -m src.cli save-screenshot --tile T001 --year 2019 --clipboard
python -m src.cli save-screenshot --tile T001 --year 2024 --clipboard
```

The command copies the image bytes without editing the pixels, writes
`raw_screenshots/2019/T001_2019.<ext>` (or `.png` from the clipboard), verifies the saved file, and
refuses to replace an existing original unless `--overwrite` is explicitly passed. It does not
infer or record an imagery date. After saving both years, run:

```powershell
python -m src.cli validate
```

### Naming rules

- Expected primary name: `T001_2019.png` and `T001_2024.png`.
- Allowed extensions are configured in `config.yaml` (`.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`,
  `.tiff`, `.bmp`).
- The filename year must match its year folder.
- Tiles outside `T001`-`T100` are reported as unrecognized, not silently dropped.
- Duplicate valid files for one tile/year are all listed; the preferred extension becomes primary
  and the pair is blocked while `skip_ambiguous_duplicates` is true.
- A missing slot is written as `validation_status=missing_file`; it is never filled with fake data.

## Configure

Edit `config.yaml`:

- `project.years`: exactly two years; the **first is the alignment reference**.
- `project.tiles.start` / `end`: tile range, default 1-100.
- `project.expected_dimensions`: optional strict width/height (e.g. `{width: 1920, height: 1080}`).
- `processing.alignment.method`: `auto`, `feature`, `ecc` or `none`.
- `processing.diff`: pixel/SSIM thresholds, combine mode, morphology, minimum region area.
- `processing.skip_existing`: reuse existing outputs unless `process --force`.
- `dashboard`: default thresholds and display width.

If you change years or tile IDs, re-run `python -m src.cli init` and `validate`; old derived
outputs may no longer correspond to the configured pairs.

## Run the pipeline

```powershell
# 1. Validate and write the manifest/pair summary + validation report
python -m src.cli validate

# 2. Align and compare every processable pair
python -m src.cli process

# Recompute selected tiles regardless of cached outputs
python -m src.cli process --tile T001 --tile T002 --force

# 3. Regenerate Markdown/CSV reports from the current manifest and results index
python -m src.cli report

# 4. Copy only pairs whose measured change is greater than 10% of valid overlap
python -m src.cli export-changes --threshold 10

# 5. Check completion
python -m src.cli status
```

Run everything through the interpreter without activation if preferred:

```powershell
.\.venv\Scripts\python.exe -m src.cli validate
.\.venv\Scripts\python.exe -m src.cli process
.\.venv\Scripts\python.exe -m src.cli report
```

## Processing details

1. The **2019** image is the reference; the **2024** image is the target.
2. Mismatched dimensions are resized by default (`processing.resize_mismatched`).
3. Alignment tries ORB + RANSAC when configured, then ECC Euclidean/Affine fallback. A transform is
   rejected for excessive scale, rotation, perspective, low inliers/coverage or low overlap.
4. If no reliable transform exists, comparison falls back to identity and the result is explicitly
   flagged `alignment_unreliable`. It never claims successful alignment.
5. Differences are computed in grayscale. `combine_mode=and` requires both the pixel and SSIM
   thresholds; `or` accepts either. Regions smaller than `min_region_area` are removed.
6. Every written PNG is reopened and verified for readability and expected dimensions. A tile is
   only marked `processed` after all required outputs pass verification.

Outputs per tile `T001`:

- `processed/aligned/2019/T001.png` and `processed/aligned/2024/T001.png`
- `processed/comparisons/T001_comparison.png` (side-by-side + heatmap + annotated)
- `processed/change_maps/T001_diff.png`
- `processed/change_maps/T001_diff_heatmap.png`
- `processed/change_maps/T001_ssim.png`
- `processed/change_maps/T001_ssim_heatmap.png`
- `processed/change_maps/T001_change_mask.png`
- `processed/change_maps/T001_valid_mask.png`
- `processed/change_maps/T001_change_overlay.png`
- `processed/change_maps/T001_annotated.png`

Results are indexed in `processed/results_index.json` and flattened in
`processed/tile_stats.csv`.

## Keep only changed 2019/2024 pairs locally

After processing, export a separate folder containing only the original screenshot pairs whose
measured difference exceeds a threshold:

```powershell
python -m src.cli export-changes --threshold 10
```

This copies both original-year images into:

```text
changes_only/
  2019/T001_2019.png
  2024/T001_2024.png
  changes.csv       exported tiles with change %, SSIM, alignment and source paths
  skipped.csv       every other tile with the exact skip reason
  README.txt        threshold and interpretation limits
```

Defaults and safety rules:

- A pair is copied only when `changed_pct_of_valid` is **greater than** the threshold (10 means
  more than 10% of valid overlap pixels).
- Pairs with unreliable alignment are excluded unless `--include-unreliable` is passed.
- Pairs with less than 50% valid overlap are excluded; change with `--min-valid-overlap`.
- Originals in `raw_screenshots/` are never moved or deleted. Only a copy is written.
- Every copied file is SHA-256 verified against its source.
- Re-running with a stricter threshold removes previously exported copies that no longer qualify
  (tracked in `changes_only/.export_manifest.json`). Use `--keep-stale` to keep them, or
  `--output some_folder` to write elsewhere.
- Use `--force` only if an exported copy differs from the original and you intend to replace it.

If no processing results exist yet, the command explains that screenshots must be added and
`validate`/`process` must be run first. It does not invent or fetch Google Earth imagery.

## Dashboard

```powershell
python -m src.cli dashboard
# or:
streamlit run app.py
```

Pages:

- **Overview** - screenshot/pair completion, processed counts, status charts, dataset issues.
- **Tile explorer** - select a tile, view reference/target side by side, adjust pixel threshold,
  SSIM threshold, combine mode and minimum region size live, inspect heatmaps/masks/overlays,
  regions, statistics and metadata, and download current outputs.
- **Changed pairs** - preview tiles above a 10% (adjustable) measured-change threshold and copy
  only their 2019/2024 originals to `changes_only/`.
- **Data quality** - issues, missing slots, unrecognized files, metadata gaps, processed quality
  flags, and regenerated report downloads.
- **Downloads** - manifest, pair summary, tile statistics, results index and report exports.

Use **Refresh data** in the sidebar after changing screenshots or metadata. The dashboard reads the
manifest/results index and does not invent values for missing dates or files.

## Metadata and review

Metadata precedence (highest first):

1. `metadata/manual_overrides.csv` (use the CLI rather than editing by hand when possible)
2. Existing manifest values for the same tile/year/filename
3. Unambiguous values preserved for the slot
4. Coordinates from `metadata/tile_registry.csv` when the manifest has none

Record manual review outcomes:

```powershell
python -m src.cli review --tile T001 --status reviewed --reviewer "Your Name" --notes "same camera"
python -m src.cli review --tile T002 --status unreliable --notes "cloud cover differs"
```

Valid imagery dates are `YYYY`, `YYYY-MM` or `YYYY-MM-DD`. Invalid calendar dates are rejected at
entry. A date whose year differs from the folder year is flagged for review, not silently accepted.

## Tests

```powershell
pytest
# or
.\.venv\Scripts\python.exe -m pytest -q
```

The suite uses only synthetic `tmp_path` images and never writes into the real dataset. It covers
pairing/missing files, invalid/duplicate/unrecognized files, expected dimensions, metadata
persistence and validation, transform sanity checks, alignment identity fallback, difference/SSIM
statistics, region/mask generation, processing outputs, reports and capture safeguards.

## Troubleshooting

| Symptom | Check |
|---|---|
| `0 valid` after `validate` | Filenames must be `T###_YYYY.ext`; file must be in the matching `raw_screenshots/YYYY/` folder and decode successfully. |
| Unrecognized file reported | Fix the filename case/pattern/extension or move it to the correct year folder. Unsupported files are never processed. |
| Duplicate file blocks a tile | Keep exactly one screenshot per tile/year; remove or archive extras outside the raw folders. |
| `pair not processable` | One year is missing/invalid, duplicates exist, or dimensions mismatch while resizing is disabled. Read `metadata/pair_summary.csv`. |
| Alignment unreliable | Verify the same place/zoom/bearing; large seasonal change, clouds, labels, rotation or low texture can defeat ORB/ECC. Try `method: ecc`, a smaller `max_rotation_deg`, or accept the flagged identity comparison only as a visual aid. |
| Very high changed % | Often misalignment, different imagery dates, brightness/seasonal changes or UI overlays. Inspect the comparison panel before interpreting. |
| Images resize unexpectedly | `processing.resize_mismatched: true` resizes the target to the reference. Set `expected_dimensions` and require consistent captures for strict checks. |
| Dashboard shows old data | Click **Refresh data**; re-run `validate`/`process` after changing files or metadata. |
| Outputs not updated after threshold changes in dashboard | Dashboard sliders are live previews. To persist new thresholds, edit `config.yaml` and run `process --force`. |
| `process` skips tiles | Existing outputs are reused by default. Use `--force` to recompute. |
| PowerShell activation blocked | Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or call `.\.venv\Scripts\python.exe` directly. |
| Streamlit port in use | `python -m src.cli dashboard --port 8502` |

## Limitations

- Google Earth Pro's historical imagery dates and positions must be verified by the user. The tool
  cannot tell whether two captures show the same camera, season or processing.
- Image differences are not change classification. Shadows, clouds, haze, labels, anti-aliasing,
  compression, UI and rendering differences can dominate the signal.
- SSIM and pixel thresholds are sensitive to contrast, resolution and resampling. Results are
  comparable within a consistent capture workflow, not across arbitrary datasets.
- Alignment assumes predominantly planar/overhead imagery and limited rotation/scale. It can fail on
  sparse, repetitive or heavily changed scenes; failures are flagged rather than hidden.
- No NDVI, radiometric calibration or land-cover claims are made. Screenshots are display-referred
  RGB images, not calibrated multispectral data.
- No Google Earth automation, scraping, access-control bypass, paid API or fabricated data is used.
- This project is for local analysis and human review, not a substitute for authoritative
  geospatial change products.
