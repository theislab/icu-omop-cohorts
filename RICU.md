# The cohorts and ricu

YAIB builds its cohorts with [ricu](https://github.com/eth-mds/ricu) and the task scripts and ricu extensions of
[YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts). The cohort builder of this repository computes the same
cohorts from an OMOP CDM: the base cohort and the exclusions of each task, YAIB's 48 hourly variables with ricu's units
and plausible ranges, and the labels. It reproduces ricu's behaviour, as we studied it in ricu's documentation, source
and outputs, in Python and SQL of its own; it contains no ricu code.

This page lists where the builder follows a choice of ricu that it would otherwise make differently, where it
deliberately differs, and the errors in ricu's definitions we found on the way. Every place where the code follows
ricu is marked `RICU CHOICE`: `grep -rn "RICU CHOICE" cohorts omop_etl/src`.

## Where we follow ricu

Following ricu keeps the cohorts and labels comparable with YAIB's. In each row, our reading would be more faithful to
the source data.

| what | ricu | our reading | code |
|---|---|---|---|
| eICU ionized calcium recorded in mg/dL or mEq/L (by its unit string; 5.6 % of the rows) | read as mmol/L, so the 0.5-2 mmol/L bounds drop it | 4.008 mg/dL and 2 mEq/L per mmol/L; `CAI_CONVERT=1` builds eICU frames that convert it | `cohorts/yaib_omop_crosswalk.py` (`CAI_AS_RICU`) |
| MIMIC-IV labevents and microbiologyevents without `hadm_id` | left out | a row inside an ICU stay of the same subject belongs to that stay | `omop_etl/src/omop_etl/mimic4.py` (`assign_to_stay`) |
| HiRID antibiotics | Salofalk and Topamax are in `abx_duration` | neither is an antibiotic | `cohorts/abx_sets.py` |
| eICU antibiotic durations | a stop before the start gives a negative duration, which shortens the course | duration 0 | `label_inputs.infection_tables` |
| KDIGO urine | window sums compared with per-hour thresholds; a stay without a weight keeps mL | the mean rate over the window, per kg | `build_cohort.score_kdigo` (`ep.tl.kdigo` with `KDIGO_URINE_CRITERIA`, a weight of 1 kg) |
| SOFA windows of a day | `[t - 24, t]`, 25 hourly rows, for the worst score of an organ and for the urine output, which is still extrapolated to 24 h | the 24 hours that end at the hour (`ep.tl.sofa`'s default) | `build_cohort.score_sofa` (`closed="both"`) |
| KDIGO thresholds | compared exactly in floating point, so a rise from 1.1 to 1.4 mg/dL (0.29999999999999982) is no rise of 0.3, and 1.2 is less than 1.5 x 0.8 (1.2000000000000002) | a value within 1e-9 of a threshold reaches it (`ep.tl.kdigo`'s default) | `build_cohort.score_kdigo` (`KDIGO_TOLERANCE = 0`) |
| `urine_rate` (KDIGO's urine input) | each row's volume divided by the hours since the stay's previous row plus 1, so hourly rows are halved and the first ICU row is divided by the gap since the last row before admission; an hour takes its largest rate | the volume of the hour, the sum of its rows | `label_inputs.ricu_urine_rate` |
| MIMIC-IV cultures | a culture goes to the stay of its admission by its chartdate, so one charted after a discharge on that day belongs to the earlier stay | by its charttime | `omop_etl/src/omop_etl/mimic4.py` (`specimen_sql`) |
| vasopressor rows at rate 0 | a row at rate 0 starts the SOFA rows (scored 0), which can set the baseline of the Sepsis-3 rise an hour earlier | a rate of 0 is no input; SOFA starts at the first organ input | `build_cohort.add_sofa_inputs`, `label_inputs.vasopressors` |
| administration ranges of vasopressors | widened by 5 min, each grouped with the first range it overlaps, narrowed again; a chain of overlaps can stay split in two | overlapping ranges merged into one | `label_inputs.merge_ranges` |
| HiRID SOFA renal and the base exclusions | urine is de-cumulated fluid balance | urine output | `build_cohort.add_sofa_inputs`, `label_inputs.hirid_fluid_urine` |
| eICU ventilation start without an end | lasts start + 6 h | lasts 6 h | `label_inputs.eicu_vent_windows` |
| windows that end before ICU admission | the end is moved to admission | the window stays before the stay | `label_inputs.window_hours` |
| MIMIC-IV `resp` | the union of the measured rates and 224688 'Respiratory Rate (Set)', the ventilator's set rate, so an hour's median mixes both | the measured rate | `cohorts/yaib_omop_crosswalk.py` (`VAR_CONCEPTS_BY_DATASET`) |

## Where we differ from ricu

- **HiRID `urine`.** ricu's is the de-cumulated fluid balance (variable 30005110), intake minus output. Our CDM and the
  frames carry the hourly urine output (10020000). `RICU_HIRID_URINE=1` builds ricu's variant for comparison.
- **MIMIC-IV urine.** ricu sums 12 outputevents items, among them 227510 'TF Residual', a tube-feed residual, and 226566
  'Urine and GU Irrigant Out', which includes irrigant. We sum the other 10 per hour.
- **MIMIC-IV oxygen saturation.** ricu's items include 226253 'SpO2 Desat Limit', an alarm setting. We take the pulse
  oximetry 220277 first and the blood gas 50817 where it is missing.
- **eICU calcium and magnesium.** Converted by the unit each row records. ricu converts magnesium in mEq/L to mg/dL by
  dividing by 1.215 where it should multiply.
- **HiRID creatinine.** µmol/L divided by 88.4, where ricu multiplies by its rounded 0.011312.
- **HiRID lymphocytes.** Kept as recorded. ricu applies a ratio to the white cell count that gives implausible values.
- **Not extracted.** MIMIC-IV prescriptions, from which ricu can read antibiotics, and eICU's microLab, from which it
  reads samples.

## Errors in ricu's definitions

Found while matching ricu. We never copy them into the CDM. Where label parity needs them, the builder or the mapping
layer reproduces them, as the last column says. The files are those of ricu (`inst/extdata/config/concept-dict.json`,
`R/`) and of YAIB-cohorts (`ricu-extensions/`).

| concept | ricu | why it is wrong | our handling |
|---|---|---|---|
| `mg` (eICU) | mEq/L rows divided by 1.215 on the way to mg/dL (concept-dict.json) | 1 mEq/L = 0.5 mmol/L = 1.215 mg/dL, so the factor multiplies; those rows end up 1.48x too low | the CDM converts mEq/L to mmol/L (x 0.5) |
| `o2sat` (MIMIC-IV) | items 220277, 226253, 50817 (concept-dict.json) | 226253 is 'SpO2 Desat Limit', an alarm setting; 50817 is a labevents item listed under chartevents and matches nothing | SpO2 220277 and lab 50817 |
| `urine` (MIMIC-IV) | includes 227510 and 226566 (concept-dict.json) | 227510 is 'TF Residual', tube-feed residual; 226566 'Urine and GU Irrigant Out' includes irrigant | the ten urine-only items |
| `lymph` (HiRID) | a ratio to the white cell count, 100 x / wbc, on 24000480 (concept-dict.json) | right for an absolute count, but ricu's output has a mean of 0.15 where a lymphocyte percentage is around 10-30 | no ratio |
| `abx_duration` (HiRID) | pharma items 1000426 Salofalk (mesalazine) and 1000760 Topamax (topiramate) (ricu-extensions, medications) | neither is an antibiotic | copied in `cohorts/abx_sets.py`, for label parity |
| `temp` (MIMIC-IV) | includes 224027 (concept-dict.json) | 'Skin Temperature', without a unit, from a skin assessment | not used |
| `kdigo_urine` | the urine rates over 6, 12 and 24 h are the sum of the hourly rates divided by the weight, compared with the per-hour thresholds 0.5 and 0.3 mL/kg/h (ricu-extensions, callback-kdigo.R) | KDIGO means a mean rate below the threshold for the whole window; a sum over 6 h is 6 times the mean, so oliguria is detected far too rarely | copied in the builder, for label parity |
| `kdigo_urine` | only stays with a weight row are divided by their weight (callback-kdigo.R) | a stay without a weight keeps its urine in mL, not mL/kg, and practically never stages; the fallback weight of 75 kg is never used | copied in the builder |
| `urine_rate` | the time since the previous row plus 1 h as the divisor (callback-kdigo.R) | consecutive hourly rows get a 2 h divisor, which halves the hourly volume | copied in the builder |
| `sofa_renal` (HiRID) | the 24 h urine from `urine`, which on HiRID is de-cumulated fluid balance (concept-dict.json) | fluid balance is intake minus output, not urine | copied for SOFA renal and the base exclusions; the hourly `urine` variable is urine output |
| `urine` (eICU) | the intakeoutput pattern `catheter.+output\|output.+catheter` (concept-dict.json) | it also matches the output of a ventricular drainage catheter, which is CSF (756 rows) | left out of urine in the CDM |
| `abx_duration` (eICU) | medication duration as stop minus start offset, negative for 13,004 antibiotic orders whose stop precedes the start (often a stop of 0) | a negative duration shortens the course in `abx_cont`, so such orders count as less coverage than an order without a stop | the CDM ends such an order at its start and keeps the source stop in `sig`; the builder copies the negative duration for label parity |
| `vent_ind` (eICU) | a start without an end within the match window gets the start time plus 6 h as its duration (R/callback-cncpt.R) | the duration should be the 6 h; a ventilation start at 50 h lasts until 106 h, and an unmatched start more than 5.5 h before admission gets a negative duration and is dropped. 2,708,400 of ricu's 2,779,162 eICU windows (97 %) have this duration. MIMIC-IV and HiRID use ventilation windows and are not affected | copied in the builder (`label_inputs.eicu_vent_windows`), for label parity |
| `expand` | an end before time 0 is set to 0 (R/utils-ts.R) | a window that ended before ICU admission is stretched to admission instead of staying before it; this affects the ventilation in SOFA respiration and the sedation windows of the GCS | copied in the builder (`label_inputs.window_hours`) for ventilation and HiRID sedation |
