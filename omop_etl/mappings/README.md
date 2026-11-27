# mappings

Shareable mapping files, joined in SQL by the dataset modules and the cdm step. Nothing here is
copied from the Athena vocabulary beyond concept-id integers; names come from the dataset
documentation, ricu's public concept dictionary (eth-mds/ricu, GPL-3) or are generic drug names.
The vocabulary the user downloads is read at run time (`vocabulary_dir` in the config).

| file | rows | content and provenance |
|---|---|---|
| `mimic4/measurements.csv` | one per MIMIC-IV itemid that becomes a MEASUREMENT row | `source_table, source_code` (itemid, written to `measurement_source_value`), `source_label` (MIMIC-IV `d_items`/`d_labitems` label, open demo), `variable` (BlendedICU name), `concept_id`, `unit_concept_id` (the UCUM unit the item is recorded in; values are stored as recorded), `rule` (`fahrenheit_above` and `unit_from_string`, see `units.csv` below, `text` = keep the chartevents text in `value_source_value`, `height_weight` = every charted height/weight). Items are ricu's itemids for the YAIB concepts. Built once from BlendedICU's `timeseries_variables.csv` (see `../../NOTICE`) |
| `mimic4/drug_units.csv` | one per inputevents `amountuom` that is converted | `source_unit, factor, unit`: dose into mg / mL / mEq (UCUM `10*-3.eq`) |
| `eicu/measurements.csv` | one per eICU source item that becomes a MEASUREMENT row | see below |
| `hirid/measurements.csv` | one per HiRID variableid | see `hirid/README.md` |
| `drug_ingredients.csv` | one per ingredient | `ingredient` (generic name), `concept_id` (RxNorm Ingredient), `pattern`: case-insensitive regex (RE2) of generic and brand names, matched against `drug_source_value` of every dataset; the first matching row in file order sets `drug_concept_id`, so combination products and specific insulins come first. Ingredients of BlendedICU's `medications_v11.json` with its ids, plus common ICU drugs whose ids were looked up once in an Athena download (see `../../NOTICE`). The consumer selects drugs by source name, never by this concept |
| `units.csv` | one per unit string | `unit_string` (lower case, blanks removed), `unit_concept_id`, `unit_name`: the unit of a row whose mapping row has rule `unit_from_string`, from the row's unit string; a string not listed, or none, gives the mapping row's unit. Rule `fahrenheit_above`: a temperature above 50 is in °F (`src/omop_etl/units.py`) |
| `conditions.csv` | one per ICD code with a concept | `icd_version, icd_code, group, concept_id`, from BlendedICU's `diagnoses.json` (essential hypertension, influenza); every other diagnosis gets concept 0 |

`eicu/measurements.csv`: one row per eICU source item that becomes a MEASUREMENT row
(`source_code` = `measurement_source_value`, `<table>:<label>`). Items from ricu's concept dictionary
(eth-mds/ricu, GPL-3) and the ricu-extensions temperature override; labels from the eICU-CRD
documentation; concept ids integers only. `unit_concept_id` is the unit the item is recorded in.
`rule` names a per-row rule: the unit from the row's unit string (`unit_from_string`: calcium,
magnesium, ionized calcium, CRP), a temperature above 50 in °F (`fahrenheit_above`), lab FiO2 in mmHg
left out, catheter-output cells as urine.

The concepts are those of OHDSI's MIMIC-IV ETL (github.com/OHDSI/MIMIC), LOINC: a MIMIC-IV item has its
concept there, and an eICU or HiRID item the concept of the MIMIC-IV item of its variable (since 2026-11-26,
`docs/OHDSI-MIMIC.md`). Concepts of the labs match the unit the item is recorded in: the LOINC
"[Mass/volume]" concept for mg/dL (MIMIC-IV, eICU), "[Moles/volume]" for SI values (HiRID). An item whose
rows come in several units (eICU calcium and magnesium) has the concept of its main unit.

## Source notes

Checked on the sources during the 2026-11 review; aggregate observations only.

- MIMIC-IV labevents albumin is mostly 50862; 53085 (~70 rows) claims mg/dL but its values are g/dL,
  as the mapping records them.
- MIMIC-IV records bicarbonate (50882), chloride (50902), potassium (50971) and sodium (50983) in
  mEq/L, which equals mmol/L for these monovalent ions; the mapping writes mmol/L.
- MIMIC-IV MCHC (51249) is recorded half as g/dL and half as %, with the same numbers.
- MIMIC-IV CRP (50889) is in mg/L; about 60 rows have no unit and a mean about twice as high. They
  are kept as mg/L, as ricu reads them.
- MIMIC-IV chartevents 223762 (temperature, °C) has about 800 values of 50 or more; they are read as
  °F.
- MIMIC-IV `diagnoses_icd` has no row without `hadm_id`; labevents and microbiologyevents do, and
  those rows are left out, as ricu does.
- MIMIC-IV transfers: every admission ends with a 'discharge' row whose careunit is UNKNOWN; the
  last ward is the row before it. In `icustays` of 3.1, `first_careunit` equals `last_careunit` on
  every row.
- MIMIC-IV ICU death (`stays_sql`): spot checks found stays transferred to another ward before the
  death correctly not counted as ICU deaths.
- eICU lab: 229,518 of 39.1 M rows (0.6 %) have no numeric result and are left out (logged by the
  stage step; the 2026-11 review had estimated 0.04 %).
- eICU lab units: the per-row unit is `labmeasurenameinterface`, the column ricu reads;
  `labmeasurenamesystem` says mg/dL for every calcium row. Calcium is mg/dL apart from ~115 rows in
  mmol/L. Magnesium is mg/dL apart from ~13,400 rows in mEq/L, whose values fit mEq/L. Ionized
  calcium labelled mmol/L has a median of 4.4, i.e. mostly mg/dL values under a wrong label; ricu
  reads them as recorded and its 0.5-2 mmol/L bounds drop them.
- eICU intakeoutput: `cellvaluenumeric` is never NULL. Most urine rows are labelled 'Urine'; ~1 %
  are catheter outputs ('Output (mL)-Straight Catheter' and similar), mapped to urine as ricu does.
- eICU medication repeats rows exactly; ~100k of 13 M collapse.
- eICU nursecharting has rows with `nursingchartcelltypevalname` 'Value' whose label says what they
  are ('MAP (mmHg)', 'Arterial Line MAP (mmHg)', 'Eye Opening', 'Motor Response', 'Verbal
  Response'). ricu does not read them and neither does the mapping.
