# OHDSI/MIMIC conventions compared with omop_etl (2026-11-04; their concepts adopted 2026-11-26)

Source: https://github.com/OHDSI/MIMIC at `d209ed3` (2025-10-20), CDM 5.3.1, BigQuery SQL, last run on
MIMIC-IV 2.2. Files: `etl/etl/*.sql`, `custom_mapping_csv/*.csv`, `crosswalk_csv/d_items_to_concept.csv`.
Concept names were looked up on the public OHDSI WebAPI. Since 2026-11-04 our CDM keeps source values
and units as they do; it converted to ricu/SI units before. Since 2026-11-26 our measurement
concepts are theirs: every MIMIC-IV itemid they map has their concept, and eICU and HiRID take the same
concept per variable (see the last sections). Before, 18 of our 87 MIMIC-IV itemids matched theirs; we
used BlendedICU's concepts, mostly SNOMED.

## Their conventions

- **Visits.**
  - VISIT_OCCURRENCE is a hospital admission (`visit_source_value = '<subject_id>|<hadm_id>'`, type 32817).
  - VISIT_DETAIL has one row per `transfers` row; ICU care units are care sites. `icustays` is not used, and
    event rows leave `visit_detail_id` NULL.
  - A lab without `hadm_id` goes to the admission whose window covers it, otherwise to a synthetic one-day
    visit.
- **Dates:** MIMIC's shifted dates, unchanged.
- **MEASUREMENT.**
  - Each itemid has a custom source concept (2-billion range, `concept_code` = itemid) with a "Maps to" link
    to a standard concept. chartevents: 169 items. labevents: 1,402 items to LOINC. outputevents: 68 items.
    There is no SOURCE_TO_CONCEPT_MAP.
  - `measurement_source_value` is the itemid, `value_source_value` the raw `value` text (always filled), and
    `unit_source_value` is `valueuom`.
  - Units are not converted. `unit_concept_id` is mapped from the source unit string (0 if unmapped).
    Exceptions:
    - temperatures outside 25-44 °C are dropped, and °F stays °F;
    - outputevents 227488 is negated;
    - "7.5 mm"-style values are parsed;
    - lab values are parsed from `value`, with `operator_concept_id` and `range_low`/`range_high`.
  - Type concepts: labs 32856, the rest 32817.
- **Unmapped items:** labs and outputevents keep concept 0; unmapped chartevents items are dropped.
- **PERSON.**
  - `year_of_birth = anchor_year - anchor_age`.
  - Gender F 8532, M 8507, otherwise 0.
  - Race from the first admission (an 8-string map, other strings 0).
- **DEATH:** first `admissions.deathtime`, capped at `dischtime`.
- **Drugs.**
  - Only `prescriptions`; inputevents are not loaded.
  - NDC is mapped via "Maps to" to RxNorm.
  - Route from a 96-string map; type 32838.
- **Other.**
  - Random ids (FARM_FINGERPRINT of a UUID).
  - One CDM_SOURCE row.
  - OBSERVATION_PERIOD type 32828.
  - Conditions: ICD with the dots removed, then "Maps to"; type 32821.

Defects in theirs, not to copy:
- the outputevents source-concept ids collide with the lab ids;
- outputevents `unit_concept_id` is NULL in the final table;
- 'bpm' maps to 8483 (counts per minute);
- `preceding_visit_detail_id` is LAG without PARTITION;
- the ethnicity map predates the MIMIC-IV 2.0 race strings.

## Adoptable at no cost

The cohorts read none of these columns.

1. Standard type concepts:
   - VISIT_OCCURRENCE type 44818518 -> 32817;
   - age OBSERVATION type 38000280 -> 32817;
   - DRUG_EXPOSURE 43542358 -> 32818 (EHR administration record). Keep 32865 for eICU admission drugs,
     which the consumer tests.
2. `measurement_type_concept_id` 32856 for lab rows, 32817 otherwise.
3. `value_source_value` filled with the verbatim source value on every row.
4. `unit_concept_id` describes `value_as_number`. Done on 2026-11-04: urine 8587 (mL), WBC and
   platelets 8848, MCHC 8713; values are stored as recorded.
   - Their `gcpt_meas_unit.csv` (Apache-2.0) maps source unit strings to UCUM, except bpm -> 8483.
5. `measurement_source_concept_id` and text-only `value_as_concept_id` 0 instead of NULL.
6. PERSON: unknown gender 0; optional `race_source_value`.
7. CARE_SITE `place_of_service_concept_id` for MIMIC from their care-unit map (`gcpt_vis_admission.csv`).
8. VISIT_OCCURRENCE admitted-from / discharged-to concepts from their location maps. 'DIED' must not be
   mapped there, since the ICU-death flag follows ricu.
9. OBSERVATION_PERIOD type 32828.
10. One CDM_SOURCE row (`cdm_version` 5.4, concept 756265).
11. `route_concept_id` 4171047 (IV) for MIMIC inputevents.
12. CONDITION_OCCURRENCE type 32821 and `condition_source_concept_id` from ICD9CM/ICD10CM codes.

## Costly

| convention | cost |
|---|---|
| Their LOINC measurement concepts | Done on 2026-11-26. HiRID's labs in SI units keep the "[Moles/volume]" LOINC codes, which the crosswalk lists next to the "[Mass/volume]" ones since iter5. |
| No unit conversion in the ETL | Done on 2026-11-04: the consumer converts (`UNIT_CONVERSIONS`, applied by ehrdata). |
| Admission = VISIT_OCCURRENCE, ICU stay = VISIT_DETAIL | The cohorts and ehrdata's `setup_obs` are built on visit = ICU stay; HiRID has no admission level. Not now (decision 2026-11-04). |
| Real shifted dates | Done on 2026-11-04 for MIMIC-IV and HiRID; eICU, which has offsets only, is placed in its discharge year. |
| Random ids, their source_value formats, their DEATH, value parsing from text | Change what the consumer reads, or lose determinism. |

## MIMIC-IV itemids: ours vs theirs (since 2026-11-26)

Every itemid we map has their concept, with these exceptions:

| itemid(s) | label | ours | why |
|---|---|---|---|
| 53085 / 53086 / 53089 / 53154 / 224639 | Albumin / ALP / Bilirubin / Lactate / Daily weight | the concept of 50862 / 50863 / 50885 / 50813 / 226512 | not in their files, which are based on MIMIC-IV 2.2 |
| derived from 220739, 223901, 223900 | GCS total | 3007194 (LOINC 9269-2) | they store the components only |
| 224685, 224684 | Tidal volume observed, set | 21490854 for both, as theirs | the cohorts read neither |

The concepts of the other datasets follow: an eICU or HiRID row takes the concept of the MIMIC-IV item of its variable.
Where MIMIC-IV has two (respiratory rate total and spontaneous, glucose from the blood gas and from chemistry), they
take the one of the variable as a whole (3024171, the total respiratory rate) or keep theirs (glucose). Deliberately
different: HiRID's labs in SI units ("[Moles/volume]": creatinine, urea, calcium, magnesium, phosphate, bilirubin,
glucose), HiRID's absolute neutrophil count (3013650), and eICU's charted GCS total.

The change relabels rows only. For every variable, static and SOFA input, and for the prefilter, the old and the new
concepts select the same source items in all three datasets (checked against the mappings), and the MIMIC-IV demo CDM
equals the BlendedICU ETL's under the old concepts (`tests/test_mimic4_demo.py`). The concepts before 2026-11-26 and
the reasons they differed (vitals in SNOMED, labs by BlendedICU's unit or by another specimen) are in this file's
history.

## Units and multiple stays in their design

- They store source units and choose the concept to match. Our CDM stores source values and units too
  (since 2026-11-04), but keeps one concept per variable across datasets, so a concept's rows can be in
  different units; the consumer converts them.
- Several ICU stays of one admission share one VISIT_OCCURRENCE in their CDM and appear only as VISIT_DETAIL
  transfer rows, with no event linked to them. ICU-stay data from their CDM would need a time join to those
  rows.
