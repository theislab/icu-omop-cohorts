# CDM conventions, and differences from the BlendedICU CDM

This ETL started as a reproduction, value for value, of the CDM that BlendedICU's ETL writes. Since then it follows the
conventions below. Where a convention moved the CDM away from BlendedICU's, the second section says how. The cohorts
built from it are compared with ricu's (`../../RICU.md`).

## Conventions

- **Timeline.** Every stay sits at its admission on the source's own clock: MIMIC-IV's and
  HiRID's date-shifted times. eICU has offsets only: a person's hospital stays, in order of
  discharge year, age and a fatal stay last, start on 1 January of their discharge year at their
  admission clock time, each not before the calendar day after the previous discharge (`eicu.py`),
  which can push a stay past its year (fewer than 10 of 166,355 on eICU 2.0). A stay's rows sit at admission +
  offset, and rows before admission and after discharge are kept.
- **Year of birth.** MIMIC-IV `anchor_year - anchor_age`; eICU discharge year minus age; HiRID
  admission year minus age. The age at a stay is its OBSERVATION row (concept 44804452), and PERSON
  holds the year of birth of the person's first stay.
- **Values and units.** MEASUREMENT values are stored as recorded (DOUBLE), and
  `unit_concept_id` is the UCUM unit they were recorded in. Where the unit varies from row to row,
  it is read from the row's unit string (`mappings/units.csv`): eICU calcium, magnesium, ionized
  calcium and CRP, MIMIC-IV CRP; a row without a known string takes the item's unit. Temperatures go
  by value: above 50 a value is °F, else the item's unit (MIMIC-IV, both items; eICU's Celsius items);
  eICU's "Temperature (F)" is °F. ricu reads every eICU ionized calcium row as mmol/L whatever its
  unit, and so do the cohorts by default (`../../RICU.md`, "Where we follow ricu").
  Lab concepts match the recorded unit: LOINC "[Mass/volume]" for mg/dL (MIMIC-IV, eICU),
  "[Moles/volume]" for SI values (HiRID).
  ricu instead converts every row of MIMIC-IV's Fahrenheit item 223761 and so drops a charted 37
  there as 2.8 °C; we keep it as 37 °C.
  Converting units is the consumer's job (`cohorts/yaib_omop_crosswalk.py`,
  `UNIT_CONVERSIONS`).
- **Data fixes kept in the ETL.**
  - a lab FiO2 in mmHg is left out (ricu sets it to NA);
  - the GCS total is derived where eye, motor and verbal are recorded together.
- **Derived admission height and weight.** The averaged rows (`raw_height`, `raw_weight`, at
  admission + 5 h) combine items charted in different units, so they are in cm and kg. The items
  themselves are staged in their own units.
- **SPECIMEN** has no visit id in CDM v5.4. FACT_RELATIONSHIP links each specimen to the visit of
  its stay ('Has associated visit' 33136, and 33137 back).
- **Text and numbers.** A text value is a number where it reads as one after trimming, else it
  stays text in `value_source_value`. Fields that mean missing ('', 'NA', 'None', ...) are NULL.
- **Ids.** Visits and persons get md5 ids of `<dataset>-<source id>`, so they are the same in every
  run. The surrogate ids of the event tables are row numbers.

## Differences from the BlendedICU CDM

| table.column | BlendedICU | now |
|---|---|---|
| all datetimes | synthetic timeline from 2000-01-01, a person's stays at least 60 days apart | each source's own clock (eICU placed in its discharge year) |
| PERSON.year_of_birth | 2000 - age | real year minus age (MIMIC-IV anchor_year - anchor_age) |
| MEASUREMENT.value_as_number | float32, converted into BlendedICU's SI units (creatinine µmol/L, glucose mmol/L, ...) | DOUBLE, as recorded (e.g. mg/dL in MIMIC-IV and eICU, µmol/L in HiRID) |
| MEASUREMENT.unit_concept_id | the converted unit | the recorded unit; °F for Fahrenheit temperatures; mL for urine (was mL/h); 10*3/uL for WBC, platelets and HiRID lymphocytes (was [U] or %); g/dL for MCHC (was %) |
| MEASUREMENT per-item height/weight (MIMIC-IV) | converted to cm/kg (lbs × 0.454) | as charted (in, lb), unit_source_value the source unit |
| VISIT_OCCURRENCE.visit_end_datetime, DEATH.death_datetime | rounded to whole seconds | exact |
| HiRID length of stay | truncated to whole seconds | exact |
| FACT_RELATIONSHIP | empty | specimen ↔ visit |
| SPECIMEN (MIMIC-IV) | cultures without charttime dropped | at chartdate + 12 h, as ricu; assigned to the stay by chartdate, as ricu's `upgrade_id` (since iter6) |
| DRUG_EXPOSURE.quantity | float32 | DOUBLE; MIMIC-IV `uL` doses × 0.001 (was × 1000) |
| DRUG_EXPOSURE.sig numbers | polars' float format | duckdb's |
| text values | polars' number grammar (' 81' stayed text) | trimmed (' 81' is 81) |
| csv floats | parsed as pandas does (last-bit differences for 17-digit values) | correctly rounded |

The differences listed in earlier versions of this file on columns the cohorts do not read
(drug ingredients, CONCEPT from the Athena download, source strings instead of BlendedICU
categories, CARE_SITE per source unit, LOCATION, OBSERVATION_PERIOD, CDM v5.4 types) still hold.

## Kept from BlendedICU

- MIMIC-IV labs and cultures go to a stay by ricu's `upgrade_id`. An admission with a
  null-`outtime` stay copies its rows onto two stays.
- MIMIC-IV ventilator mode (223849) rows keep MIMIC's `valuenum` in `value_as_number`.
- eICU `intakeoutput:albumin` and `intakeoutput:magnesium` cells feed albumin and magnesium.
  BlendedICU matched lab labels in every table.
- eICU `vitalperiodic:cvp` and `st1`-`st3` rows under concept 0.
- eICU DRUG_EXPOSURE keeps cancelled medication orders and rows with a null drug name. The
  infusion unit is the text in the first parentheses of the drug name.
- HiRID: the variableid set per concept is BlendedICU's. 30005010 stays under concept 0 next to
  30005110. The stay ends at the last observation of any variable, and the death time is the last
  heart rate or MAP row.
