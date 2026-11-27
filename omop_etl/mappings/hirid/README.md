# mappings/hirid

## measurements.csv

One row per HiRID `observations` variableid that becomes a MEASUREMENT row. Columns as in every
`<dataset>/measurements.csv`:

| column | meaning |
|---|---|
| `source_table` | `observations` |
| `source_code` | HiRID variableid; written verbatim to `measurement_source_value` |
| `source_label` | description from ricu's public concept dictionary (`concept-dict.json`, GPL-3; the ricu concept name in parentheses), or a short description in our own words for items ricu does not list. The HiRID variable reference itself is credentialed and is not copied here |
| `variable` | the BlendedICU variable name |
| `concept_id`, `unit_concept_id` | OMOP ids (integers only); `unit_concept_id` is the UCUM unit the variable is recorded in, and values are stored as recorded |
| `rule` | a derivation in `hirid.py`: `gcs_component` eye/motor/verbal, summed into a derived total (`derived:gcs_total`, concept 3007194) where all three are recorded at one time |

The variableid set per concept is BlendedICU's selection by reference name. Concept 0 rows are kept
on purpose: the consumer reads fluid balance (30005110) and the ventilator airway code (15001552)
by source value. Unit conversions are the consumer's (`cohorts/yaib_omop_crosswalk.py`,
`UNIT_CONVERSIONS`).

Drug names are not mapped here: `drug_source_value` is the pharma name read from the HiRID
variable reference at run time; ingredients come from `mappings/drug_ingredients.csv`.

Neutrophils (24000550) are an absolute count (concept 3013650, 10^3/µL). ricu-extensions' `neut` is
the % of the wbc of the same hour (`blood_cell_ratio`); the cohorts compute it
(`cohorts/ehrdata_dynamics.ratio_of_wbc`).
