"""Per-dataset antibiotic sets for the sepsis label, taken from the `abx_duration` concept of YAIB-cohorts
(https://github.com/rvandewater/YAIB-cohorts, MIT; ricu-extensions/configs/medications/concept-dict.json) and matched
on DRUG_EXPOSURE.drug_source_value, which holds the source label (miiv d_items label, hirid pharma name, eicu
drugname).

miiv and hirid: the concept lists item ids; the labels below are those of the miiv d_items dictionary (also in the
open MIMIC-IV demo) and of HiRID's pharma items as HIRID-ICU-Benchmark publishes them
(https://github.com/ratschlab/HIRID-ICU-Benchmark, MIT; preprocessing/resources/varref.tsv). Labels absent from the
CDM match nothing: 29 of the 51 miiv items (the ETL does not extract them) and most of the hirid ones, including IV Vancocin (pharmaid 189), Rocephin 2g, Tazobac 4g, Fortam, Cepimex and
Co-Amoxi. eicu: ricu's two drugname regexes, each on its table (infusiondrug, medication) where
the CDM records the table in `sig`; on older CDMs either regex on any row.
"""

from __future__ import annotations

# (itemid, d_items label)
MIIV_ABX = [(225798, 'Vancomycin'),
 (225837, 'Acyclovir'),
 (225838, 'Ambisome'),
 (225840, 'Amikacin'),
 (225842, 'Ampicillin'),
 (225843, 'Ampicillin/Sulbactam (Unasyn)'),
 (225844, 'Atovaquone'),
 (225845, 'Azithromycin'),
 (225847, 'Aztreonam'),
 (225848, 'Caspofungin'),
 (225850, 'Cefazolin'),
 (225851, 'Cefepime'),
 (225853, 'Ceftazidime'),
 (225855, 'Ceftriaxone'),
 (225857, 'Chloroquine'),
 (225859, 'Ciprofloxacin'),
 (225860, 'Clindamycin'),
 (225862, 'Colistin'),
 (225863, 'Daptomycin'),
 (225865, 'Doxycycline'),
 (225866, 'Erythromycin'),
 (225868, 'Ethambutol'),
 (225869, 'Fluconazole'),
 (225871, 'Foscarnet'),
 (225873, 'Gancyclovir'),
 (225875, 'Gentamicin'),
 (225876, 'Imipenem/Cilastatin'),
 (225877, 'Isoniazid'),
 (225879, 'Levofloxacin'),
 (225881, 'Linezolid'),
 (225882, 'Mefloquine'),
 (225883, 'Meropenem'),
 (225884, 'Metronidazole'),
 (225885, 'Micafungin'),
 (225886, 'Moxifloxacin'),
 (225888, 'Nafcillin'),
 (225889, 'Oxacillin'),
 (225890, 'Penicillin G potassium'),
 (225892, 'Piperacillin'),
 (225893, 'Piperacillin/Tazobactam (Zosyn)'),
 (225895, 'Pyrazinamide'),
 (225896, 'Quinine'),
 (225897, 'Ribavirin'),
 (225898, 'Rifampin'),
 (225899, 'Bactrim (SMX/TMP)'),
 (225900, 'Dalfopristin/Quinupristin (Synercid)'),
 (225902, 'Tobramycin'),
 (225903, 'Valgancyclovir'),
 (225905, 'Voriconazole'),
 (227691, 'Keflex'),
 (228003, 'Tamiflu')]  # ricu-extensions lists it (configs/medications/concept-dict.json): the list is MIMIC-IV's
# antibiotics item category, which holds antivirals, antifungals and chloroquine too; kept for parity

# (pharmaid, pharma name); note the double space in 'Vancocin  Amp 500 mg'
HIRID_ABX = [(163, 'Fortam Stechamp 2g'),
 (176, 'Ciproxin 200 mg/100ml'),
 (181, 'Metronidazole inf 500 mg/100ml'),
 (186, 'Rimactan inf 300 mg'),
 (189, 'Vancocin  Amp 500 mg'),
 (300, 'Metronidazole tabl 200 mg'),
 (326, 'Noroxin tabl 400 mg'),
 (331, 'Vancocin oral Kps 250 mg'),
 (351, 'Rimactan Kps 600 mg'),
 (405, 'Tazobac 2 g inf'),
 (1000234, 'Zinacef inj 100 mg/ml'),
 (1000272, 'Augmentin Tabl 625 mg'),
 (1000273, 'Clamoxyl Inj Lsg'),
 (1000274, 'Augmentin 1.2 Inj Lsg'),
 (1000284, 'Penicillin G 1 Mio'),
 (1000299, 'Kefzol Inj Lsg'),
 (1000300, 'Cepimex'),
 (1000302, 'Fortam 1 g Inj Lsg'),
 (1000304, 'Rocephin 2g'),
 (1000305, 'Zinacef Amp 1.5 g'),
 (1000306, 'Zinat Tabl 500 mg'),
 (1000315, 'Ciproxin Tbl 250 mg'),
 (1000317, 'Klacid Tbl 500 mg'),
 (1000318, 'Klacid Amp 500 mg'),
 (1000320, 'Dalacin C Kps 300 mg'),
 (1000321, 'Dalacin C Phosphat Inj Lsg 300 mg'),
 (1000322, 'Dalacin Phosphat Inj Lsg 600 mg'),
 (1000335, 'Vibravenös Inj Lsg 100 mg 5 ml'),
 (1000348, 'Myambutol 400 mg/4ml'),
 (1000352, 'Floxapen Inj Lsg'),
 (1000363, 'Garamycin'),
 (1000365, 'SDD Gentamycin/Polymyxin Kps'),
 (1000390, 'Rifater Drg 50 mg'),
 (1000407, 'Tavanic Tbl 500 mg'),
 (1000408, 'Tavanic Inf Lsg 500 mg 100 ml'),
 (1000424, 'Meronem 1g'),
 (1000425, 'Meronem 500 mg'),
 (1000426, 'Salofalk Tbl 500 mg'),  # RICU CHOICE: not an antibiotic, kept as ricu lists it
 (1000437, 'Flagyl Tbl 500 mg'),
 (1000483, 'Tazobac Inf 4g'),
 (1000507, 'Bactrim forte Lacktbl'),
 (1000508, 'Bactrim Inf Lsg'),
 (1000518, 'Tienam 500 mg'),
 (1000519, 'Obracin 80 mg'),
 (1000549, 'Augmentin Inj 2.2g'),
 (1000601, 'Clamoxyl Inj Lsg 2g'),
 (1000648, 'Rocephin 1 g Inf Lsg'),
 (1000666, 'Cepimex Amp 1g'),
 (1000670, 'Floxapen Inj Lsg 2g'),
 (1000671, 'Kefzol Stechamp 2g'),
 (1000760, 'Topamax Tbl 50 mg'),  # RICU CHOICE: not an antibiotic, kept as ricu lists it
 (1000781, 'Augmentin AD Tbl 1 g'),
 (1000791, 'Erythrocin Inf Lsg'),
 (1000797, 'Rimactan Kps 300 mg'),
 (1000812, 'Augmentin 2.2 Inf Lsg'),
 (1000825, 'Cefepime 2g Amp.'),
 (1000829, 'Dalacin C 600 Phosphat Amp.'),
 (1000830, 'Fortam 2g Inj Lsg'),
 (1000837, 'Piperacillin-Tazobactam 2.25 Inj Lsg'),
 (1000838, 'Rocephin 2 g Inf Lsg'),
 (1000854, 'Avalox Filmtbl 400 mg'),
 (1000855, 'Avalox Inf Lsg 400 mg'),
 (1000893, 'Ciproxin Tbl 500 mg'),
 (1000894, 'Ciproxin Infusion 400 mg'),
 (1001005, 'Colidimin Tbl 200 mg'),
 (1001068, 'Rifampicin Filmtbl 600 mg'),
 (1001075, 'Doxyclin Tbl 100 mg'),
 (1001079, 'Xifaxan Tabl 550 mg'),
 (1001084, 'Meropenem 500 mg'),
 (1001086, 'Meropenem 1g'),
 (1001095, 'Co-Amoxi Tbl 625 mg'),
 (1001096, 'Co-Amoxi Tbl 1g'),
 (1001097, 'Co-Amoxi 1.2 g Inj Lsg'),
 (1001098, 'Co-Amoxi 2.2g Inf Lsg'),
 (1001168, 'Clindamycin Posphat 300'),
 (1001169, 'Clindamycin Posphat 600'),
 (1001170, 'Clindamycin Kps 300 mg'),
 (1001171, 'Norfloxacin Filmtbl 400 mg'),
 (1001173, 'Rifampicin Inf Lsg'),
 (1001193, 'Bactrim Amp 400/80 mg Inf Lsg'),
 (1001198, 'Bactrim Amp. 400/80 mg Inf Lsg')]

EICU_ABX_REGEX = (
    # infusiondrug
    r"bactrim|cipro|flagyl|metronidazole|zithromax|zosyn|(((amika|cleo|ofloxa)|(azithro|clinda|tobra|vanco)my)c"
    r"|(ampi|oxa|peni|pipera)cill|cefazol|levaqu|rifamp)in",
    # medication
    r"cipro|flagyl|maxipime|metronidazole|tazobactam|zosyn|cef(azolin|epime)"
    r"|(((azithro|clinda|vanco)my|ofloxa|vanco)c|levaqu|piperacill|roceph)in",
)


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


EICU_ABX_TABLES = ("infusiondrug", "medication")


def abx_condition(ds: str, col: str = "d.drug_source_value", sig_col: str | None = None) -> str:
    """SQL condition selecting ricu's antibiotics of dataset `ds` in DRUG_EXPOSURE. With
    `sig_col` (the CDM records the source table in DRUG_EXPOSURE.sig), each eicu regex
    applies to its own table as in ricu; without it, either regex on any row."""
    if ds == "eicu":
        # R's grepl in ricu's rgx_itm is case-insensitive (ignore.case = TRUE)
        conds = [f"regexp_matches({col}, {_quote(r)}, 'i')" for r in EICU_ABX_REGEX]
        if sig_col is not None:
            conds = [f"({c} AND {sig_col} LIKE '%table={t}%')" for c, t in zip(conds, EICU_ABX_TABLES)]
        return "(" + " OR ".join(conds) + ")"
    names = {"miiv": MIIV_ABX, "hirid": HIRID_ABX}[ds]
    return f"{col} IN (" + ", ".join(_quote(n) for _, n in names) + ")"
