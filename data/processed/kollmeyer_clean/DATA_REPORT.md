# Kollmeyer LG 18650HG2 data cleaning report (v2)

- Date: 2026-09-28
- Code: `src/preprocess_kollmeyer.py` (every manual decision is written as a constant at the top of the file)
- Run record: `notebooks/01_preprocess_kollmeyer.ipynb`
- Per-file record: `manifest.json` (source, test type, batch, full capacity and its source, dropped rows, splits, logging interval, original start and end times of every series)
- Overall timeline: `file_timeline.csv` (the 208 raw files in real time order)

This report lists every data-processing decision (D1, D2, ...) with its reason, so it can be rewritten directly into the thesis.

---

## 0. Summary

| Item | Count |
|---|---|
| Raw CSV files | 208 (6 temperature folders) |
| Raw files included | 100 -> main data set of 102 series (2 files split in two at logging gaps), 4,789,846 rows, 251 MB |
| Raw files excluded | 108 (98 Charge files + 10 faulty tests) |
| `_excluded/` | 129 series (the 108 excluded files, 15 removed "warm up -> charge -> cool down" tails, the first run of 582_LA92), 159,091 rows, 7 MB |

Every series is an independent time series with the columns `time_s, voltage, current, temperature, soc, ah`. No windowing, no normalisation, no train/test split and no resampling were done. These are left to the model stages (EKF, CNN, LSTM, Transformer), each as it needs.

---

## 1. Data source and reading

**D1. Data set.** The public LG 18650HG2 (nominal 3 Ah) data set of Kollmeyer et al., McMaster University. The raw data are on drive D and are only read, never changed. Test equipment: a Digatron 75 A channel (voltage and current accuracy 0.1 % of full scale); temperature from an AD592 sensor on the middle of the cell surface (±1 °C). 204 of the 208 CSV headers name the same cell (LG HG2 18650_SN62A4); the exception of the other 4 (batch 575) is in D43.

**D2. Test types and file counts.**

| Test type | −20 °C | −10 °C | 0 °C | 10 °C | 25 °C | 40 °C | Total |
|---|---|---|---|---|---|---|---|
| Drive cycles (UDDS / HWFET / LA92 / US06) | 4 | 4 | 4 | 4 | 4 | 4 | 24 |
| Mixed1-8 (random mix of the four cycles) | 8 | 8 | 7 | 7 | 8 | 8 | 46 |
| Cap_1C (1C capacity test) | 2 | 2 | 1 | 2 | 2 | 1 | 10 |
| C20DisCh (C/20 discharge + C/20 charge) | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| Dis_0p5C / Dis_2C (0.5C, 2C constant-current discharge) | 2 | 2 | 2 | 2 | 2 | 2 | 12 |
| HPPC (pulses) | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| PausCycl (the first 18-36 minutes of the UDDS profile, see D26) | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| Charge (1C charge between two tests) | 16 | 19 | 15 | 14 | 17 | 17 | 98 |
| Total | 35 | 38 | 32 | 32 | 36 | 35 | 208 |

**D3. Real test order.** Sorted by time stamp the real order is 25 °C (10/25-10/31) -> 40 °C (11/02-11/08) -> 10 °C (11/12-11/26) -> 0 °C (11/26-12/02) -> −10 °C (12/04-12/19) -> −20 °C (12/21-12/26), all in 2018. The Readme puts 40 °C before 25 °C, which does not match the data. Because a single cell was used throughout, the capacity measured at each temperature reflects both temperature and aging (about 276 Ah discharged in total by the last capacity test).

**D4. CSV format.** The format of all 208 files was checked one by one: rows 1-28 are a description, row 29 the column names, row 30 the units, data from row 31. Columns used: ProgTime (time since program start), Time Stamp (real date and time, 1 s resolution), Step, Status, StepTime, Procedure, Voltage, Current, Temperature, Capacity.

**D5. Time stamps.**
- A Time Stamp exactly at midnight carries only the date (73 rows in 9 files); it is completed with 0 h when read.
- The tester's Time Stamp equals "system time at program start + real elapsed time", so programs started before the end of daylight saving time on 2018-11-04 (all of 25 °C, and 555 at 40 °C) are recorded in EDT (UTC−4), later ones in EST (UTC−5). This is why 555 and 556 seem to overlap by 1 hour; converted to UTC they are 18 s apart.
- The manifest and the timeline give both the local time of the original record and UTC.

---

## 2. Output format

**D6. One CSV file per series.** Reasons:
- The EKF may be written in MATLAB or Python and the neural networks in Python; every tool reads CSV directly.
- The column names are in the header and the files open in Excel for manual checks (the largest file has about 160,000 rows, below Excel's row limit).
- Written with the original precision, no information lost: time 3 decimals, voltage, current, temperature and ah 5 decimals each, soc 6 decimals.
- Drawback: larger and slower to read than binary. Reading everything takes only a few seconds; model stages can cache npy or pt files if needed. Parquet needs pyarrow, which was not installed to keep the environment unchanged.

**D7. Columns.**

| Column | Unit | Description |
|---|---|---|
| time_s | s | counted from the first row of the series (from ProgTime), strictly increasing |
| voltage | V | terminal voltage |
| current | A | negative for discharge, positive for charge (as in the raw data) |
| temperature | °C | cell surface temperature |
| soc | fraction 0-1 | definition in section 3 |
| ah | Ah | raw cumulative Ah of the tester (Capacity column), for the rows kept after cleaning |

**D8. The `ah` column is only a reference for the label and must never be a model input.** It is kept so that the SOC can be recomputed with another definition later (e.g. the nominal 3.0 Ah or the C/20 capacity): `soc_new = soc_anchor + (ah − ah_anchor) / new full capacity`, with soc_anchor and ah_anchor stored in the manifest. ah itself is the result of Coulomb counting, i.e. the source of the SOC label; using it as an input would hand the answer to the model.

**D9. File names and folders.**
- Main data set: `<temperature>/<original file name>.csv`, with `_part1`, `_part2` added when split at a logging gap.
- Excluded data: `_excluded/<temperature>/...`. A removed tail is called `<original file name>_tail.csv`; the first run of 582 is `582_LA92_attempt1_partK.csv`.
- Every series in manifest.json has `excluded: true/false`; excluded ones also have `exclusion_reason` and `exclusion_class` (a, b, c, see D29).

---

## 3. SOC definition

**D10. How SOC is computed.** `soc = soc_anchor + (ah − ah_anchor) / full capacity`.
- Before every test the cell is charged at about 24 °C with 1C to 4.2 V / 50 mA, so in general soc_anchor = 1 and ah_anchor = 0 (the Ah counter is 0 at the start of the test).
- Two exceptions: Dis_0p5C, see D14; the first kept row of 582_LA92 has already discharged 0.0144 Ah, so it starts at 99.4 %.

**D11. Meaning of the full capacity.** The full capacity is the batch's 1C discharge capacity at the actual test temperature (section 4). SOC = 0 means "discharged at 1C to 2.8 V at that temperature", the usual way to use this data set per temperature. **This differs from the Kollmeyer example script**: their preprocessing computes SOC with the nominal 3.0 Ah, so in their data the SOC of the −10 °C drive cycles only falls to about 28 % and at 25 °C to about 14 %. Any comparison with their published results must state this difference.

**D12. The SOC of characterisation tests can be negative.** HPPC, C/20, 0.5C and 2C use the 1C full capacity of the same temperature (the same yardstick as the drive-cycle labels), so the EKF can use the soc column of the C/20 file directly as the OCV-SOC curve and compare it with the drive-cycle labels without conversion. The price: at low temperature these tests discharge more than the 1C capacity and the SOC falls below 0:
- end of the C/20 discharge: −47 % at −20 °C, −13 % at −10 °C, −4.6 % at 0 °C, −3.5 % at 10 °C, −1.5 % at 25 °C, −0.7 % at 40 °C;
- end of HPPC: −36 % at −20 °C, about 0 % at 40 °C.

A negative value means "more charge has been removed than the 1C full capacity".

**D13. SOC slightly above 100 % is not clipped.** With regenerative braking at the start of a cycle the current is positive and Ah is slightly above 0; SOC reaches at most 100.07 %. This is the real Coulomb count; clipping would break the link between SOC and current.

**D14. Start SOC of Dis_0p5C (back-calculated from the Ah balance).** Dis_0p5C directly follows the C/20 charge, which was done at the test temperature, not with a 1C charge to full at 24 °C as for the other tests. At low temperature the cell is not really full when the CV phase stops at 50 mA. Two independent pieces of evidence:
- If the removed 24 °C recharge part is counted on, the HPPC and 2C tails end at 100.3-101.5 %, but Dis_0p5C ends at 107.5 % at −20 °C and 102.8 % at −10 °C.
- The rest voltage after the C/20 charge is lower than after a full charge at 24 °C: 47 mV lower at −20 °C, 25 mV at −10 °C, 13 mV at 0 °C.

The charge missing at the start is therefore back-calculated as "Ah balance of the Dis_0p5C tail − Ah balance of the same-temperature Dis_2C tail", which gives the start SOC:

| Temperature | −20 °C | −10 °C | 0 °C | 10 °C | 25 °C | 40 °C |
|---|---|---|---|---|---|---|
| Start SOC | 93.05 % | 97.63 % | 98.87 % | 99.54 % | 100 % (no tail) | 100.14 % |

The uncertainty is about ±0.5 percentage points, much smaller than the error of simply using 100 % (about 7 percentage points at −20 °C). The Ah counter of Dis_0p5C is not reset at the start of the discharge but continues from the C/20 test, so ah_anchor is the Ah at the start of the discharge step (equal to the last Ah of the C/20 file).

**D15. The SOC of the C/20 charge does not close; kept as is.** The charged Ah of C/20 differs from the discharged Ah (1.7 % less at −20 °C; 1.5-3.4 % more at −10, 0 and 10 °C; 6.6-6.8 % more at 25 and 40 °C), so the C/20 files end with soc between 0.975 and 1.069. A normal Coulomb efficiency is above 99.5 %, so this looks like an accumulated current offset over about 20 hours at the small current of 0.15 A (the accuracy is 0.1 % of the 75 A full scale, i.e. ±75 mA; an offset of about 5 mA explains it). The measured values were not changed; if the EKF uses the charge part, normalising each part separately is recommended.

---

## 4. Full capacity

**D16. One rule set.** The full capacity of every batch must come from a capacity measured at the batch's actual temperature:
1. **Rule 1**: the batch's own Cap_1C file exists and was measured at the right temperature -> use it.
2. **Rule 2**: the batch's Cap_1C file is missing (the test was done but is not in the data set) -> back-calculate from the tester's cutoff capacity (Ah discharged at the end of each cycle / 0.95). Condition: the back-calculated value is within ±0.5 % of the aging-model prediction; this threshold was set beforehand.
3. **Rule 3**: the batch's capacity is shown to have been measured at the wrong temperature -> borrow the nearest valid capacity at the same temperature, with an aging correction for the Ah discharged between the two tests.

**D17. Aging model.**
- The capacity fade is not consistent with time (0.17-2.4 %/day) but is proportional to the discharged Ah. The model is therefore: later capacity = earlier capacity × (1 − rate × Ah discharged in between).
- The rate is calibrated on paired 1C capacity tests at the same temperature: 551 -> 552 at 25 °C gives 0.0615 %/Ah, 596 -> 602 at −10 °C gives 0.0690 %/Ah, pooled **0.0652 %/Ah**, used for every temperature except −20 °C.
- 610 -> 611 at −20 °C gives **0.2592 %/Ah**, clearly larger, because at −20 °C the 1C capacity is mainly limited by resistance. This rate **comes from this one pair only**.
- Validation: the pooled rate predicts the 4 missing Cap_1C of the data set (cutoff capacities of 582, 590, 601, 604) with errors between −0.19 % and +0.40 %.
- The cumulative discharge includes tests that were done but are not in the data set, with estimated discharges: the Cap_1C of 556, 567, 582, 590, 601, 604, and the UDDS, HWFET, LA92 of 567. The basis is the time gap between the program start and the first file (estimates and basis in the manifest under `aging_model.missing_tests`).

**D18. Full capacity and final SOC of every batch.**

| Temperature | Batch | Files included | Rule | Full capacity (Ah) | Source | Final SOC of the cycles |
|---|---|---|---|---|---|---|
| −20 °C | 610 | UDDS, HWFET, LA92, US06, Mixed1-2, Cap_1C | 1 | 1.6713 | 610_Cap_1C (−19.8 -> −15.4 °C during discharge) | 5.0 % |
| −20 °C | 611 | Mixed3-8, Cap_1C, PausCycl | 1 | 1.6172 | 611_Cap_1C (−19.9 -> −15.6 °C) | 5.0 % (PausCycl 85.0 %) |
| −10 °C | 596 | UDDS, HWFET, LA92, Cap_1C | 1 | 2.2634 | 596_Cap_1C (−10.1 -> −6.4 °C) | 5.0 % |
| −10 °C | 601 | US06, Mixed1-2 | 2 | 2.2419 | back-calculated from the cutoff capacity, −0.19 % from the prediction (reference 602, 9.8 Ah apart) | 5.0 % |
| −10 °C | 602 | Mixed4-5, Cap_1C | 1 | 2.2319 | 602_Cap_1C (−10.0 -> −6.5 °C) | 5.0 % |
| −10 °C | 604 | Mixed3, Mixed6-8, PausCycl | 2 | 2.2196 | back-calculated, −0.07 % (reference 602, 7.4 Ah) | 5.0 % (PausCycl 85.0 %) |
| 0 °C | 589 | UDDS, HWFET, LA92, US06, Mixed1-2, Cap_1C | 1 | 2.4734 | 589_Cap_1C (−0.6 -> 2.2 °C) | 5.0 % |
| 0 °C | 590 | Mixed4-8, PausCycl | 2 | 2.4501 | back-calculated, +0.35 % (reference 589, 19.8 Ah) | 5.0 % (PausCycl 85.0 %) |
| 10 °C | 567 | US06, Mixed1-2 | rule 2 failed, rule 3 used instead | 2.6332 | 2.5465 of 576 with an aging correction for 50.4 Ah (see D19) | 5.8 % |
| 10 °C | 571 | Mixed4-8, PausCycl | 3 | 2.5953 | 2.5465 of 576 with an aging correction for 28.8 Ah (see D20) | 9.0 % (PausCycl 85.6 %) |
| 10 °C | 576 | UDDS, HWFET, Cap_1C | 1 | 2.5465 | 576_Cap_1C (9.2 -> 11.0 °C) | 5.0 % |
| 10 °C | 582 | LA92 | 2 | 2.5421 | back-calculated, +0.40 % (reference 576, 8.7 Ah) | 5.0 % (start 99.4 %) |
| 25 °C | 551 | UDDS, LA92, US06, Mixed1-2, Cap_1C | 1 | 2.7264 | 551_Cap_1C (23.9 -> 25.5 °C) | 5.0 % |
| 25 °C | 552 | Mixed3-8, Cap_1C | 1 | 2.6928 | 552_Cap_1C (23.8 -> 25.5 °C) | 5.0 % |
| 40 °C | 556 | UDDS, HWFET, LA92, US06, Mixed1-2 | 2 (no same-temperature reference) | 2.6585 | back-calculated, cross-temperature check only (see D21) | 5.0 % |
| 40 °C | 557 | Mixed3 | 3 | 2.6197 | 2.6585 of 556 with an aging correction for 22.3 Ah (see D20) | 9.3 % |

The Cap_1C files use their own batch's capacity, so they all go from 100 % to 0 %. Inside one batch the final SOC of the cycles differs by at most 0.01 percentage points, because the tester always stopped at the same Ah.

**D19. Batch 567.**
- 567's own Cap_1C is not in the data set; the back-calculated cutoff capacity is 2.6101 Ah.
- With the aging model, going back 50.4 Ah from 576_Cap_1C (2.5465 Ah, measured at 9.2-11.0 °C), the capacity of 567 at that time should be **2.628-2.638 Ah** (once with the rate of each calibration pair; middle value 2.6332).
- **2.6101 is 0.88 % below the prediction, beyond the ±0.5 % threshold set beforehand, so it was not used; the rule-3 approach gives 2.6332 Ah and a final SOC of 5.8 %.** The direction of the deviation of 2.6101 (too low) matches the two cold Cap_1C tests (571, 557), but the file of that test is not in the data set, so its temperature cannot be checked.
- **Uncertainty range**: with 2.6101 the final SOC would be 5.0 %, so the final SOC of 567 is stated as **5.0 %-6.0 %**.
- 2.5465 is more than 3 % below the expected capacity of 567 at that time and was ruled out.
- The main grounds for the choice are "can it be checked" and the aging trend. **The rest-voltage check cannot separate the two possibilities and is not used**: aging makes the 1C capacity fall faster than the C/20 capacity, so both hypotheses explain the low rest voltage of 567.

**D20. Batches 571 and 557 (rule 3).**
- 571_Cap_1C was actually discharged at 0.7-3.6 °C, and during 557_Cap_1C the cell cooled from 38.4 °C to 1.3 °C; neither is at its folder temperature, yet the tester stopped at these too-small capacities.
- 571: borrows the nearest valid 10 °C capacity, 576 (2.5465 Ah, 5.5 days and 28.8 Ah apart), aging-corrected to **2.5953 Ah**, final SOC 9.0 %. The value of 567 failed rule 2 and is not used as reference.
- 557: borrows 556, the only usable value at 40 °C (2.6585 Ah, 1.6 days and 22.3 Ah apart), corrected to **2.6197 Ah**, final SOC 9.3 %.
- Both final SOCs are within the plausible range of 5-12 % set beforehand.

**D21. Batch 556: no same-temperature reference at 40 °C, cross-temperature check only.**
- The only capacity test at 40 °C, 557_Cap_1C, is cold, so the back-calculated 2.6585 Ah of 556 cannot be checked the rule-2 way.
- Cross-temperature check instead: from 552 (25 °C, 2.6928 Ah), aged to the time of 556, gives 2.6340 Ah. **Assuming the 1C capacity at 40 °C is 0.5 %-1.0 % above that at 25 °C**, the expected range is 2.647-2.660 Ah; 2.6585 lies inside.
- 2.6585 is the only usable value at 40 °C and is used.

**D22. Characterisation batches (HPPC, C/20, 0.5C, 2C).**
- These 6 batches have neither their own capacity test nor a cutoff capacity. The full capacity is borrowed from the drive-cycle batch that follows, and, consistent with rule 3, corrected file by file for the Ah discharged between the file's start and the reference capacity test.
- −20 °C uses the rate of the 610/611 pair, 0.2592 %/Ah (**from that one pair only**); the other temperatures use the pooled rate of 0.0652 %/Ah.

| Temperature | Reference (before correction) | HPPC | C20DisCh | Dis_0p5C | Dis_2C |
|---|---|---|---|---|---|
| −20 °C | 610: 1.6713 | 1.7071 (8.1 Ah, +2.14 %) | 1.6964 (5.7 Ah, +1.50 %) | 1.6853 (3.2 Ah, +0.84 %) | 1.6766 (1.2 Ah, +0.32 %) |
| −10 °C | 596: 2.2634 | 2.2774 (9.4 Ah) | 2.2734 (6.8 Ah) | 2.2696 (4.2 Ah) | 2.2661 (1.9 Ah) |
| 0 °C | 589: 2.4734 | 2.4897 (10.1 Ah) | 2.4853 (7.3 Ah) | 2.4810 (4.7 Ah) | 2.4770 (2.3 Ah) |
| 10 °C | 576: 2.5465 | 2.5642 (10.6 Ah) | 2.5592 (7.6 Ah) | 2.5548 (4.9 Ah) | 2.5505 (2.4 Ah) |
| 25 °C | 551: 2.7264 | 2.7467 (11.3 Ah) | 2.7409 (8.1 Ah) | 2.7359 (5.3 Ah) | 2.7310 (2.6 Ah) |
| 40 °C | 556: 2.6585 | 2.6776 (11.0 Ah) | 2.6722 (7.9 Ah) | 2.6675 (5.2 Ah) | 2.6628 (2.5 Ah) |

In brackets: the Ah discharged up to the reference capacity test. Except at −20 °C the corrections are +0.12 % to +0.74 %.

---

## 5. Included and excluded files

**D23. The 100 raw files included.**
- −20 °C (19): HPPC, C20DisCh, Dis_0p5C, Dis_2C of 607; Cap_1C, UDDS, HWFET, LA92, US06, Mixed1-2 of 610; Cap_1C, Mixed3-8, PausCycl of 611.
- −10 °C (19): the 4 characterisation tests of 593; Cap_1C, UDDS, HWFET, LA92 of 596; US06, Mixed1-2 of 601; Cap_1C, Mixed4-5 of 602; Mixed3, Mixed6-8, PausCycl of 604.
- 0 °C (17): the 4 characterisation tests of 585; Cap_1C, UDDS, HWFET, LA92, US06, Mixed1-2 of 589; Mixed4-8, PausCycl of 590.
- 10 °C (17): the 4 characterisation tests of 575; US06, Mixed1-2 of 567; Mixed4-8, PausCycl of 571; Cap_1C, UDDS, HWFET of 576; LA92 of 582.
- 25 °C (17): the 4 characterisation tests of 549; Cap_1C, UDDS, LA92, US06, Mixed1-2 of 551; Cap_1C, Mixed3-8 of 552.
- 40 °C (11): the 4 characterisation tests of 555; UDDS, HWFET, LA92, US06, Mixed1-2 of 556; Mixed3 of 557.

**D24. Constant-current and pulse tests are included.** Cap_1C, C/20, 0.5C, 2C and HPPC are all kept, because the EKF needs them to identify the OCV-SOC curve and the equivalent-circuit parameters.

**D25. The Charge files (98) are not included.** They are the routine 1C constant-current constant-voltage charges after every test, done at a cell temperature of 22-25 °C by design (independent of the folder temperature), with 60 s logging; they are not tests. They are cleaned and stored in `_excluded/` with an empty soc (see D30).

**D26. Only 4 PausCycl files are included.** Comparing the current profiles shows that PausCycl is the first 18-36 minutes of the UDDS profile: the cell is discharged from full to about 85 %, then rests for 10 minutes. The 4 files at −20, −10, 0 and 10 °C are normal and included. 552_PausCycl at 25 °C and 562_PausCycl at 40 °C are not (see D29).

**D27. The two cold Cap_1C files are not included.** 571_Cap_1C and 557_Cap_1C are no longer used for the full capacity (D20), and as time series they do not match their folder temperature.

**D28. Exclusions already confirmed in v1 are kept.** 551_HWFET and 562_Mixed4-8 stay excluded; reasons in D29.

**D29. Classes of excluded files.** (a) wrong folder, belongs to another temperature; (b) test aborted or logging error; (c) normal but special data.

| File | Class | What happened, and the evidence |
|---|---|---|
| 551_HWFET (25 °C) | b | 10/29 01:36, between Charge3 and Charge4. The whole file is a 10-minute rest (current always 0); the HWFET profile was not executed and was never repeated, so the data set has no HWFET at 25 °C |
| 552_PausCycl (25 °C) | b | 10/31, the last test at 25 °C. All 4 TABLE steps ended at once; only a 1.6 h rest and one 0.015 Ah top-up charge |
| 562_Mixed4-8, 562_PausCycl (40 °C folder) | b | 11/07 21:23 to 11/08 20:14, between the 40 °C and 10 °C test groups. The previous file, 557_Charge10, stopped before charging; afterwards the chamber was not switched on and the cell followed room temperature (18.5-25 °C, minimum 18.5 °C at night). The program restarted at the Charge10 step and skipped Cap_1C; the cutoff capacity was only about 0.2 Ah, so each Mixed discharged only 0.19 Ah. The profiles really are Mixed4-8. The temperature is close to the real temperature of the 25 °C folder, but there was no capacity test at that time (61 Ah discharged since 552_Cap_1C), so they are not relabelled as 25 °C data |
| 571_Cap_1C (10 °C) | a | 11/14 10:06. The charge files before and after are normal; the chamber was cold only during this hour: the cell cooled from 9.2 °C to 4.3 °C within 50 s and discharged at 0.7-3.6 °C, in effect a test at about 1-4 °C. The temperature kept falling during the test, and it is 15 days (54 Ah) before the 0 °C group, so it is unsuitable as the 0 °C full capacity |
| 557_Cap_1C (40 °C) | b | 11/07 10:49. During the discharge the cell cooled all the way from 38.4 °C to 1.3 °C; not at any constant temperature, cannot be reclassified |
| 98 Charge files | - | see D25 |

Both cold Cap_1C tests are the first step of the CyclesB program; the chamber was cold only in this step and normal before and after. The cause is not recorded.

**D30. Excluded data are still cleaned and stored.** In `_excluded/`, with the same columns and the ah column; the manifest marks them `excluded: true` with the reason. The soc column is filled only where it is well defined:
- removed tails: continue from the SOC at the end of the test, ending at 100.3-101.5 %;
- 571_Cap_1C, 557_Cap_1C: with their own measured capacity, from 100 % to 0 %. **Note: not the same yardstick as the main data set** (stated in the manifest);
- 551_HWFET: 100 % throughout;
- 552_PausCycl: with the full capacity of 552, ending at 100.6 %;
- first run of 582_LA92: with the full capacity of 582, starting at 100 %;
- the 6 files of 562: the start is 100 %, but the full capacity at that time and place cannot be determined; soc left empty;
- Charge files: the start SOC is uncertain; soc left empty.

---

## 6. Cleaning steps

**D31. Rows with empty values are removed.** Rows with an empty voltage, current, temperature, Ah or ProgTime are removed. Of all 208 files only 582_LA92 has 3 such rows, all in the part of the first run (see D33), so in the main data set this rule removes 0 rows; it is kept as a safeguard.

**D32. Rows with a repeated time stamp are removed; the first is kept.** Judged by ProgTime: a row equal to the previous one is removed.
- At the end of every step the tester logs an extra row with the same time, mostly identical to the previous row. 5,501 rows are removed from the main data set: about 200 per HPPC file, 0-2 per drive-cycle file.
- Rows 3868-8026 of 602_Mixed4 are the same record written 4159 times (identical time stamp, voltage, current). This is a "frozen" logger, directly followed by a 416 s gap; this rule removes the frozen records as well. v1 had kept these 4160 rows at the end of part1.

**D33. Only the part of 582_LA92 resumed after the pause is kept.**
- This LA92 first started on 11/25 at 20:59; after about 103 s the regenerative current pushed the voltage to 4.2345 V and the tester stopped on over-voltage. It paused for 13.2 hours and continued on 11/26 at 10:11.
- The logger wrote the 11/25 data again and inserted them into the 11/26 data, causing time to run backwards and 3 empty rows.
- Treatment: only rows with an 11/26 Time Stamp are kept (from 10:11:10, the state after the 13.2 h rest, start SOC 99.4 %), 89,701 rows. Compared with v1 ("drop the first 1945 rows") this keeps 907 more rows (91 s) of normal data after the restart.
- The first run of 11/25, de-duplicated, is in `_excluded/`. The file name and header date of this file are 11/25, but the LA92 kept actually ran on 11/26 from 10:11 to 12:40.

**D34. The "warm up -> 1C charge -> cool down" part at the end of HPPC, 0.5C and 2C is removed.**
- Except at 25 °C, after these three test types the file continues with: about 1 hour of rest to warm the cell to about 24 °C, a 1C charge, and a rest to cool back to the test temperature. The tester labels this part with the Procedure `NN_Char_Charge`, clearly separate from the test, and the Ah counter is reset there.
- The cut is made by Procedure. Every rest inside the test is kept. The test program has no rest after the last discharge; the rest after the discharge already belongs to NN_Char_Charge, and its first sample (60 s after the discharge) is already 1-2.5 °C off the test temperature, 10-20 °C after 5 minutes, so there is no "rest after discharge at the test temperature" to keep.
- At 25 °C these tests have no tail; their charges are in 549_Charge.

| File | Cut row (data row, from 0) | Raw CSV row | End of the kept part | Rows removed |
|---|---|---|---|---|
| 607 / 593 / 585 / 575 / 555_HPPC | 35860 / 39875 / 40008 / 44006 / 44305 | 35891 / 39906 / 40039 / 44037 / 44336 | the last SOC-lowering discharge to 2.8 V plus 1 row of 0 s rest; the 610 s rest before it is kept | 207 / 209 / 211 / 213 / 214 |
| 607 / 593 / 585 / 575 / 555_Dis_0p5C | 81 / 95 / 101 / 107 / 107 | 112 / 126 / 132 / 138 / 138 | the discharge (to 2.8 V) | 202 / 207 / 209 / 211 / 211 |
| 607 / 593 / 585 / 575 / 555_Dis_2C | 14 / 20 / 24 / 26 / 27 | 45 / 51 / 55 / 57 / 58 | the discharge (to 2.8 V) | 185 / 199 / 206 / 210 / 211 |

The removed rows include 1 "program end row" after the tail (D35). The removed tails are stored in `_excluded/` (D30).

**D35. Program-end marker rows are removed.** Rows with status STO (program stop) and the row after a tail where the Ah counter is reset are program markers, not measurements. 23 files have one each; none is in the main data set (the ends of 10 characterisation files and of 13 Charge files).

**D36. Splitting at logging gaps.**
- Rule: if either ProgTime or the time stamp converted to UTC has a step between two adjacent rows larger than "normal logging interval of the part + 3 s", the series is split. For data logged every 0.1 s this means "more than 3 seconds".
- The logging interval is judged by the tester's levels (0.1, 1, 10, 60 s): the median interval inside the step, rounded up to the nearest level. Without this, files logged every 10 s or 60 s would be split at every row.
- The time stamp is used only to find gaps (1 s resolution); repeated time stamps (D32) are still judged by ProgTime.
- Split positions (only these 4 files and 7 places in all the data):

| File | Split position | Reason | Result |
|---|---|---|---|
| 602_Mixed4 (main data set) | before row 8027 | ProgTime gap of 416.0 s (frozen rows removed first, see D32) | part1: 3,868 rows, 387 s, SOC 100 -> 93.5 %; part2: 57,649 rows, SOC 88.3 -> 5.0 % |
| 571_Mixed6 (main data set) | before row 391 | hidden pause: the time stamp jumps from 11/15 01:10:55 to 11:36:21 (10.4 h) while ProgTime advances only 0.003 s | part1: 391 rows, 39 s, SOC 100 -> 99.6 %; part2: 73,529 rows |
| 549_Charge (_excluded) | before rows 215, 431, 644 | gaps of 80,263 s, 151,846 s, 1,669 s (the file merges 4 separate charges) | 4 parts |
| first run of 582_LA92 (_excluded) | before rows 1943, 426 | gaps of 16.2 s, 3.6 s | 3 parts, one with a single row |

**D37. Hidden pauses.**
- Both hidden pauses have the same mechanism: at 10 °C, near full charge, a regenerative current pulse pushed the terminal voltage above 4.23 V (571_Mixed6: 4.2399 V, +2.86 A; 582_LA92: 4.2345 V, +2.99 A); the tester stopped for protection and the test was continued by hand the next day.
- ProgTime stops during the pause, so the pause cannot be found from ProgTime alone; the time stamp must be checked as well.
- In both pauses the Ah count is continuous, so the SOC label is not affected; but the cell rested for more than 10 hours and its polarisation state changed completely, so the series must be split.
- A scan of all files with the new rule found no other hidden pause.

**D38. Small gaps are not split.** Every LA92 repetition (every 14,361 rows) leaves a logging gap of 1.4-2.1 s, and UDDS has one at row 137,001. All are below the threshold and, as in v1, not split.

**D39. time_s starts at 0; no resampling.** time_s of every series starts at 0; the original start and end times (local and UTC) and the ProgTime row numbers are in the manifest. Different tests have different logging intervals: 0.1 s for drive cycles and PausCycl, 10 s for Cap_1C, 60 s for C/20, 0.5C and 2C, and HPPC switches between 0.1, 1 and 10 s. No resampling is done here; it is left to the model stages.

---

## 7. Data quality checks

**D40. Temperature check.** Criterion: the **minimum** cell temperature during the test part (discharge or cycle steps) is within 3 °C of the folder temperature. The minimum is used because self-heating only raises the temperature, so the minimum is closest to the chamber temperature.
- All 102 series of the main data set pass.
- Every failure is in the exclusion list: the 6 files of 562, 571_Cap_1C and 557_Cap_1C.
- The tests of the 40 °C folder actually ran at 38.2-39.5 °C and those of 25 °C at 23.6-24.3 °C, a constant offset. At low temperature US06 and the 2C discharge self-heat by 4-6 °C, but they all start at the chamber temperature.

**D41. Scan for frozen records.**
- All files were searched for stretches where "time advances but voltage, current, temperature and Ah all stay the same for more than 30 seconds": 2352 stretches, **all in rest steps** (current 0), the longest 15 minutes.
- The cause is the quantisation step: about 0.17 mV for voltage, about 0.105 °C for temperature. Late in a rest the voltage changes so slowly that the readings repeat.
- The voltage jump at the end of every stretch was checked: median 0; only 2 stretches above 0.42 mV. One is the join between two separate charges in 549_Charge; the other is only 0.5 mV, i.e. noise.
- Conclusion: all are normal rests and are not removed. The only real logging faults are the frozen rows of 602_Mixed4 (D32) and the two hidden pauses (D37).

**D42. Fingerprint check of the drive-cycle profiles.**
- Every drive-cycle file was checked with the correlation of its power profile; all match their file names (0.93-1.00).
- PausCycl matches the first part of UDDS (0.97-0.995).
- The data set has no HWFET at 25 °C and no Mixed3 at 0 °C and 10 °C: at these two temperatures the CyclesB program skipped Charge9 and Mixed3.

**D43. Batch 575 (HPPC, C20DisCh, Dis_0p5C, Dis_2C at 10 °C): wrong cell profile.**
- The header says LG ICR18650MG1 (nominal 2.9 Ah, "embedded thermocouple"). But the temperature readings lie on the same 0.10516 °C quantisation grid with the same offset as the other files; the 10 °C HPPC resistance (38 mOhm at 50 % SOC) lies between 0 °C (53 mOhm) and 25 °C (27 mOhm); and the C/20 capacity follows the aging trend. All of this shows it is still the same HG2 cell and only the wrong cell profile was chosen in the software.
- The only real effect: **the test currents were set for 2.9 Ah, 3.3 % lower than at the other temperatures** (e.g. 2C is 5.8 A, not 6 A), and each HPPC SOC step is 0.145 Ah, not 0.15 Ah.
- Kept, with a note in the manifest.

**D44. Regression comparison with v1.**
- In files whose full capacity did not change, voltage, current and temperature are identical to v1 point by point (differences within float32 precision).
- In files whose full capacity changed, the SOC satisfies (soc_new − 1) × C_new = (soc_old − 1) × C_old exactly.
- Every difference in row counts is explained: removed repeated rows, the frozen rows of 602, and the 907 extra rows kept after the restart of 582.

---

## 8. Known limitations

1. Where the full capacity depends on the aging model (567, 571, 557 and the characterisation files) there is model error. Within extrapolations of 7-20 Ah the validation error of the model is about ±0.4 %; 567 is extrapolated by 50 Ah and its error may be larger; its final SOC is stated as 5.0-6.0 % (D19).
2. The aging rate of the −20 °C characterisation tests comes from one pair of data only (D22).
3. 556 has only a cross-temperature check and depends on the assumption "the capacity at 40 °C is 0.5-1.0 % above that at 25 °C" (D21).
4. The start SOC of Dis_0p5C at low temperature is back-calculated from the Ah balance, with an uncertainty of about ±0.5 percentage points (D14).
5. The Coulomb count of the C/20 charge does not close (D15); Ah measured at small currents has limited accuracy.
6. The SOC of the characterisation tests uses the 1C capacity as yardstick and becomes negative at low temperature (D12). A thermodynamic SOC (0-100 % of the C/20 capacity) can be recomputed from the ah column.
7. C/20, 0.5C and 2C are logged every 60 s; they suit quasi-static analysis only, not the identification of fast dynamics.
8. The test currents of batch 575 are 3.3 % lower than at the other temperatures (D43).
9. The same cell aged throughout the data set; differences between temperatures also contain the effect of aging (D3).

---

## 9. Main differences from v1 (`archive/v1_2026-09-28/`)

| Aspect | v1 | v2 |
|---|---|---|
| Output | drive cycles and Mixed only, already turned into float32 npy of (V, I, T, SOC) for the LSTM | every test type, one independent series (CSV) per raw file, plus the excluded data |
| Full capacity of 567 | borrowed the cold capacity of 571, 2.4851 (final 0.2 %) | 2.6332 (final 5.8 %) |
| Full capacity of 571 | its own cold capacity, 2.4851 (final 5.0 %) | 2.5953 (final 9.0 %) |
| Full capacity of 557 | its own cold capacity, 2.5004 (final 5.0 %) | 2.6197 (final 9.3 %) |
| 590 / 601 / 604 / 582 | borrowed the nearest Cap_1C (final 5.9 / 4.6 / 5.5 / 5.2 %) | each its own cutoff capacity (final 5.0 % each) |
| 602_Mixed4 | 4160 frozen records left at the end of part1 | frozen records removed |
| 571_Mixed6 | the 10.4 h hidden pause was not found | split at the pause |
| 582_LA92 | first 1945 rows dropped | only the first run of 11/25 and the duplicated rows dropped; 907 more rows kept |
| Repeated time stamps | kept | removed, the first one kept |
| Time | not stored | time_s, plus the original local time and UTC (manifest) |
