# Result workbooks

- `l1_hazard_perf_rep3.xlsx` - median of 3 runs, occupancy-scaled parameters.
  This is the authoritative one; the numbers in the top-level README come from it.
- `l1_hazard_perf_occ.xlsx`  - single run, occupancy-scaled parameters.
- `l1_hazard_perf.xlsx`      - single run, fixed gfx1250 parameters (128/1/256).

Each workbook has one sheet per category plus a Summary sheet whose last table
counts improvements and regressions. Time and latency columns are
lower-is-better; the Summary table already accounts for direction, a raw sort on
`change_%` does not.

Single-run workbooks are kept only to show how much the noise moved the totals:
single run gave 52 improvements against 53 regressions, medians give 35 against 71.
