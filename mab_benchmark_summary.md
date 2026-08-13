# mAb de novo sequencing benchmark — setup, metrics, results

Working notes for the XA-Novo mAb head-to-head and the nine-species PLM ablation. Last touched 2026-06-06.

## What we are trying to show

Casanovo + multi-protease peptide-language-model (pepLM) prior produces longer, more accurate de novo contigs of antibody chains than: (a) plain Casanovo v5.0, and (b) Hu et al.'s XA-Novo (an antibody-finetuned Casanovo v3.2.0). We test this on the XA-Novo data (PXD060500) under the exact same metric tool (npysearch) and assembler (ALPS) so any gap is attributable to the model, not the scoring pipeline.

## Datasets

### XA-Novo mAb benchmark (PXD060500, iProX IPX0009706000)

| mAb     | species | ref HC (aa) | ref LC (aa) | spectra | role |
|---------|---------|-------------|-------------|---------|------|
| 2B4     | mouse   | 443         | 220         | 5 proteases | mono mAb |
| 36H6    | mouse   | 443         | 220         | 5 proteases | mono mAb |
| 85F7    | mouse   | 443         | 220         | 5 proteases | mono mAb |
| S2P6    | human   | 448         | 217         | 5 proteases | mono mAb |
| BD5514  | human   | 455         | 215         | 5 proteases | mono mAb (just converted from .raw → .mgf) |
| S2BDLH  | human   | (S2P6+BD5514+SA55 mix) | — | 5 proteases | 3-mAb mixture, for separation power |

Each mAb is digested with **5 proteases** (AspN, chymotrypsin, elastase, pepsin, trypsin) — one HCD MS/MS file per protease. Spectra raw files were downloaded from PXD060500 and converted with `~/thermo/ThermoRawFileParser -f 0`. Reference FASTAs are I/L-folded mature heavy + light chain sequences from the XA-Novo SP-MEGD_Fusion GitHub.

SA58 is the only XA-Novo mAb whose **raw spectra are not in PXD060500** — only their published PSM CSV is available. So SA58 is metric-only (their PSMs through our metric pipeline), not a same-data head-to-head.

### Nine-species PLM ablation (Mes et al. 2022 / 9-species benchmark)

9 species (human, mouse, yeast, *B. subtilis*, honeybee, tomato, cowpea, *M. mazei* archaeon, *C. endoloripes*) digested by trypsin. Each species has:
- `9s_<sp>_dnps.mztab` — Casanovo baseline
- `9s_<sp>_asymbnln.mztab` — same-species PLM, asymbnln fusion head (the +PP arm)
- `9s_<sp>_plmhuman_hybrid.mztab` — cross-species (human PLM transfer)

Used for the precision-coverage curves (`plot_scatter_precision.py`) and per-length / per-spectral-angle ablations (`plot_precision_by_length.py`, `plot_precision_by_sa.py`).

## Models compared (the "arms")

| arm                   | description                                                                                  | ckpt / env                           |
|-----------------------|----------------------------------------------------------------------------------------------|--------------------------------------|
| `vanilla`             | Stock Casanovo v5.0, no pepLM, no fusion head                                                | `casanovo_v5_0_0.ckpt` / `khsam` env |
| `xanovo_v3`           | XA-Novo's antibody-finetuned Casanovo v3.2.0 ckpt (`epoch=9-step=550000.ckpt`, Zenodo 17266057). Requires `khsam_casanovov3` env (Casanovo v3.x has incompatible state_dict shape vs v4+). | XA-Novo ckpt / `khsam_casanovov3` env |
| `germline_<species>`  | Casanovo v5 + multi-protease pepLM trained on **species-matched germline antibody sequences** (mouse for 2B4/36H6/85F7, human for S2P6/BD5514). Uses the `fusion_model.pth` (LN-cas + LN-plm) head. | v5 + germline pepLM / `khsam` env    |
| `multiplm_asymbnln`   | The "Casanovo + multi" deliverable: Casanovo v5 + multi-protease pepLM with the **asymbnln** (BN-cas + LN-plm) fusion head, trained on MassiveKB. This is what we expect to win.                                                  | v5 + asymbnln fusion / `khsam` env   |

All arms run with `n_beams=5`, `top_match=1`, `predict_batch_size=1024` (vanilla / germline / multiplm) or `512` (xanovo_v3). Per-(mAb, protease, arm) Lance dir isolation to avoid commit conflicts (one of the gotchas — Casanovo's default `lance_dir` is a single shared path and parallel inference races on it).

Slurm scripts: `xa_novo/run_casanovo_{vanilla,xanovo,germline}_{mouse,human}_mab_array.sh` for mAb arms; `scripts/rerun_9s_inference_*.slurm` for the 9-species arms.

## Inference → assembly → scoring pipeline

For one (mAb, arm) pair:

1. **Inference** — 5 Slurm tasks (one per protease) produce `casanovo_<mab>_<protease>_<arm>_<jobid>.mztab` under `xa_novo/PXD060500_<mab>/casanovo_results/`. Each mztab has one PSM row per MS/MS spectrum: predicted peptide, per-AA scores, mean score.

2. **Per-protease PSM gather** — `parse_mztab_for_alps()` reads each of the 5 mztabs and normalizes the peptide string (strips brackets, since ALPS reasons over residue identity). Output is a list of `(spectrum_id, peptide, mean_score)` tuples.

3. **ALPS-format CSV** — `write_alps_csv()` filters by `score_cutoff` (default `0.0` — i.e. no filtering for the 3-arm comparison) and writes columns `Spectrum Name, Casanovo Peptide, Casanovo Mean Score`.

4. **ALPS assembly** — `run_alps()` invokes `java -jar tools/ALPS/ALPS.jar <csv> <k> <c>` for **k ∈ {7, 8, 10}** with `c=20` (top 20 contigs per k). ALPS does **reference-free** k-mer overlap-graph assembly. Important: the assembler sees only peptide sequences, no chain labels, no FASTA — it has no way to separate HC from LC peptides. Output: one FASTA per k.

5. **Union across k** — `union_contigs()` collapses the three per-k FASTAs into one set, **deduplicated on the I/L-folded sequence** (so I vs L variants of the same contig don't double-count). Typically yields ~45-60 unique contigs per arm.

6. **Metric step** — for each chain (HC, LC) of the reference, we run **two** independent local aligners on the unioned contigs:
   - `npysearch_metrics(contigs, ref, min_identity=0.75, max_accepts=5)` — BLAST-like local alignment via `npysearch` (BLOSUM62-style), the same tool PowerNovo uses in their paper.
   - `beslic_metrics(contigs, ref, min_aligned=6, min_identity=0.50)` — Smith-Waterman via `Bio.Align.PairwiseAligner` with the identity matrix, gap_open=−11, gap_extend=−1, I/L folded. Stricter substitution model.

   The reference is **only used at this metric step**. Swapping the reference would change the score but not the contig set.

Drivers:
- `dnps_hybrid/assembly.py` — 2-arm comparison (vanilla / germline_<sp>) for single mAbs. Output: `alps_2arm_all/2arm_summary.tsv`.
- `xa_novo/assemble_ourmodel_s2p6.py`, `_s2bdlh.py` — `multiplm_asymbnln` arm, per-mAb.
- `dnps_hybrid/assembly.py` also contains the ALPS input, execution, FASTA, alignment-metric, and contig-union helpers.

## Metric definitions (what each column means)

Output columns in `3arm_summary.tsv`:

| column            | definition                                                                                                                                  |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `n_contigs`       | # contigs in the union across k=7,8,10 (post I/L dedup).                                                                                    |
| `n_mapped`        | # contigs whose local alignment to the reference chain passed thresholds (`min_identity=0.75` for npysearch).                                |
| `coverage`        | fraction of reference positions covered by at least one mapped contig alignment. Range [0, 1].                                              |
| `covered_positions` | absolute count of reference positions covered. `coverage = covered_positions / ref_len`.                                                   |
| `accuracy`        | weighted identity of mapped portions: `total_matches / total_aligned` summed across all mapped contig alignments.                            |
| `matches`         | sum of per-position matches across all mapped contig alignments.                                                                            |
| `aligned`         | sum of aligned positions across all mapped contig alignments.                                                                               |
| `longest_contig`  | **Raw** length of the longest contig in the mapped set. **Can be inflated by chimerism** — see caveat below.                                |
| `longest_aligned` | Length of the aligned portion of that longest contig vs *this specific chain's* reference. The trustworthy per-chain "longest stretch".     |

### The chimerism caveat

Because ALPS sees a mixed pool of HC + LC peptides without chain labels, it sometimes joins a HC k-mer to an LC k-mer when their last/first `k` residues happen to overlap. That produces a single chimeric contig containing both a HC-derived stretch and an LC-derived stretch.

This **does not bias coverage or accuracy** — npysearch aligns the chimera against each chain independently, and only the chain-relevant residues contribute to each chain's score. But it **does inflate `longest_contig`**: the same chimeric contig is selected as "longest mapped" for both HC and LC, even though only one stretch of it actually belongs to that chain.

Sanity-check tell: `longest_contig ≫ longest_aligned` ⇒ chimera. Example from the current S2P6 results:
- S2P6 / germline_human / HC: longest_contig=558, longest_aligned=454 (chimera; clean HC stretch is ~454 aa)
- S2P6 / germline_human / LC: longest_contig=558, longest_aligned=121 (same 558 aa contig; only 121 aa of it is LC)
- 2B4 / germline_mouse / HC: longest_contig=364, longest_aligned=366 (clean — barely any chimerism)

**When quoting "longest reconstructed HC/LC", use `longest_aligned`, not `longest_contig`.**

## Coverage thresholds and historical drift

- Both metrics drop contigs that don't pass identity / length floors. Defaults:
  - npysearch: `min_identity=0.75`, `max_accepts=5`
  - Beslic: `min_aligned=6`, `min_identity=0.50`
- Score cutoffs (per-AA mean score) on the input PSMs: `0.0` (default; no filtering) and `0.043` (Herceptin-calibrated AA-precision-95% cutoff, used in the in-house IgG1 work). The 3-arm benchmark currently uses `0.0` only.
- A **fixed bug** worth noting: the original `load_reference("HC")` implementation did a substring match on FASTA record IDs and silently included the DECOY chain, doubling the apparent ref length (446 → 892 aa). Numbers in the current TSV are post-fix.

## Current results — 3-arm comparison (cutoff 0.0)

Source: `xa_novo/alps_3arm_all/3arm_summary.tsv`, run 2026-06-06.

### Heavy chain

| mAb    | arm              | ref aa | cov%   | acc%  | longest_contig | longest_aligned |
|--------|------------------|-------:|-------:|------:|---------------:|----------------:|
| 2B4    | vanilla          | 443    | 98.65  | 96.05 | 195            | 195             |
| 2B4    | xanovo_v3        | 443    | 100.00 | 97.69 | 264            | 264             |
| 2B4    | germline_mouse   | 443    | 99.77  | 92.26 | **364**        | **366**         |
| BD5514 | vanilla          | 455    | 98.02  | 99.53 | 128            | 129             |
| BD5514 | xanovo_v3        | 455    | 93.85  | 97.96 | 113            | 113             |
| BD5514 | germline_human   | 455    | 100.00 | 92.72 | **205**        | **206**         |
| S2P6   | vanilla          | 448    | 100.00 | 93.66 | 259            | 155             |
| S2P6   | xanovo_v3        | 448    | 98.21  | 94.59 | 144            | 144             |
| S2P6   | germline_human   | 448    | 100.00 | 93.02 | 558 (chimera)  | **454**         |

### Light chain

| mAb    | arm              | ref aa | cov%   | acc%  | longest_contig | longest_aligned |
|--------|------------------|-------:|-------:|------:|---------------:|----------------:|
| 2B4    | vanilla          | 220    | 100.00 | 92.36 | 145            | 160             |
| 2B4    | xanovo_v3        | 220    | 100.00 | 96.29 | 160            | 160             |
| 2B4    | germline_mouse   | 220    | 100.00 | 96.26 | **220**        | **220**         |
| BD5514 | vanilla          | 215    | 100.00 | 97.26 | 91             | 91              |
| BD5514 | xanovo_v3        | 215    | 100.00 | 99.22 | 78             | 78              [[[[|
| BD5514 | germline_human   | 215    | 100.00 | 99.17 | **104**        | **104**         |
| S2P6   | vanilla          | 217    | 100.00 | 93.51 | 259            | 121             |
| S2P6   | xanovo_v3        | 217    | 100.00 | 93.40 | 372            | 121             |
| S2P6   | germline_human   | 217    | 100.00 | 96.12 | 558 (chimera)  | 121             |

### Take-aways

1. **Coverage is near-saturating across all arms.** Every mAb hits ≥98% reference coverage in HC, 100% in LC. So the discriminator is **not coverage** — it's contig length and accuracy.
2. **`germline_<species>` produces the longest contigs on the heavy chain** for 2B4 (366 aa vs 195/264), BD5514 (206 vs 129/113), and S2P6 (454 vs 155/144). Light chains saturate near full length (220, 104, 121 aa) — limited by the protease coverage, not the model.
3. **Accuracy on HC is a wash** between the three arms (90s%). The pepLM is buying contig length, not per-AA accuracy. Vanilla and xanovo_v3 are slightly more accurate on HC for some mAbs (e.g. BD5514 vanilla 99.53%, xanovo 97.96%, germline 92.72%) — accuracy / length trade-off worth noting.
4. **xanovo_v3 is not consistently better than vanilla.** It wins on 2B4 HC (264 vs 195) but loses on BD5514 HC (113 vs 128 / cov 93.85% vs 98.02%) and S2P6 HC longest (144 vs 155). Their antibody-finetuned ckpt does **not** dominate stock v5 once we control the metric and assembler.

## Status (still in flight)

- `multiplm_asymbnln` (our model) arm assemblies: done for S2P6, S2BDLH; not yet done for 2B4 / 36H6 / 85F7 (the mztabs exist, the per-mAb `assemble_ourmodel_*.py` driver does not). Once those land we have a 4-arm comparison for the same mAbs.
- `xanovo_v3` on 36H6 (2/5 proteases done) and 85F7 (0/5) is still running — the original 2h wall was too short for v3 inference; resubmitted with 6h wall as job `19143558`.
- `xanovo_v3` on BD5514: done. `vanilla` and `germline_human` for BD5514: done.
- 9-species cross-species asymbnln re-runs: bundle job `19142933` and per-species jobs `19143555/56/57` still in flight; mouse/honeybee/tomato `plmhuman_hybrid.mztab` symlinks re-pointed at the new asymbnln outputs.
- Krug 2020 BL plex 10 inference (LUAD-finetuned v5 ± pepLM, asymbnln fusion head): plex10.mgf split into 4 quarters (260,519 / 260,520 spectra each); 8 array tasks running as job `19143661` (4 parts × 2 arms).
- S2BDLH: only `multiplm_asymbnln` mztabs exist; vanilla / xanovo_v3 / germline_human were never set up for the mixture — open question whether we want them.

## Local paths

```
xa_novo/                                                  # all XA-Novo benchmark work
  epoch=9-step=550000.ckpt                                # XA-Novo v3 ckpt
  cfg_v3/config_xanovo_v3.yaml                            # XA-Novo v3 inference config
  PXD060500_<mAb>/                                        # per-mAb data
    raw/                                                  # Thermo .raw downloads
    mgf/                                                  # ThermoRawFileParser MGFs
    casanovo_results/                                     # all (arm, protease) mztabs + logs
    <mAb>_ref.fasta                                       # I/L-folded HC + LC reference
  alps_3arm_all/3arm_summary.tsv                          # 3-arm comparison output
  alps_s2p6/, alps_s2bdlh/, alps_sa58/                    # per-mAb our-model output
  run_casanovo_{vanilla,xanovo,germline}_*_array.sh       # Slurm array drivers
  assembly.py                                              # 3-arm driver
  assemble_ourmodel_*.py                                  # multiplm_asymbnln drivers
  assemble_xanovo_sa58.py                                 # SA58 PSM-only path

dnps_hybrid/assembly.py                                   # canonical driver and helpers
plot_scatter_precision.py                                 # 9-species PC + scatter
plot_precision_by_{length,sa}.py                          # 9-species ablations
run_krug_plex_inference.slurm                             # Krug plex inference

/s/project/denovo-prosit/SamKhan/dnps_hybrid/             # bulk data
  casanovo/fusion_model{,_asymbnln}.pth                   # 9-species fusion heads
  _exp/luad_casanovo_ft/                                  # LUAD-finetuned Casanovo + Krug
  <species>/results/9s_<sp>_*.mztab                       # 9-species inference outputs
```
