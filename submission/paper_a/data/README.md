# Data for the Paper-A reproducibility package

The package ships every **derived** artifact and every **input** the CPU
producers read, except the one raw database file, which is third-party and
redistributed by its authors.

## MetalCytoToxDB.csv (required by the fingerprint probe)

`molmetal/scripts/metal_activity_transfer_probe.py` reads the raw cytotoxicity
database from `$MOLMETAL_DATA_DIR/MetalCytoToxDB.csv`. The copy used for the
paper is 4.8 MB.

- Source: MetalCytoToxDB (Krasnov et al., *J. Med. Chem.* 2026, 69(8), 8838).
- The `reproduce.sh` wrapper sets `MOLMETAL_DATA_DIR` to `<pkg>/.data/molmetal`
  by default; place the file there, or export `MOLMETAL_DATA_DIR` yourself.

    mkdir -p .data/molmetal
    cp /path/to/MetalCytoToxDB.csv .data/molmetal/MetalCytoToxDB.csv

Without this file the five `metal_activity_transfer*` entries cannot run; the
driver reports them as FAIL (producer exit != 0), which is the intended signal.

## tmQM (only for the GPU bond-perception retrain)

The tmQM corpus is the pretraining source for Claim 2 and the ground truth for
the bond-perception reference. It is **not** needed for the CPU subset: the
three `tmqm_bond_perception*` reports the synthesis reads are shipped under
`molmetal/reports/`. A GPU retrain of those reports needs the corpus at
`$MOLMETAL_TMQM_DIR`; see the tmQM distribution (Tudor et al., 2020).

## MoleculeNet (only for the GPU MoleculeNet retrain)

The `moleculenet_auc_dmpnn_5seed` entry (GPU, ~hours) downloads bbbp / bace /
hiv / clintox via its own loader; nothing is shipped here for it.
