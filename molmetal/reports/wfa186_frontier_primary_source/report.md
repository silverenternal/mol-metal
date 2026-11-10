# Wf-A186 -- PRIMARY-SOURCE VERIFICATION OF THE 92.7-94 % FRONTIER BAND

* wave: **Wf-A186**, dated **2026-11-01**
* producer: `molmetal/reports/wfa186_frontier_primary_source/tools/build.py`
* machine artefact: `primary_source_verification.json` (same run, same data)
* GPU: **none -- CPU-only, 0 CUDA calls, 0 GPU allocations**

Every numeral below was parsed out of the pinned bytes at build time and carries a `file:line` locator plus the sha256 of the bytes it came from. Nothing is transcribed from memory or from another document.

## 0. THE RETRIEVAL LEDGER

| arXiv | status | e-print bytes | sha256 | retrieved (UTC) | live == cached |
|---|---|---|---|---|---|
| `2305.01140v1` | 200 | 655,684 | `b112f892ca89cd0e371f8bd9e4882d47dae6e5f384ad2b51ba21ace9e59cac23` | 2026-11-01T13:05:30Z | bytes **True**, sha256 **True** |
| `2603.27113v1` | 200 | 504,186 | `87c1788e9e9826cfbdf4e0788feed1a796c8a50218eecc6905df771f10c9c329` | 2026-11-01T13:05:30Z | bytes **True**, sha256 **True** |
| `2609.08333v4` | 200 | 8,508,562 | `e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224` | 2026-11-01T13:05:31Z | bytes **True**, sha256 **True** |

* `2305.01140v1` -- Geometric Latent Diffusion Models for 3D Molecule Generation (Minkai Xu, Alexander S. Powers, Ron O. Dror, Stefano Ermon, Jure Leskovec) -- icml2023.tex:88, 14 files in the tarball, https://arxiv.org/e-print/2305.01140v1 -> https://arxiv.org/src/2305.01140v1
* `2603.27113v1` -- Hierarchy-Guided Topology Latent Flow for Molecular Graph Generation (Urvi Awasthi, Alexander Lobo, Leonid Zhukov) -- iclr2025_conference.tex:25, 19 files in the tarball, https://arxiv.org/e-print/2603.27113v1 -> https://arxiv.org/src/2603.27113v1
* `2609.08333v4` -- Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules (Weichi Yao^1, Cameron Gruich^2, Bryan R. Goldsmith^2, Yixin Wang^3) -- main.tex:121, 20 files in the tarball, https://arxiv.org/e-print/2609.08333v4 -> https://arxiv.org/src/2609.08333v4

**The live re-fetch MATCHES the Wf-A175 cached receipt byte-for-byte and sha256-for-sha256 on all three papers.** One discrepancy, fully explained: the Wf-A175 receipt's per-file `file_bytes` field is a UTF-8 **character** count, not a byte count -- it equals `len(raw.decode('utf-8'))` for every file where the two differ, and equals `len(raw)` for every file where they agree. The e-print-level byte count and sha256 are byte-exact and match.

## 1. THE TABLE CELLS, AS THE BYTES PRINT THEM

### 2305.01140v1

| table | row label | column (group) | printed value | locator |
|---|---|---|---|---|
| tab:qm9_results | Data | QM9 / Atom Sta (%) | 99.0 | `icml2023.tex:480` |
| tab:qm9_results | Data | QM9 / Mol Sta (%) | 95.2 | `icml2023.tex:480` |
| tab:qm9_results | Data | QM9 / Valid (%) | 97.7 | `icml2023.tex:480` |
| tab:qm9_results | Data | QM9 / Valid & Unique (%) | 97.7 | `icml2023.tex:480` |
| tab:qm9_results | Data | DRUG / Atom Sta (%) | 86.5 | `icml2023.tex:480` |
| tab:qm9_results | Data | DRUG / Valid (%) | 99.9 | `icml2023.tex:480` |
| tab:qm9_results | ENF | QM9 / Atom Sta (%) | 85.0 | `icml2023.tex:482` |
| tab:qm9_results | ENF | QM9 / Mol Sta (%) | 4.9 | `icml2023.tex:482` |
| tab:qm9_results | ENF | QM9 / Valid (%) | 40.2 | `icml2023.tex:482` |
| tab:qm9_results | ENF | QM9 / Valid & Unique (%) | 39.4 | `icml2023.tex:482` |
| tab:qm9_results | ENF | DRUG / Atom Sta (%) | - | `icml2023.tex:482` |
| tab:qm9_results | ENF | DRUG / Valid (%) | - | `icml2023.tex:482` |
| tab:qm9_results | G-Schnet | QM9 / Atom Sta (%) | 95.7 | `icml2023.tex:483` |
| tab:qm9_results | G-Schnet | QM9 / Mol Sta (%) | 68.1 | `icml2023.tex:483` |
| tab:qm9_results | G-Schnet | QM9 / Valid (%) | 85.5 | `icml2023.tex:483` |
| tab:qm9_results | G-Schnet | QM9 / Valid & Unique (%) | 80.3 | `icml2023.tex:483` |
| tab:qm9_results | G-Schnet | DRUG / Atom Sta (%) | - | `icml2023.tex:483` |
| tab:qm9_results | G-Schnet | DRUG / Valid (%) | - | `icml2023.tex:483` |
| tab:qm9_results | GDM | QM9 / Atom Sta (%) | 97.0 | `icml2023.tex:484` |
| tab:qm9_results | GDM | QM9 / Mol Sta (%) | 63.2 | `icml2023.tex:484` |
| tab:qm9_results | GDM | QM9 / Valid (%) | - | `icml2023.tex:484` |
| tab:qm9_results | GDM | QM9 / Valid & Unique (%) | - | `icml2023.tex:484` |
| tab:qm9_results | GDM | DRUG / Atom Sta (%) | 75.0 | `icml2023.tex:484` |
| tab:qm9_results | GDM | DRUG / Valid (%) | 90.8 | `icml2023.tex:484` |
| tab:qm9_results | GDM-aug | QM9 / Atom Sta (%) | 97.6 | `icml2023.tex:485` |
| tab:qm9_results | GDM-aug | QM9 / Mol Sta (%) | 71.6 | `icml2023.tex:485` |
| tab:qm9_results | GDM-aug | QM9 / Valid (%) | 90.4 | `icml2023.tex:485` |
| tab:qm9_results | GDM-aug | QM9 / Valid & Unique (%) | 89.5 | `icml2023.tex:485` |
| tab:qm9_results | GDM-aug | DRUG / Atom Sta (%) | 77.7 | `icml2023.tex:485` |
| tab:qm9_results | GDM-aug | DRUG / Valid (%) | 91.8 | `icml2023.tex:485` |
| tab:qm9_results | EDM | QM9 / Atom Sta (%) | 98.7 | `icml2023.tex:486` |
| tab:qm9_results | EDM | QM9 / Mol Sta (%) | 82.0 | `icml2023.tex:486` |
| tab:qm9_results | EDM | QM9 / Valid (%) | 91.9 | `icml2023.tex:486` |
| tab:qm9_results | EDM | QM9 / Valid & Unique (%) | 90.7 | `icml2023.tex:486` |
| tab:qm9_results | EDM | DRUG / Atom Sta (%) | 81.3 | `icml2023.tex:486` |
| tab:qm9_results | EDM | DRUG / Valid (%) | 92.6 | `icml2023.tex:486` |
| tab:qm9_results | EDM-Bridge | QM9 / Atom Sta (%) | 98.8 | `icml2023.tex:487` |
| tab:qm9_results | EDM-Bridge | QM9 / Mol Sta (%) | 84.6 | `icml2023.tex:487` |
| tab:qm9_results | EDM-Bridge | QM9 / Valid (%) | 92.0* | `icml2023.tex:487` |
| tab:qm9_results | EDM-Bridge | QM9 / Valid & Unique (%) | 90.7 | `icml2023.tex:487` |
| tab:qm9_results | EDM-Bridge | DRUG / Atom Sta (%) | 82.4 | `icml2023.tex:487` |
| tab:qm9_results | EDM-Bridge | DRUG / Valid (%) | 92.8* | `icml2023.tex:487` |
| tab:qm9_results | GraphLDM | QM9 / Atom Sta (%) | 97.2 | `icml2023.tex:489` |
| tab:qm9_results | GraphLDM | QM9 / Mol Sta (%) | 70.5 | `icml2023.tex:489` |
| tab:qm9_results | GraphLDM | QM9 / Valid (%) | 83.6 | `icml2023.tex:489` |
| tab:qm9_results | GraphLDM | QM9 / Valid & Unique (%) | 82.7 | `icml2023.tex:489` |
| tab:qm9_results | GraphLDM | DRUG / Atom Sta (%) | 76.2 | `icml2023.tex:489` |
| tab:qm9_results | GraphLDM | DRUG / Valid (%) | 97.2 | `icml2023.tex:489` |
| tab:qm9_results | GraphLDM-aug | QM9 / Atom Sta (%) | 97.9 | `icml2023.tex:490` |
| tab:qm9_results | GraphLDM-aug | QM9 / Mol Sta (%) | 78.7 | `icml2023.tex:490` |
| tab:qm9_results | GraphLDM-aug | QM9 / Valid (%) | 90.5 | `icml2023.tex:490` |
| tab:qm9_results | GraphLDM-aug | QM9 / Valid & Unique (%) | 89.5 | `icml2023.tex:490` |
| tab:qm9_results | GraphLDM-aug | DRUG / Atom Sta (%) | 79.6 | `icml2023.tex:490` |
| tab:qm9_results | GraphLDM-aug | DRUG / Valid (%) | 98.0 | `icml2023.tex:490` |
| tab:qm9_results | GeoLDM | QM9 / Atom Sta (%) | 98.9 ± 0.1 | `icml2023.tex:491` |
| tab:qm9_results | GeoLDM | QM9 / Mol Sta (%) | 89.4 ± 0.5 | `icml2023.tex:491` |
| tab:qm9_results | GeoLDM | QM9 / Valid (%) | 93.8 ± 0.4 | `icml2023.tex:491` |
| tab:qm9_results | GeoLDM | QM9 / Valid & Unique (%) | 92.7 ± 0.5 | `icml2023.tex:491` |
| tab:qm9_results | GeoLDM | DRUG / Atom Sta (%) | 84.4 | `icml2023.tex:491` |
| tab:qm9_results | GeoLDM | DRUG / Valid (%) | 99.3 | `icml2023.tex:491` |
| app:tab:ablation | GeoLDM (k=1, KL-reg)* | Atom Sta (%) | 95.45 | `icml2023.tex:891` |
| app:tab:ablation | GeoLDM (k=1, KL-reg)* | Mol Sta (%) | 40.7 | `icml2023.tex:891` |
| app:tab:ablation | GeoLDM (k=1, KL-reg)* | Valid (%) | 83.7 | `icml2023.tex:891` |
| app:tab:ablation | GeoLDM (k=1, KL-reg)* | Valid & Unique (%) | 83.5 | `icml2023.tex:891` |
| app:tab:ablation | GeoLDM (k=16, ES-reg) | Atom Sta (%) | 98.6 | `icml2023.tex:892` |
| app:tab:ablation | GeoLDM (k=16, ES-reg) | Mol Sta (%) | 86.0 | `icml2023.tex:892` |
| app:tab:ablation | GeoLDM (k=16, ES-reg) | Valid (%) | 92.4 | `icml2023.tex:892` |
| app:tab:ablation | GeoLDM (k=16, ES-reg) | Valid & Unique (%) | 92.2 | `icml2023.tex:892` |
| app:tab:ablation | GeoLDM (k=8, ES-reg) | Atom Sta (%) | 98.7 | `icml2023.tex:893` |
| app:tab:ablation | GeoLDM (k=8, ES-reg) | Mol Sta (%) | 87.1 | `icml2023.tex:893` |
| app:tab:ablation | GeoLDM (k=8, ES-reg) | Valid (%) | 92.1 | `icml2023.tex:893` |
| app:tab:ablation | GeoLDM (k=8, ES-reg) | Valid & Unique (%) | 92.0 | `icml2023.tex:893` |
| app:tab:ablation | GeoLDM (k=4, ES-reg) | Atom Sta (%) | 98.8 | `icml2023.tex:894` |
| app:tab:ablation | GeoLDM (k=4, ES-reg) | Mol Sta (%) | 87.4 | `icml2023.tex:894` |
| app:tab:ablation | GeoLDM (k=4, ES-reg) | Valid (%) | 92.6 | `icml2023.tex:894` |
| app:tab:ablation | GeoLDM (k=4, ES-reg) | Valid & Unique (%) | 92.5 | `icml2023.tex:894` |
| app:tab:ablation | GeoLDM (k=1, ES-reg) | Atom Sta (%) | 98.9 ± 0.1 | `icml2023.tex:895` |
| app:tab:ablation | GeoLDM (k=1, ES-reg) | Mol Sta (%) | 89.4 ± 0.5 | `icml2023.tex:895` |
| app:tab:ablation | GeoLDM (k=1, ES-reg) | Valid (%) | 93.8 ± 0.4 | `icml2023.tex:895` |
| app:tab:ablation | GeoLDM (k=1, ES-reg) | Valid & Unique (%) | 92.7 ± 0.5 | `icml2023.tex:895` |

### 2603.27113v1

| table | row label | column (group) | printed value | locator |
|---|---|---|---|---|
| tab:qm9_main | E-NF | AS (%)↑ | 85.0 | `tables/qm9_benchmarks_og.tex:29` |
| tab:qm9_main | E-NF | MS (%)↑ | 4.9 | `tables/qm9_benchmarks_og.tex:29` |
| tab:qm9_main | E-NF | Val (%)↑ | 40.2 | `tables/qm9_benchmarks_og.tex:29` |
| tab:qm9_main | E-NF | Val&Uniq (%)↑ | 39.4 | `tables/qm9_benchmarks_og.tex:29` |
| tab:qm9_main | G-Schnet | AS (%)↑ | 95.7 | `tables/qm9_benchmarks_og.tex:30` |
| tab:qm9_main | G-Schnet | MS (%)↑ | 68.1 | `tables/qm9_benchmarks_og.tex:30` |
| tab:qm9_main | G-Schnet | Val (%)↑ | 85.5 | `tables/qm9_benchmarks_og.tex:30` |
| tab:qm9_main | G-Schnet | Val&Uniq (%)↑ | 80.3 | `tables/qm9_benchmarks_og.tex:30` |
| tab:qm9_main | GDM | AS (%)↑ | 97.0 | `tables/qm9_benchmarks_og.tex:31` |
| tab:qm9_main | GDM | MS (%)↑ | 63.2 | `tables/qm9_benchmarks_og.tex:31` |
| tab:qm9_main | GDM | Val (%)↑ | -- | `tables/qm9_benchmarks_og.tex:31` |
| tab:qm9_main | GDM | Val&Uniq (%)↑ | -- | `tables/qm9_benchmarks_og.tex:31` |
| tab:qm9_main | GDM-aug | AS (%)↑ | 97.6 | `tables/qm9_benchmarks_og.tex:32` |
| tab:qm9_main | GDM-aug | MS (%)↑ | 71.6 | `tables/qm9_benchmarks_og.tex:32` |
| tab:qm9_main | GDM-aug | Val (%)↑ | 90.4 | `tables/qm9_benchmarks_og.tex:32` |
| tab:qm9_main | GDM-aug | Val&Uniq (%)↑ | 89.5 | `tables/qm9_benchmarks_og.tex:32` |
| tab:qm9_main | EDM | AS (%)↑ | 98.7±0.1 | `tables/qm9_benchmarks_og.tex:33` |
| tab:qm9_main | EDM | MS (%)↑ | 82.0±0.4 | `tables/qm9_benchmarks_og.tex:33` |
| tab:qm9_main | EDM | Val (%)↑ | 91.9±0.5 | `tables/qm9_benchmarks_og.tex:33` |
| tab:qm9_main | EDM | Val&Uniq (%)↑ | 90.7±0.6 | `tables/qm9_benchmarks_og.tex:33` |
| tab:qm9_main | Bridge | AS (%)↑ | 98.7±0.1 | `tables/qm9_benchmarks_og.tex:34` |
| tab:qm9_main | Bridge | MS (%)↑ | 81.8±0.2 | `tables/qm9_benchmarks_og.tex:34` |
| tab:qm9_main | Bridge | Val (%)↑ | -- | `tables/qm9_benchmarks_og.tex:34` |
| tab:qm9_main | Bridge | Val&Uniq (%)↑ | 90.2 | `tables/qm9_benchmarks_og.tex:34` |
| tab:qm9_main | Bridge+Force | AS (%)↑ | 98.8±0.1 | `tables/qm9_benchmarks_og.tex:35` |
| tab:qm9_main | Bridge+Force | MS (%)↑ | 84.6±0.3 | `tables/qm9_benchmarks_og.tex:35` |
| tab:qm9_main | Bridge+Force | Val (%)↑ | 92.0 | `tables/qm9_benchmarks_og.tex:35` |
| tab:qm9_main | Bridge+Force | Val&Uniq (%)↑ | 90.7 | `tables/qm9_benchmarks_og.tex:35` |
| tab:qm9_main | GraphLDM | AS (%)↑ | 97.2 | `tables/qm9_benchmarks_og.tex:36` |
| tab:qm9_main | GraphLDM | MS (%)↑ | 70.5 | `tables/qm9_benchmarks_og.tex:36` |
| tab:qm9_main | GraphLDM | Val (%)↑ | 83.6 | `tables/qm9_benchmarks_og.tex:36` |
| tab:qm9_main | GraphLDM | Val&Uniq (%)↑ | 82.7 | `tables/qm9_benchmarks_og.tex:36` |
| tab:qm9_main | GraphLDM-aug | AS (%)↑ | 97.9 | `tables/qm9_benchmarks_og.tex:37` |
| tab:qm9_main | GraphLDM-aug | MS (%)↑ | 78.7 | `tables/qm9_benchmarks_og.tex:37` |
| tab:qm9_main | GraphLDM-aug | Val (%)↑ | 90.5 | `tables/qm9_benchmarks_og.tex:37` |
| tab:qm9_main | GraphLDM-aug | Val&Uniq (%)↑ | 89.5 | `tables/qm9_benchmarks_og.tex:37` |
| tab:qm9_main | GeoLDM | AS (%)↑ | 98.9±0.1* | `tables/qm9_benchmarks_og.tex:38` |
| tab:qm9_main | GeoLDM | MS (%)↑ | 89.4±0.5* | `tables/qm9_benchmarks_og.tex:38` |
| tab:qm9_main | GeoLDM | Val (%)↑ | 93.6±0.2 | `tables/qm9_benchmarks_og.tex:38` |
| tab:qm9_main | GeoLDM | Val&Uniq (%)↑ | 92.7±0.5 | `tables/qm9_benchmarks_og.tex:38` |
| tab:qm9_main | GCDM | AS (%)↑ | 98.7±0.1 | `tables/qm9_benchmarks_og.tex:39` |
| tab:qm9_main | GCDM | MS (%)↑ | 85.7±0.4 | `tables/qm9_benchmarks_og.tex:39` |
| tab:qm9_main | GCDM | Val (%)↑ | 94.8±0.2* | `tables/qm9_benchmarks_og.tex:39` |
| tab:qm9_main | GCDM | Val&Uniq (%)↑ | 93.3±0.0* | `tables/qm9_benchmarks_og.tex:39` |
| tab:qm9_main | HLTF (ours) | AS (%)↑ | 98.8±0.1 | `tables/qm9_benchmarks_og.tex:41` |
| tab:qm9_main | HLTF (ours) | MS (%)↑ | 86.8±0.3 | `tables/qm9_benchmarks_og.tex:41` |
| tab:qm9_main | HLTF (ours) | Val (%)↑ | 93.8±0.6 | `tables/qm9_benchmarks_og.tex:41` |
| tab:qm9_main | HLTF (ours) | Val&Uniq (%)↑ | 92.9±0.2 | `tables/qm9_benchmarks_og.tex:41` |
| tab:qm9_pb | AS (%) ↑ | GeoLDM | 98.9±0.1* | `tables/qm9_benchmarks_og.tex:61` |
| tab:qm9_pb | AS (%) ↑ | GCDM | 98.7±0.1 | `tables/qm9_benchmarks_og.tex:61` |
| tab:qm9_pb | AS (%) ↑ | HLTF (ours) | 98.8±0.1 | `tables/qm9_benchmarks_og.tex:61` |
| tab:qm9_pb | MS (%) ↑ | GeoLDM | 89.4±0.5* | `tables/qm9_benchmarks_og.tex:62` |
| tab:qm9_pb | MS (%) ↑ | GCDM | 85.7±0.4 | `tables/qm9_benchmarks_og.tex:62` |
| tab:qm9_pb | MS (%) ↑ | HLTF (ours) | 86.8±0.3 | `tables/qm9_benchmarks_og.tex:62` |
| tab:qm9_pb | Val (%) ↑ | GeoLDM | 93.6±0.2 | `tables/qm9_benchmarks_og.tex:63` |
| tab:qm9_pb | Val (%) ↑ | GCDM | 94.8±0.2* | `tables/qm9_benchmarks_og.tex:63` |
| tab:qm9_pb | Val (%) ↑ | HLTF (ours) | 93.8±0.6 | `tables/qm9_benchmarks_og.tex:63` |
| tab:qm9_pb | Val&Uniq (%) ↑ | GeoLDM | 92.7±0.5 | `tables/qm9_benchmarks_og.tex:64` |
| tab:qm9_pb | Val&Uniq (%) ↑ | GCDM | 93.3±0.0* | `tables/qm9_benchmarks_og.tex:64` |
| tab:qm9_pb | Val&Uniq (%) ↑ | HLTF (ours) | 92.9±0.2 | `tables/qm9_benchmarks_og.tex:64` |
| tab:qm9_pb | Novel (%) ↑ | GeoLDM | 53.5±0.6 | `tables/qm9_benchmarks_og.tex:65` |
| tab:qm9_pb | Novel (%) ↑ | GCDM | 58.7±0.5* | `tables/qm9_benchmarks_og.tex:65` |
| tab:qm9_pb | Novel (%) ↑ | HLTF (ours) | 56±0.7 | `tables/qm9_benchmarks_og.tex:65` |
| tab:qm9_pb | PB-Valid (%) ↑ | GeoLDM | 93.1±0.4 | `tables/qm9_benchmarks_og.tex:66` |
| tab:qm9_pb | PB-Valid (%) ↑ | GCDM | 91.9±0.5 | `tables/qm9_benchmarks_og.tex:66` |
| tab:qm9_pb | PB-Valid (%) ↑ | HLTF (ours) | 94±0.5* | `tables/qm9_benchmarks_og.tex:66` |
| tab:geom_main | EQGAT-diff | Valid (%) | 59.7 | `tables/geom_drugs_results.tex:14` |
| tab:geom_main | EQGAT-diff | V&U (%) | 59.7 | `tables/geom_drugs_results.tex:14` |
| tab:geom_main | EQGAT-diff | V&U&N (%) | 59.5 | `tables/geom_drugs_results.tex:14` |
| tab:geom_main | EQGAT-diff +PP | Valid (%) | 84.2 | `tables/geom_drugs_results.tex:15` |
| tab:geom_main | EQGAT-diff +PP | V&U (%) | 84.2 | `tables/geom_drugs_results.tex:15` |
| tab:geom_main | EQGAT-diff +PP | V&U&N (%) | 84.0 | `tables/geom_drugs_results.tex:15` |
| tab:geom_main | FlowMol | Valid (%) | 59.8 | `tables/geom_drugs_results.tex:16` |
| tab:geom_main | FlowMol | V&U (%) | 59.8 | `tables/geom_drugs_results.tex:16` |
| tab:geom_main | FlowMol | V&U&N (%) | 59.7 | `tables/geom_drugs_results.tex:16` |
| tab:geom_main | FlowMol +PP | Valid (%) | 84.2 | `tables/geom_drugs_results.tex:17` |
| tab:geom_main | FlowMol +PP | V&U (%) | 84.2 | `tables/geom_drugs_results.tex:17` |
| tab:geom_main | FlowMol +PP | V&U&N (%) | 84.1 | `tables/geom_drugs_results.tex:17` |
| tab:geom_main | GCDM | Valid (%) | 0.2 | `tables/geom_drugs_results.tex:18` |
| tab:geom_main | GCDM | V&U (%) | 0.2 | `tables/geom_drugs_results.tex:18` |
| tab:geom_main | GCDM | V&U&N (%) | 0.2 | `tables/geom_drugs_results.tex:18` |
| tab:geom_main | GCDM +PP | Valid (%) | 95.2 | `tables/geom_drugs_results.tex:19` |
| tab:geom_main | GCDM +PP | V&U (%) | 95.2 | `tables/geom_drugs_results.tex:19` |
| tab:geom_main | GCDM +PP | V&U&N (%) | 95.2 | `tables/geom_drugs_results.tex:19` |
| tab:geom_main | GeoLDM | Valid (%) | 2.9 | `tables/geom_drugs_results.tex:20` |
| tab:geom_main | GeoLDM | V&U (%) | 2.9 | `tables/geom_drugs_results.tex:20` |
| tab:geom_main | GeoLDM | V&U&N (%) | 2.9 | `tables/geom_drugs_results.tex:20` |
| tab:geom_main | GeoLDM +PP | Valid (%) | 69.6 | `tables/geom_drugs_results.tex:21` |
| tab:geom_main | GeoLDM +PP | V&U (%) | 69.3 | `tables/geom_drugs_results.tex:21` |
| tab:geom_main | GeoLDM +PP | V&U&N (%) | 69.3 | `tables/geom_drugs_results.tex:21` |
| tab:geom_main | SemlaFlow | Valid (%) | 87.5 * | `tables/geom_drugs_results.tex:22` |
| tab:geom_main | SemlaFlow | V&U (%) | 87.4 * | `tables/geom_drugs_results.tex:22` |
| tab:geom_main | SemlaFlow | V&U&N (%) | 87.0 * | `tables/geom_drugs_results.tex:22` |
| tab:geom_main | SemlaFlow +PP | Valid (%) | 93.1 * | `tables/geom_drugs_results.tex:23` |
| tab:geom_main | SemlaFlow +PP | V&U (%) | 92.9 * | `tables/geom_drugs_results.tex:23` |
| tab:geom_main | SemlaFlow +PP | V&U&N (%) | 92.4 * | `tables/geom_drugs_results.tex:23` |
| tab:geom_main | HLTF (ours) | Valid (%) | 85.5 | `tables/geom_drugs_results.tex:25` |
| tab:geom_main | HLTF (ours) | V&U (%) | 85.4 | `tables/geom_drugs_results.tex:25` |
| tab:geom_main | HLTF (ours) | V&U&N (%) | 85.0 | `tables/geom_drugs_results.tex:25` |
| tab:geom_main | HLTF (ours) +PP | Valid (%) | 92.2 | `tables/geom_drugs_results.tex:26` |
| tab:geom_main | HLTF (ours) +PP | V&U (%) | 92.0 | `tables/geom_drugs_results.tex:26` |
| tab:geom_main | HLTF (ours) +PP | V&U&N (%) | 91.2 | `tables/geom_drugs_results.tex:26` |

### 2609.08333v4

| table | row label | column (group) | printed value | locator |
|---|---|---|---|---|
| tbl:pcqm4mv2_generation (a) | C. | UAE-3D | 99.60 | `src/appendix.tex:112-115` |
| tbl:pcqm4mv2_generation (a) | C. | FlowMol | 100.00 | `src/appendix.tex:112-115` |
| tbl:pcqm4mv2_generation (a) | C. | EF-TALFM | 92.98 | `src/appendix.tex:112-115` |
| tbl:pcqm4mv2_generation (a) | S. | UAE-3D | 83.84 | `src/appendix.tex:116-119` |
| tbl:pcqm4mv2_generation (a) | S. | FlowMol | 98.72 | `src/appendix.tex:116-119` |
| tbl:pcqm4mv2_generation (a) | S. | EF-TALFM | 98.25 | `src/appendix.tex:116-119` |
| tbl:pcqm4mv2_generation (a) | U.&S. | UAE-3D | 83.81 | `src/appendix.tex:120-123` |
| tbl:pcqm4mv2_generation (a) | U.&S. | FlowMol | 98.27 | `src/appendix.tex:120-123` |
| tbl:pcqm4mv2_generation (a) | U.&S. | EF-TALFM | 98.21 | `src/appendix.tex:120-123` |
| tbl:pcqm4mv2_generation (a) | N.&U.&S. | UAE-3D | 76.03 | `src/appendix.tex:124-127` |
| tbl:pcqm4mv2_generation (a) | N.&U.&S. | FlowMol | 69.83 | `src/appendix.tex:124-127` |
| tbl:pcqm4mv2_generation (a) | N.&U.&S. | EF-TALFM | 94.33 | `src/appendix.tex:124-127` |
| tbl:pcqm4mv2_generation (b) | Bond lengths | UAE-3D | 99.96 | `src/appendix.tex:146-149` |
| tbl:pcqm4mv2_generation (b) | Bond lengths | FlowMol | 100.00 | `src/appendix.tex:146-149` |
| tbl:pcqm4mv2_generation (b) | Bond lengths | EF-TALFM | 99.79 | `src/appendix.tex:146-149` |
| tbl:pcqm4mv2_generation (b) | Bond angles | UAE-3D | 99.93 | `src/appendix.tex:150-153` |
| tbl:pcqm4mv2_generation (b) | Bond angles | FlowMol | 100.00 | `src/appendix.tex:150-153` |
| tbl:pcqm4mv2_generation (b) | Bond angles | EF-TALFM | 99.32 | `src/appendix.tex:150-153` |
| tbl:pcqm4mv2_generation (b) | Internal steric clash | UAE-3D | 99.64 | `src/appendix.tex:154-157` |
| tbl:pcqm4mv2_generation (b) | Internal steric clash | FlowMol | 99.90 | `src/appendix.tex:154-157` |
| tbl:pcqm4mv2_generation (b) | Internal steric clash | EF-TALFM | 97.22 | `src/appendix.tex:154-157` |
| tbl:pcqm4mv2_generation (b) | Aromatic-ring flatness | UAE-3D | 100.00 | `src/appendix.tex:158-161` |
| tbl:pcqm4mv2_generation (b) | Aromatic-ring flatness | FlowMol | 100.00 | `src/appendix.tex:158-161` |
| tbl:pcqm4mv2_generation (b) | Aromatic-ring flatness | EF-TALFM | 99.93 | `src/appendix.tex:158-161` |
| tbl:pcqm4mv2_generation (b) | Double-bond flatness | UAE-3D | 99.91 | `src/appendix.tex:162-165` |
| tbl:pcqm4mv2_generation (b) | Double-bond flatness | FlowMol | 100.00 | `src/appendix.tex:162-165` |
| tbl:pcqm4mv2_generation (b) | Double-bond flatness | EF-TALFM | 98.09 | `src/appendix.tex:162-165` |
| tbl:pcqm4mv2_generation (b) | Internal energy | UAE-3D | 99.39 | `src/appendix.tex:166-169` |
| tbl:pcqm4mv2_generation (b) | Internal energy | FlowMol | 99.83 | `src/appendix.tex:166-169` |
| tbl:pcqm4mv2_generation (b) | Internal energy | EF-TALFM | 97.69 | `src/appendix.tex:166-169` |
| tbl:pcqm4mv2_generation (b) | All PoseBusters checks | UAE-3D | 98.91 | `src/appendix.tex:171` |
| tbl:pcqm4mv2_generation (b) | All PoseBusters checks | FlowMol | 99.73 | `src/appendix.tex:171` |
| tbl:pcqm4mv2_generation (b) | All PoseBusters checks | EF-TALFM | 93.13 | `src/appendix.tex:171` |
| tbl:pcqm4mv2_generation_additional | C. | UAE-3D^* | 99.60 | `src/appendix.tex:196-202` |
| tbl:pcqm4mv2_generation_additional | C. | UAE-3D^§ | 99.71 | `src/appendix.tex:196-202` |
| tbl:pcqm4mv2_generation_additional | C. | FlowMol^† | 100.00 | `src/appendix.tex:196-202` |
| tbl:pcqm4mv2_generation_additional | C. | FlowMol^‡ | 100.00 | `src/appendix.tex:196-202` |
| tbl:pcqm4mv2_generation_additional | C. | FlowMol^ʷ | 100.00 | `src/appendix.tex:196-202` |
| tbl:pcqm4mv2_generation_additional | C. | EF-TALFM | 92.98 | `src/appendix.tex:196-202` |
| tbl:pcqm4mv2_generation_additional | S. | UAE-3D^* | 83.84 | `src/appendix.tex:203-209` |
| tbl:pcqm4mv2_generation_additional | S. | UAE-3D^§ | 85.89 | `src/appendix.tex:203-209` |
| tbl:pcqm4mv2_generation_additional | S. | FlowMol^† | 98.72 | `src/appendix.tex:203-209` |
| tbl:pcqm4mv2_generation_additional | S. | FlowMol^‡ | 99.00 | `src/appendix.tex:203-209` |
| tbl:pcqm4mv2_generation_additional | S. | FlowMol^ʷ | 98.57 | `src/appendix.tex:203-209` |
| tbl:pcqm4mv2_generation_additional | S. | EF-TALFM | 98.25 | `src/appendix.tex:203-209` |
| tbl:pcqm4mv2_generation_additional | U.&S. | UAE-3D^* | 83.81 | `src/appendix.tex:210-216` |
| tbl:pcqm4mv2_generation_additional | U.&S. | UAE-3D^§ | 85.88 | `src/appendix.tex:210-216` |
| tbl:pcqm4mv2_generation_additional | U.&S. | FlowMol^† | 98.27 | `src/appendix.tex:210-216` |
| tbl:pcqm4mv2_generation_additional | U.&S. | FlowMol^‡ | 98.49 | `src/appendix.tex:210-216` |
| tbl:pcqm4mv2_generation_additional | U.&S. | FlowMol^ʷ | 98.10 | `src/appendix.tex:210-216` |
| tbl:pcqm4mv2_generation_additional | U.&S. | EF-TALFM | 98.21 | `src/appendix.tex:210-216` |
| tbl:pcqm4mv2_generation_additional | N.&U.&S. | UAE-3D^* | 76.03 | `src/appendix.tex:217-223` |
| tbl:pcqm4mv2_generation_additional | N.&U.&S. | UAE-3D^§ | 76.62 | `src/appendix.tex:217-223` |
| tbl:pcqm4mv2_generation_additional | N.&U.&S. | FlowMol^† | 69.83 | `src/appendix.tex:217-223` |
| tbl:pcqm4mv2_generation_additional | N.&U.&S. | FlowMol^‡ | 65.45 | `src/appendix.tex:217-223` |
| tbl:pcqm4mv2_generation_additional | N.&U.&S. | FlowMol^ʷ | 66.50 | `src/appendix.tex:217-223` |
| tbl:pcqm4mv2_generation_additional | N.&U.&S. | EF-TALFM | 94.33 | `src/appendix.tex:217-223` |

## 2. THE FIVE FINDINGS

### F1 -- Is GeoLDM's Validity & Uniq 92.7 % GeoLDM's OWN published cell, with HLTF merely reproducing it identically?

**VERDICT: ESTABLISHED**

Verbatim quotes (method: read out of the `.tex` extracted from the live e-print tarball; each carries the sha256 of that `.tex` and of the e-print):

> `2305.01140v1` :: `icml2023.tex:491`
>
>     \rowcolor{lightgray} \textbf{\method} & \textbf{98.9} $\pm$ 0.1 & \textbf{89.4} $\pm$ 0.5 & \textbf{93.8} $\pm$ 0.4 & \textbf{92.7} $\pm$ 0.5 & \textbf{84.4} & \textbf{99.3} \\

> `2305.01140v1` :: `icml2023.tex:468`
>
> \caption{Results of atom stability, molecule stability, validity, and validity$\times$uniqueness. A higher number indicates a better generation quality. Metrics are calculated with 10000 samples generated from each model. On QM9, we run the evaluation for 3 times and report the derivation. Note that, for DRUG dataset, molecule stability and uniqueness metric are omitted since they are nearly $0\%$ and $100\%$ respectively for all the methods. Compared with previous methods, the latent space with both invariant and equivariant variables enables \method to achieve up to 7\% improvement for the validity of large molecule generation.

> `2305.01140v1` :: `icml2023.tex:549`
>
> We generate $10,000$ samples from each method to calculate the above metrics, and the results are reported in \cref{tab:qm9_results}. As shown in the table, \method outperforms competitive baseline methods on all metrics with an obvious margin. 

> `2603.27113v1` :: `tables/qm9_benchmarks_og.tex:38`
>
> GeoLDM & $98.9\!\pm\!0.1$* & $89.4\!\pm\!0.5$* & $93.6\!\pm\!0.2$ & $92.7\!\pm\!0.5$ \\

> `2603.27113v1` :: `tables/qm9_benchmarks_og.tex:64`
>
> Val\&Uniq (\%) $\uparrow$     & $92.7\pm0.5$       & $93.3\pm0.0$*   & \underline{$92.9\pm0.2$} \\

**Consequence for the repo.**

GeoLDM's OWN paper (arXiv:2305.01140v1, icml2023.tex, label tab:qm9_results) reports its own QM9 row as Valid & Unique = 92.7 ± 0.5 over 10,000 samples -- the caption at icml2023.tex:468 reads 'Metrics are calculated with 10000 samples generated from each model' and the body at icml2023.tex:549 reads 'We generate $10,000$ samples from each method'. HLTF's Table 1(a) row 'GeoLDM' and Table 1(b) row 'Val&Uniq (%)' carry the SAME printed value 92.7±0.5 -- HLTF reproduced it, it did not originate it. CONSEQUENCE FOR THE REPO: the record's own `sources` block has NO GeoLDM entry (its keys are EF-TALFM / FlowMol3 / GGFlow / HLTF / PAFlow / YuelBond -- see molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources), so Reading A's lower endpoint is NOT substantiated by the record's own sources: the 92.7 % cell is present in the repo ONLY as a TRANSCRIPT of HLTF's table, with the originating paper absent from the record's source list. A third paper in the same scan (YuelBond) quotes 92.7 % for an unrelated observable (per-bond-type classification F1), so the numeral alone does not identify the cell.

### F2 -- Which of HLTF's tables carries the standardized post-processing recipe -- Table 1 (QM9, where BOTH Reading-A endpoints live) or Table 2 (GEOM-DRUGS)?

**VERDICT: ESTABLISHED**

Verbatim quotes (method: read out of the `.tex` extracted from the live e-print tarball; each carries the sha256 of that `.tex` and of the e-print):

> `2603.27113v1` :: `sections/experiments.tex:8`
>
> We follow the published 3D unconditional-generation protocols adopted by prior work. On QM9 we report atom stability (AS), molecule stability (MS), RDKit validity (Val), and Val\&Uniq on 10{,}000 unconditional samples. On GEOM-DRUGS we report Valid, Valid\&Unique (V\&U), and Valid\&Unique\&Novel (V\&U\&N) on 100{,}000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation\cite{doi:10.1021/ja00051a040}).  Since our central hypothesis concerns topology feasibility (valence/connectivity/ring consistency) as the dominant bottleneck, these success-rates directly reflect the failure modes we target. While these protocols directly measure end-to-end feasibility, they do not fully characterize distribution matching (e.g., scaffold/property coverage) or conformer realism beyond pass/fail validity and optional relaxation. We therefore interpret success-rate gains primarily as reductions in topology-driven failure modes, and leave broader distributional and geometry-quality analyses to future work (see Appendix \ref{app:limitations} for related limitations).

> `2603.27113v1` :: `sections/experiments.tex:38`
>
> Standardized post-processing mainly relieves \emph{geometric} strain (e.g., unrealistic bond lengths/angles and local steric clashes) and cannot repair infeasible topology (e.g., invalid valence or disconnected graphs). For HLTF it raises Valid from 85.5 to 92.2, converting $\approx 46\%$ of the pre-PP invalid samples under the same pipeline.\footnote{Computed as $(92.2-85.5)/(100-85.5)$.} This motivates the ablations below: we isolate which components reduce \emph{topology-driven} invalidity in the no-PP regime while preserving high PP success.

> `2603.27113v1` :: `sections/experiments.tex:34`
>
> Table~\ref{tab:geom_main} reports GEOM-DRUGS success rates. Without post-processing, HLTF achieves Valid/V\&U/V\&U\&N = 85.5/85.4/85.0, within 2.0 points of the best raw Valid in the table. With post-processing, HLTF reaches 92.2/92.0/91.2, remaining close to SemlaFlow under the same PP pipeline (Valid gap 0.9; 93.1 vs.\ 92.2). SemlaFlow is an explicit discrete-structure generator (including bond types), consistent with the view that explicit topology modeling is important in drug-like regimes; HLTF is competitive but we do not claim state-of-the-art headline success on this benchmark. The contribution is a complementary mechanism (hierarchy-guided planning + constraint-aware sampling) that remains competitive while targeting  \textbf{specific topology-driven failure modes}.

> `2603.27113v1` :: `tables/geom_drugs_results.tex:8`
>
> \caption{{GEOM-DRUGS success rates.} ``PP'' denotes standardized post-processing \cite{buttenschoen2025evaluation}. Asterisk * denotes the best value in the category and underline denotes the second best.}

> `2603.27113v1` :: `iclr2025_conference.tex:55`
>
> Generating chemically valid 3D molecules is hindered by \textbf{discrete bond topology}: small local bond errors can cause global failures (valence violations, disconnections, implausible rings), especially for drug-like molecules with long-range constraints. Many unconditional 3D generators emphasize coordinates and then infer bonds or rely on post-processing, leaving topology feasibility weakly controlled. We propose \textbf{Hierarchy-Guided Latent Topology Flow (HLTF)}, a planner–executor model that generates bond graphs \emph{with} 3D coordinates, using a latent multi-scale plan for global context and a constraint-aware sampler to suppress topology-driven failures. On \textbf{QM9}, HLTF achieves \textbf{98.8\% atom stability} and \textbf{92.9\% valid-and-unique}, improving \textbf{PoseBusters validity to 94.0\%} (\textbf{+0.9} over the strongest reported baseline). On \textbf{GEOM-DRUGS}, HLTF attains \textbf{85.5\%/85.0\%} validity/valid–unique–novel without post-processing and \textbf{92.2\%/91.2\%} after standardized relaxation, within \textbf{0.9} points of the best post-processed baseline. Explicit topology generation also reduces ``false-valid’’ samples that pass RDKit sanitization but fail stricter checks.\end{abstract}

> `2603.27113v1` :: `sections/appendix.tex:256`
>
> \subsection{Post-Processing: Valence Repair}

> `2603.27113v1` :: `sections/appendix.tex:269`
>
> We report metrics both with and without this repair to isolate the model's inherent validity.

> `2603.27113v1` :: `sections/methods.tex:254`
>
> Optionally, we apply a post-processing valence repair that deletes low-confidence incident bonds until all atoms satisfy their maximum valence;

**Consequence for the repo.**

HLTF's OWN wording scopes the standardized recipe to GEOM-DRUGS ONLY: the QM9 sentence in Section 4.1 (sections/experiments.tex:8) carries NO post-processing clause, while the immediately following GEOM-DRUGS sentence reads 'On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation)'. The QM9 Table 1 carries no '+PP' rows at all; the '+PP' rows live in Table 2 (GEOM-DRUGS, tables/geom_drugs_results.tex, label tab:geom_main), whose caption declares that 'PP' denotes standardized post-processing. Both of the band's Reading-A endpoints -- GeoLDM Val&Uniq and HLTF PB-Valid -- are QM9 Table 1 cells, so they carry NO post-processing. CONSEQUENCE: X2 is REFUTED BY THE PRIMARY SOURCE for the 92.7-94 % band. The LIVE catalogue row docs/HONEST_NEGATIVE_CATALOG.md, section 6.265, subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS' -- the class-(a) POST-PROCESSING row of the four-breaks table -- asserts 'the band is measured on the published post-processed stream' and is WRONG for THIS band. It was inherited from a DIFFERENT band: catalogue section 6.252, whose band is 37.4 % to 96.0 % and whose frontier numbers ARE post-processed (Buttenschoen et al. arXiv:2505.00518 raw -> post-processed pairs). CAVEAT (not a refutation): HLTF also applies an OPTIONAL valence repair that DELETES low-confidence incident bonds; the paper states 'We report metrics both with and without this repair' and does not mark which arm Table 1 carries. That repair deletes BONDS, not components, so it is not the largest-fragment retention the band question is about.

### F3 -- Does GeoLDM's own paper state that its model generates disconnected components, and does it apply a fragment fix to the reported QM9 numbers?

**VERDICT: ESTABLISHED**

Verbatim quotes (method: read out of the `.tex` extracted from the live e-print tarball; each carries the sha256 of that `.tex` and of the e-print):

> `2305.01140v1` :: `icml2023.tex:917`
>
> As shown in the two figures, the model is always able to generate realistic molecular geometries for both small and large size molecules. An outlier case is that the model occasionally generates disconnected components, as shown in the rightest column of \cref{app:fig:vis-drug}, which happens more often when trained on the large molecule DRUG dataset. However, this phenomenon actually is not a problem and is common in all non-autoregressive molecule generative models~\cite{zang2020moflow,jo2022score}, and can be easily fixed by just filtering the smaller components.

> `2305.01140v1` :: `icml2023.tex:532`
>
> \textbf{Evaluation Metrics.} We measure model performances by evaluating the chemical feasibility of generated molecules, indicating whether the model can learn chemical rules from data. Given molecular geometries, we first predict bond types (single, double, triple, or none) by pair-wise atomic distances and atom types. Then we calculate the \textit{atom stability} and \textit{molecule stability} of the predicted molecular graph. The first metric captures the proportion of atoms that have the right valency, while the latter is the proportion of generated molecules for which all atoms are stable. In addition,  We report \textit{validity} and \textit{uniqueness} metrics, which are the percentages of valid (measured by \textsc{RDKit}) and unique molecules among all the generated compounds.

> `2305.01140v1` :: `icml2023.tex:549`
>
> We generate $10,000$ samples from each method to calculate the above metrics, and the results are reported in \cref{tab:qm9_results}. As shown in the table, \method outperforms competitive baseline methods on all metrics with an obvious margin. 

**Consequence for the repo.**

GeoLDM's own paper DOES name the artefact: 'the model occasionally generates disconnected components ... this phenomenon actually is not a problem ... and can be easily fixed by just filtering the smaller components.' The phrase 'can be easily fixed' is a REMEDY PROPOSED, not a procedure applied to the reported numbers -- the paper's ONLY occurrence of 'filter' in the whole source is that sentence, and the numbers in tab:qm9_results are the unfiltered ones. The same sentence says the disconnection 'happens more often when trained on the large molecule DRUG dataset', i.e. it is a DRUG-side observation, not a QM9-side correction. CONSEQUENCE: the Wf-A175 row-5 verdict for GeoLDM ('PROTOCOL-UNSPECIFIED') can be UPGRADED to RAW with a verbatim locator -- GeoLDM's QM9 92.7 ± 0.5 is a RAW-stream number. This is independent primary-source evidence against X2 and X3 for the band's lower endpoint.

### F4 -- GeoLDM's own paper reports its QM9 Valid (%) as 93.8 +- 0.4, while HLTF's Table 1(a)/1(b) tabulate the GeoLDM column's Val (%) as 93.6 +- 0.2. Do the two source tables disagree?

**VERDICT: ESTABLISHED -- the two source tables DISAGREE, and no document on disk records it**

Verbatim quotes (method: read out of the `.tex` extracted from the live e-print tarball; each carries the sha256 of that `.tex` and of the e-print):

> `2305.01140v1` :: `icml2023.tex:491`
>
>     \rowcolor{lightgray} \textbf{\method} & \textbf{98.9} $\pm$ 0.1 & \textbf{89.4} $\pm$ 0.5 & \textbf{93.8} $\pm$ 0.4 & \textbf{92.7} $\pm$ 0.5 & \textbf{84.4} & \textbf{99.3} \\

> `2603.27113v1` :: `tables/qm9_benchmarks_og.tex:38`
>
> GeoLDM & $98.9\!\pm\!0.1$* & $89.4\!\pm\!0.5$* & $93.6\!\pm\!0.2$ & $92.7\!\pm\!0.5$ \\

> `2603.27113v1` :: `tables/qm9_benchmarks_og.tex:63`
>
> Val (\%) $\uparrow$           & $93.6\pm0.2$       & $94.8\pm0.2$*   & \underline{$93.8\pm0.6$} \\

> `2603.27113v1` :: `tables/qm9_benchmarks_og.tex:8-12`
>
> \caption{\textbf{QM9 results.}
> (a) Unconditional generation; (b) novelty and PoseBusters validity.
> Baselines reproduced from GCDM Table~1 \cite{morehead2023gcdm}.
> Arrows $\uparrow/\downarrow$ indicate whether higher/lower is better.
> Asterisk * denotes best; underline denotes second best.}

**Consequence for the repo.**

BOTH numerals are verbatim from the primary sources: GeoLDM's own paper prints Valid = 93.8 ± 0.4 for the GeoLDM row of tab:qm9_results (icml2023.tex:491); HLTF prints the GeoLDM column's Val as 93.6±0.2 in Table 1(a) (tables/qm9_benchmarks_og.tex:38) and 93.6±0.2 in Table 1(b) (tables/qm9_benchmarks_og.tex:63). The SAME observable (RDKit validity of GeoLDM's QM9 output) therefore carries two different printed values in two papers, and HLTF's own Table 1 caption states 'Baselines reproduced from GCDM Table 1', i.e. the column is a TRANSCRIPT, so the disagreement is a transcription difference that neither paper flags. NOTE: 93.8 is ALSO HLTF's OWN Val in HLTF's own table (93.8±0.6), so the numeral 93.8 is carried in at least two distinct roles. WHICH NUMERALS ARE BAND ENDPOINTS: under BOTH readings the band's endpoints are 92.7 / 94.0 (Reading A) or 92.98 / 94.33 (Reading B); 93.8 and 93.6 are NOT band endpoints under either reading and must never be published as such. REPO-STATE MEASUREMENT: a repo-wide scan of tracked .md/.json/.tex/.py/.yaml files under docs/, TODO/, metrics/, molmetal/ and paper/ for '93.6' returns hits ONLY in an unrelated NCI-60 split percentage (93.6 / 2.0 / 4.4 %) -- zero hits in the frontier sense. The disagreement is UNRECORDED.

### F5 -- Does EF-TALFM's own Table 3(a) confirm the repo's existing EF-TALFM entries -- the row labels and columns, the C. / S. / U. & S. / N. & U. & S. definitions, the 10,000 denominator, and the statement that the connectivity count is taken BEFORE the largest-component retention?

**VERDICT: ESTABLISHED**

Verbatim quotes (method: read out of the `.tex` extracted from the live e-print tarball; each carries the sha256 of that `.tex` and of the e-print):

> `2609.08333v4` :: `src/appendix.tex:101`
>
>     \textbf{(a) End-to-end generation results (\%)}

> `2609.08333v4` :: `src/appendix.tex:93`
>
>     Panel (a) rates are computed over 10,000 generated molecules. C. denotes the fraction of connected molecules; S. the fraction passing strict RDKit sanitization; U.\,\&\,S. the fraction of unique sanitized canonical SMILES; and N.\,\&\,U.\,\&\,S. the fraction that are sanitized, unique, and absent from the training set. Panel (b) PoseBusters rates are computed among the N.\,\&\,U.\,\&\,S. molecules: 7,603 for UAE-3D, 6,983 for FlowMol, and 9,433 for EF-TALFM. Missing internal-energy evaluations (45 for UAE-3D, 12 for FlowMol, and 191 for EF-TALFM) are counted as failures in both the internal-energy and joint pass rates. For fragmented outputs, SMILES-based and PoseBusters metrics are computed on the atom-count largest connected component after RDKit sanitization.}

> `2609.08333v4` :: `src/appendix.tex:60`
>
>     \item Connectivity: \% of molecules with exactly one connected component, i.e., one fragment.

> `2609.08333v4` :: `src/appendix.tex:54`
>
> For fragmented outputs, SMILES-based and PoseBusters metrics use the atom-count largest connected component after sanitization.

> `2609.08333v4` :: `src/appendix.tex:55`
>
> Connectivity, sanitization, unique sanitized yield and novel unique sanitized yield use all generated samples as their denominator; individual PoseBusters check rates use the novel, unique, sanitized subset.

> `2609.08333v4` :: `src/appendix.tex:183`
>
>     We report connectivity of the reconstructed molecule and sanitization, uniqueness, and novelty rates computed after retaining the atom-count largest connected component of each sampled molecule and applying strict RDKit sanitization.

> `2609.08333v4` :: `src/appendix.tex:1010`
>
> Starting from the initially reconstructed heavy-atom graph and the bond orders inferred from atom- and molecule-level features, we retain the largest connected component and discard the rest, since the reconstructed graph may contain multiple components. 

> `2609.08333v4` :: `src/appendix.tex:308`
>
> Among the $92{,}221$ connected molecules, $90{,}807$ reconstructed graphs pass strict sanitization before geometry refinement.

> `2609.08333v4` :: `src/appendix.tex:82`
>
> \Cref{tbl:pcqm4mv2_generation} shows that relative to FlowMol, EF-TALFM nearly matches the sanitization rate (98.25\% versus 98.72\%) and unique sanitized yield (98.21\% versus 98.27\%), while its connectivity is moderately lower (92.98\% versus 98.72\%).

**Consequence for the repo.**

CONFIRMED VERBATIM. Table 3(a) of arXiv:2609.08333v4 is titled '(a) End-to-end generation results (%)' and its columns are UAE-3D / FlowMol / EF-TALFM; its caption defines the four observables and fixes the denominator ('Panel (a) rates are computed over 10,000 generated molecules'); and the ORDERING is explicit in the Table 4 caption, which says 'We report connectivity of the reconstructed molecule and sanitization, uniqueness, and novelty rates computed after retaining the atom-count largest connected component of each sampled molecule and applying strict RDKit sanitization' -- connectivity is reported OF THE RECONSTRUCTED MOLECULE, while the companion rates are computed AFTER the retention. Two independent corroborations sit in the same source: Appendix D states the pipeline always retains ('we retain the largest connected component and discard the rest, since the reconstructed graph may contain multiple components'), and the population accounting counts 'Among the 92,221 connected molecules, 90,807 reconstructed graphs pass strict sanitization before geometry refinement'. CONSEQUENCE: the repo's EF-TALFM POST-PROCESSING cell, which already calls that arm MIXED ('MIXED -- the lower cell is a RAW connectivity count, the upper cell is a POST-PROCESSED composite', molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM.cells.POST-PROCESSING.post_processed), is CORRECT; the Wf-A183 repair that made the 'RAW' claim reading-conditional is CORRECT. ONE NEW, UNRECORDED OBSERVATION: EF-TALFM's own prose at src/appendix.tex:82 compares EF-TALFM's connectivity 92.98 % against '98.72 %' -- but 98.72 % is FlowMol's SANITIZATION cell in Table 3(a); FlowMol's connectivity cell in the same table reads 100.00. The paper's prose compares a connectivity rate against a sanitization rate. This does not change any band numeral, but it means the appendix prose is not a safe source for the connectivity comparison.

## 3. THE FOUR INVARIANTS (hold under BOTH readings -- the only publishable claims)

* **i_the_band_is_external** -- The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.
* **ii_it_is_not_one_homogeneous_quantity** -- It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a six-check pose verdict), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.
* **iii_it_is_not_on_our_scale** -- It is NOT on our scale. OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.
* **iv_the_source_cells_are_not_settled** -- The source cells of the band are NOT settled on disk.

OUR side: 8.51 % -- n_single / samples drawn, where n_single = the pool rows whose decoded molecule has exactly ONE RDKit fragment, measured at the RAW emitted SMILES -- NO post-processing, NO LargestFragmentChooser, NO explicit H, NO UFF. (source `metrics/by_metric/cfm_fragmentation_audit.json` -> `/summary/generation_side/single_molecule_rate_end_to_end`, sha256 `351c4d00af54ef83b88308597bfb67c093d5df6eb8c6e753251f923f00a3240f`).

Band endpoints, named per reading:

* **band_endpoints_under_reading_a**
  * lower: 92.7 ± 0.5 -- arXiv:2305.01140v1 icml2023.tex:491 (GeoLDM's own paper)
  * upper: 94±0.5* -- arXiv:2603.27113v1 tables/qm9_benchmarks_og.tex:66 (Table 1(b), HLTF's own PB-Valid cell)
* **band_endpoints_under_reading_b**
  * lower: 92.98 -- arXiv:2609.08333v4 src/appendix.tex:112-115 (Table 3(a), RAW connectivity)
  * upper: 94.33 -- arXiv:2609.08333v4 src/appendix.tex:124-127 (Table 3(a), post-processed companion)

Numerals that are NOT band endpoints under either reading:

* **93.8** -- GeoLDM's own QM9 Valid (arXiv:2305.01140v1 icml2023.tex:491) AND HLTF's own QM9 Val (arXiv:2603.27113v1 tables/qm9_benchmarks_og.tex:41) -- two different papers' own figures, not a band endpoint under either reading.
* **93.6** -- HLTF's tabulation of GeoLDM's Val (arXiv:2603.27113v1 tables/qm9_benchmarks_og.tex:38 and :63), which CONTRADICTS GeoLDM's own 93.8 for the same observable -- see F4.

The repo holds **3** competing provenance readings, not two:

* Reading A (frontier_matrix.json -> band_provenance): GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both post_processed: false, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral. Reproduces the quoted numerals exactly.
* Reading B (frontier_matrix.json -> sources.EF-TALFM): EF-TALFM's own Table 3(a) RAW connectivity C. = 92.98 % counted BEFORE the largest-component retention, paired with the post-processed N. & U. & S. = 94.33 % companion cell. Does NOT reproduce the quoted numerals.
* Reading C (docs/HONEST_NEGATIVE_CATALOG.md, section 6.266.8, 'The frontier papers, pinned from primary sources (Wf-A168) -- and one correction to the previous'): the band is a composite of the EF-TALFM cells 92.98 % / 94.33 % / 93.13 %, which have three different denominators. Also does NOT reproduce the quoted numerals. This THIRD reading is NOT among the two recorded in frontier_matrix.json -- the contest is three-way, not two-way.

## 4. THE CATALOGUE ROW THAT F2 REFUTES

`docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS' -> the POST-PROCESSING row` -- verbatim:

```
| **POST-PROCESSING** | the band is measured on the published **post-processed** stream | both arms are **RAW** — no LargestFragmentChooser, no explicit H, no UFF, no post-processing |
```

file sha256 `a2e91cb095f23db3edc05636eb24c741388f20f1455fc7142585f9f41d42f89d` (2,676,880 bytes).

The line number this row used to be cited by is DEMOTED -- the catalogue grows, so a line number into it is perishable. The durable locator is the section anchor above; the superseded numbers are preserved VERBATIM in the dated demotion block (`locator_demotion_2026_11_02_stale_by_construction`).

It was inherited from a different band: the 6.252 band is a CONNECTIVITY band measured on Buttenschoen et al. (arXiv:2505.00518) POST-PROCESSED rates; the 92.7-94 % band's Reading-A endpoints are HLTF QM9 Table 1 cells that carry NO post-processing, and its Reading-B lower endpoint is a RAW connectivity count.

**Proposed, dated, additive correction (Wf-A186, 2026-11-01) -- PROPOSED ONLY -- this wave did NOT write to docs/.**

CORRECTION (2026-11-01, Wf-A186) -- the POST-PROCESSING row of this four-breaks table is WRONG FOR THE 92.7-94 % BAND, and is inherited from a DIFFERENT band (catalogue section 6.252, lines 18603 and 18671, whose band is 37.4 % to 96.0 % and whose frontier rates ARE post-processed). ESTABLISHED FROM THE PRIMARY SOURCES: HLTF's standardized post-processing recipe (largest fragment, explicit hydrogens, UFF relaxation) is scoped by HLTF ITSELF to GEOM-DRUGS ONLY -- arXiv:2603.27113v1 sections/experiments.tex:8 reads 'On QM9 we report atom stability (AS), molecule stability (MS), RDKit validity (Val), and Val&Uniq on 10,000 unconditional samples. On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation)' -- the QM9 Table 1 has no '+PP' rows (the '+PP' rows are in Table 2, tables/geom_drugs_results.tex, label tab:geom_main). BOTH Reading-A band endpoints are QM9 Table 1 cells and carry NO post-processing; the Reading-B lower endpoint is a RAW connectivity count (EF-TALFM arXiv:2609.08333v4 src/appendix.tex:183 reports connectivity OF THE RECONSTRUCTED MOLECULE while the companion rates are computed AFTER the largest-component retention). GeoLDM's own paper confirms the RAW reading for the lower endpoint: arXiv:2305.01140v1 icml2023.tex:917 names disconnected components as a known artefact that 'can be easily fixed by just filtering the smaller components' -- a remedy PROPOSED, not applied to the reported numbers. CORRECTED CELL: 'the band is NOT measured on a post-processed stream -- under Reading A both endpoints are HLTF QM9 Table 1 cells with NO post-processing; under Reading B the lower endpoint is a RAW connectivity count and only the upper companion cell is post-processed'. The row's OUR-SIDE half ('both arms are RAW') stands.

## 5. THE X1-X4 CLASS -- LIVE STATUS ON DISK

* **X1** = no endpoint is a connected-component count / THE BAND IS NOT A CONNECTIVITY BAND
  * live status: SURVIVES ONLY IN DATED SUPERSEDED KEYS (correct per AGENTS.md 5.5 rule #6): the repo's own record molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.headline_finding_superseded_2026_11_01 and -> sources.HLTF.cells.METRIC.note_superseded_2026_11_01 -- both are DATED sibling keys of their live counterparts; the LIVE band_provenance.headline_finding is READING-QUALIFIED and no longer carries it. The repo's own X-class list already marks it FORBIDDEN (molmetal/reports/wfa161_frontier_labelling/report.md:314).
  * this wave: does NOT reproduce it; this report asserts only the four invariants that hold under BOTH readings.
* **X2** = the band is measured on the POST-PROCESSED stream
  * live status: ONE LIVE SURFACE REMAINS: docs/HONEST_NEGATIVE_CATALOG.md, section 6.265, subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS', the class-(a) POST-PROCESSING row of the 92.7-94 % band's four-breaks table. The metric artefact metrics/by_metric/wfa155_metal_coordination_sphere.json carries the axis only in reading-conditional form ('whether the band as a whole is RAW is NOT settled on disk'), and the two generator scripts carry it only in comments. The repo already refuted it from its OWN record (frontier_matrix.json -> band_provenance -> endpoint_candidates, post_processed: false); this wave REFUTES IT FROM THE PRIMARY SOURCE (F2).
  * this wave: REFUTED by the primary source for this band (F2).
* **X3** = their Connectivity is computed AFTER retaining the largest component
  * live status: no live surface found; the repo's X-class list already marks it REFUTED (molmetal/reports/wfa164_coordination_sphere_narrative/synthetic_pool_artifact.json:99).
  * this wave: REFUTED by the primary source (F5): EF-TALFM reports connectivity OF THE RECONSTRUCTED MOLECULE, before the retention.
* **X4** = the band is minted here: lower = GeoLDM Val&Uniq, upper = HLTF PB-Valid
  * live status: no live surface found; already marked as asserting ONE reading as settled (molmetal/reports/wfa164_coordination_sphere_narrative/pool_specific_literal_census.json:148).
  * this wave: does NOT reproduce it; F1 adds that the record's own sources block has NO GeoLDM entry at all.

## 6. THE SENTENCE WE MAY SAY, AND THE SENTENCES WE MAY NOT

### MAY

* MAY: 'The published 92.7-94 % band is EXTERNAL to Mol-Metal, is NOT one homogeneous quantity, is NOT on our scale, and its source cells are NOT settled on disk.'
* MAY: 'Under Reading A, the band's lower endpoint is GeoLDM's OWN published QM9 Valid & Unique cell (arXiv:2305.01140v1 icml2023.tex:491), reproduced identically by HLTF; the upper endpoint is HLTF's own QM9 PoseBusters-validity cell (arXiv:2603.27113v1 tables/qm9_benchmarks_og.tex:66).' -- always with the reading named.
* MAY: 'Under Reading B, the band's lower endpoint is EF-TALFM's RAW connectivity cell (arXiv:2609.08333v4 src/appendix.tex:112-115), counted BEFORE the largest-component retention, and the upper endpoint is that table's post-processed N. & U. & S. companion cell.' -- always with the reading named.
* MAY: 'Neither reading's band endpoint is post-processed in its own right: the Reading-A endpoints are HLTF QM9 Table 1 cells, which carry no post-processing; the Reading-B lower endpoint is a RAW connectivity count.'
* MAY: 'OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.'

### MAY NOT

* MAY NOT: 'THE BAND IS NOT A CONNECTIVITY BAND' / 'no endpoint is a connected-component count' -- false under Reading B.
* MAY NOT: 'the band is measured on the POST-PROCESSED stream' -- false for this band; it was inherited from the 37.4-96.0 % band of catalogue section 6.252.
* MAY NOT: 'their Connectivity is computed AFTER retaining the largest component' -- EF-TALFM reports connectivity OF THE RECONSTRUCTED MOLECULE, before the retention.
* MAY NOT: 'the band is minted here: lower = GeoLDM Val&Uniq, upper = HLTF PB-Valid' -- that asserts one reading as settled.
* MAY NOT: quote 93.8 or 93.6 as a band endpoint -- under both readings the endpoints are 92.7 / 94.0 or 92.98 / 94.33.
* MAY NOT: quote any cell of these three tables without its file:line locator and the sha256 of the bytes it came from.

## 7. ADDITIONAL OBSERVATIONS AND WHAT THIS WAVE DID NOT ESTABLISH

* UNRECORDED LOCATOR ERROR, still live in a published metric artefact: metrics/by_metric/wfa155_metal_coordination_sphere.json says 'HLTF arXiv:2603.27113v1 Table 1(b) reports six models RAW and POST-PROCESSED side by side'. The RAW/+PP pairs are in Table 2 (GEOM-DRUGS, tables/geom_drugs_results.tex, label tab:geom_main); Table 1(b) (tables/qm9_benchmarks_og.tex, label tab:qm9_pb) is QM9 and reports six OBSERVABLES (AS / MS / Val / Val&Uniq / Novel / PB-Valid) for three models, with no +PP rows. The same Table-1-vs-Table-2 confusion is the mechanism by which the X2 class survives.
* The Wf-A175 cached receipt's per-file 'file_bytes' field is a UTF-8 CHARACTER count, not a byte count. The e-print-level byte count and sha256 are byte-exact and match the live re-fetch for all three papers.
* The repo holds THREE competing provenance readings for the band, not two: see invariants.competing_readings.
* GeoLDM's paper contains exactly one occurrence of 'filter' in its whole source (icml2023.tex:917), and it is the proposed-but-not-applied fragment filter; no reported number is filtered by it.
* The repo cites HLTF's second author as 'Alexander Arjun Lobo'; the paper's own \author line reads 'Alexander Lobo'.

* NOT ESTABLISHED -- This wave did NOT establish WHICH cells the repo's literature scan selected to mint the band -- the selection rule is not recorded on disk. It established what each candidate cell IS, from the primary sources.
* NOT ESTABLISHED -- This wave did NOT establish whether HLTF's QM9 Table 1 numbers are the with-valence-repair or the without-valence-repair arm: HLTF states 'We report metrics both with and without this repair' and does not mark which arm Table 1 carries. The repair deletes BONDS, not components, so it does not bear on the largest-component question.

## 8. Wf-A190 (2026-11-02) -- LOCATOR DEMOTION: the record and the catalogue are cited by JSON PATH / SECTION ANCHOR, not by line number

**the LIVE locator is a JSON PATH (into the record) or a SECTION ANCHOR (into the catalogue). A line number may only appear inside a DATED historical record that says, in words, that it was measured on a given date and that it is STALE BY CONSTRUCTION. Never delete a line number: demote it.**

**MEASURED on 2026-11-02.** The record `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` is **1136 lines / 108408 bytes** (sha256 `878c387e2b9c1bd8c86d087e6266b19853f4f7f5eb05da1381c7ee46d945aa67`) and the catalogue `docs/HONEST_NEGATIVE_CATALOG.md` is **1993469 bytes** (sha256 `27f953f0a7a4b6eb5f3f465741d5a34d9576fd6870a7643b2aa1de42cb8b6502`). Both are UNCHANGED since 2026-11-01 -- neither has moved -- and BOTH are STALE BY CONSTRUCTION: the record is regenerated and keeps growing, and the catalogue is appended to by every wave.

**Superseded renderings, PRESERVED VERBATIM** (AGENTS.md 5.5 rule #6: append-only + dated; nothing is deleted). Every line number below is STALE BY CONSTRUCTION:

| where | superseded rendering (verbatim) |
|---|---|
| `retrieval[1].authors_note` | the repo cites the second author as 'Alexander Arjun Lobo' (molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:779); the paper's own \author line reads 'Alexander Lobo'. |
| `x_class_status.X1.live_status` | SURVIVES ONLY IN DATED SUPERSEDED KEYS (correct per AGENTS.md 5.5 rule #6): molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:145 (band_provenance.headline_finding_superseded_2026_11_01) and :763-764 (sources.HLTF.cells.METRIC.note_superseded_2026_11_01); the LIVE band_provenance.headline_finding is READING-QUALIFIED and no longer carries it. The repo's own X-class list already marks it FORBIDDEN (molmetal/reports/wfa161_frontier_labelling/report.md:314). |
| `x_class_status.X2.live_status` | ONE LIVE SURFACE REMAINS: docs/HONEST_NEGATIVE_CATALOG.md:20819 (the class-(a) POST-PROCESSING row of the 92.7-94 % band's four-breaks table). The metric artefact metrics/by_metric/wfa155_metal_coordination_sphere.json carries the axis only in reading-conditional form ('whether the band as a whole is RAW is NOT settled on disk'), and the two generator scripts carry it only in comments. The repo already refuted it from its OWN record (frontier_matrix.json -> band_provenance -> endpoint_candidates, post_processed: false); this wave REFUTES IT FROM THE PRIMARY SOURCE (F2). |
| `findings.F2.consequence` | HLTF's OWN wording scopes the standardized recipe to GEOM-DRUGS ONLY: the QM9 sentence in Section 4.1 (sections/experiments.tex:8) carries NO post-processing clause, while the immediately following GEOM-DRUGS sentence reads 'On GEOM-DRUGS we report Valid, Valid&Unique (V&U), and Valid&Unique&Novel (V&U&N) on 100,000 samples, both raw and after standardized post-processing (largest fragment, explicit hydrogens, UFF relaxation)'. The QM9 Table 1 carries no '+PP' rows at all; the '+PP' rows live in Table 2 (GEOM-DRUGS, tables/geom_drugs_results.tex, label tab:geom_main), whose caption declares that 'PP' denotes standardized post-processing. Both of the band's Reading-A endpoints -- GeoLDM Val&Uniq and HLTF PB-Valid -- are QM9 Table 1 cells, so they carry NO post-processing. CONSEQUENCE: X2 is REFUTED BY THE PRIMARY SOURCE for the 92.7-94 % band. The LIVE catalogue row docs/HONEST_NEGATIVE_CATALOG.md:20819 -- the class-(a) POST-PROCESSING row of the four-breaks table -- asserts 'the band is measured on the published post-processed stream' and is WRONG for THIS band. It was inherited from a DIFFERENT band: catalogue section 6.252 (line 18603), whose band is 37.4 % to 96.0 % and whose frontier numbers ARE post-processed (Buttenschoen et al. arXiv:2505.00518 raw -> post-processed pairs). CAVEAT (not a refutation): HLTF also applies an OPTIONAL valence repair that DELETES low-confidence incident bonds; the paper states 'We report metrics both with and without this repair' and does not mark which arm Table 1 carries. That repair deletes BONDS, not components, so it is not the largest-fragment retention the band question is about. |
| `invariants.competing_readings[2]` | Reading C (docs/HONEST_NEGATIVE_CATALOG.md:21273): the band is a composite of the EF-TALFM cells 92.98 % / 94.33 % / 93.13 %, which have three different denominators. Also does NOT reproduce the quoted numerals. This THIRD reading is NOT among the two recorded in frontier_matrix.json -- the contest is three-way, not two-way. |
| `our_side.catalogue_quote.path` | docs/HONEST_NEGATIVE_CATALOG.md:18680-18684 |
| `catalogue_row.inherited_from.band_locator` | docs/HONEST_NEGATIVE_CATALOG.md:18671 |
| `catalogue_row.inherited_from.may_may_not_block` | docs/HONEST_NEGATIVE_CATALOG.md:18678-18688 |

**Superseded NUMERIC locators, preserved VERBATIM** (the fields that held a bare line number rather than a `path:line` string):

| superseded field | preserved verbatim at | value |
|---|---|---|
| `catalogue_row.line` | `catalogue_row.surrounding_four_breaks_table.line_numbers_measured_2026_11_01.post_processing_line` | 20819 |
| `catalogue_row.surrounding_four_breaks_table.header_line` | `catalogue_row.surrounding_four_breaks_table.line_numbers_measured_2026_11_01.header_line` | 20815 |
| `catalogue_row.surrounding_four_breaks_table.denominator_line` | `catalogue_row.surrounding_four_breaks_table.line_numbers_measured_2026_11_01.denominator_line` | 20817 |
| `catalogue_row.surrounding_four_breaks_table.metric_line` | `catalogue_row.surrounding_four_breaks_table.line_numbers_measured_2026_11_01.metric_line` | 20818 |
| `catalogue_row.surrounding_four_breaks_table.hydrogens_line` | `catalogue_row.surrounding_four_breaks_table.line_numbers_measured_2026_11_01.hydrogens_line` | 20820 |
| `catalogue_row.inherited_from.heading_line` | `catalogue_row.inherited_from.line_numbers_measured_2026_11_01.heading_line` | 18603 |

**Superseded KEY NAME, preserved VERBATIM** (the key itself embedded a line number into the growing catalogue):

| where | superseded key name (verbatim) |
|---|---|
| top level | `catalogue_row_20819` |

**What those lines ACTUALLY contained on 2026-11-02 (measured on the record / the catalogue):**

| cited line | what it actually held |
|---|---|
| `145` | band_provenance.headline_finding_superseded_2026_11_01 -- the cited line is CORRECT today, but it is a line number into a GROWING record, so it is perishable. |
| `763-764` | sources.GGFlow.reported_numbers.label and .locator -- a GGFlow (arXiv:2411.05676v1) row. The key the superseded locator NAMED, sources.HLTF.cells.METRIC.note_superseded_2026_11_01, is at line 830. The locator was ALREADY WRONG on 2026-11-02, not merely perishable. |
| `779` | sources.GGFlow.reported_numbers.quote -- a GGFlow quote row. The string the superseded locator NAMED ('Urvi Awasthi, Alexander Arjun Lobo, Leonid Zhukov') is at sources.HLTF.citation.authors (line 846) and in retrieval_log[arxiv_id=2603.27113, kind=arxiv-abs].extracted (line 451). The locator was ALREADY WRONG on 2026-11-02. |
| `20819` | docs/HONEST_NEGATIVE_CATALOG.md -- the class-(a) POST-PROCESSING row of the 92.7-94 % band's four-breaks table, inside section 6.265, subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS'. CORRECT today; STALE BY CONSTRUCTION. |
| `18603` | docs/HONEST_NEGATIVE_CATALOG.md -- the heading of section 6.252 itself. CORRECT today; STALE BY CONSTRUCTION. |
| `21273` | docs/HONEST_NEGATIVE_CATALOG.md -- a body line inside section 6.266.8 ('The frontier papers, pinned from primary sources (Wf-A168) -- and one correction to the previous'). CORRECT today; STALE BY CONSTRUCTION. |

**The durable replacements (JSON PATHS and SECTION ANCHORS):**

| what | durable locator |
|---|---|
| the record's dated headline key | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance.headline_finding_superseded_2026_11_01` |
| the record's dated HLTF METRIC note | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.HLTF.cells.METRIC.note_superseded_2026_11_01` |
| the record's HLTF author string | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.HLTF.citation.authors` |
| the record's arXiv-abstract receipt carrying the same string | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> retrieval_log[arxiv_id=2603.27113, kind=arxiv-abs].extracted` |
| the catalogue's 92.7-94 % four-breaks table, POST-PROCESSING row | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS' -> the POST-PROCESSING row` |
| the catalogue's 37.4-96.0 % band section | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.252` |
| the catalogue's primary-source reading table | `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.266.8` |

**Corrections to this wave's own demotion block (append-only, dated):**

* **2026-11-02, Wf-A190.** Superseded rendering, preserved verbatim: `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.264 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS'`. Corrected to: `docs/HONEST_NEGATIVE_CATALOG.md -> section 6.265 -> subsection '4. THE PUBLISHED 92.7-94 % BAND IS NOT OUR BAND -- FOUR INDEPENDENT BREAKS'`. MEASURED against the catalogue on disk on 2026-11-02 (sha256 27f953f0a7a4b6eb5f3f465741d5a34d9576fd6870a7643b2aa1de42cb8b6502, 1993469 B, 21689 lines): the '## §6.264' heading is at line 20288 and the '## §6.265' heading is at line 20625, so the four-breaks table and its '### 4.' subsection heading (line 20808) sit under §6.265, NOT §6.264. The §6.264 sibling is '### 6. THE PUBLISHED 92.7-94 % VALIDITY BAND IS NOT OUR BAND -- FOUR BREAKS' (line 20487). The first rendering of this wave's demotion named §6.264; it is preserved VERBATIM above and corrected here.

*Section 8 appended 2026-11-02 (Wf-A190, consumer side). Every number above is MEASURED; the line numbers are STALE BY CONSTRUCTION by design.*

