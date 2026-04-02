# Error Model Estimation with SOLQC

Estimate error models (insertion, deletion, substitution rates) from real-world DNA storage datasets using [SOLQC](https://yoav-orlev.gitbook.io/solqc) (Synthetic Oligo Library Quality Control).

## 1. Install and run SOLQC

SOLQC runs as a Docker container:

```bash
docker run -p 5000:5000 solqc/tool
```

Then open `http://localhost:5000/` in your browser.

> **Note:** If port 5000 is in use (e.g. AirPlay on Mac), find and kill the process: `lsof -i :5000` then `kill -9 <PID>`.

> **Remote server:** If SOLQC runs on a remote machine, set up SSH port forwarding from your local machine:
> ```bash
> ssh -L 5000:localhost:5000 user@remote-host
> ```
> Then open `http://localhost:5000/` locally.

## 2. Convert data to SOLQC format

Convert a `train.txt` file (format: `read1|read2|...|readN:ground_truth`) into SOLQC input files. Since our data already has reads matched to ground truths, we generate a pre-matched CSV and skip SOLQC's own matching step.

```bash
python data/error_model/convert_to_solqc.py <path/to/train.txt> --output-dir <output_dir> [--fraction 0.1]
```

This produces:
- **`design.csv`**: ground truth sequences
- **`reads_matching.csv`**: pre-matched reads with alignment cigar paths

Example for Microsoft dataset at different data fractions:

```bash
python data/error_model/convert_to_solqc.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/microsoft_05pct  --fraction 0.05
python data/error_model/convert_to_solqc.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/microsoft_10pct  --fraction 0.10
python data/error_model/convert_to_solqc.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/microsoft_25pct  --fraction 0.25
python data/error_model/convert_to_solqc.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/microsoft_50pct  --fraction 0.50
python data/error_model/convert_to_solqc.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/microsoft_75pct  --fraction 0.75
python data/error_model/convert_to_solqc.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/microsoft_100pct --fraction 1.0
```

## 3. Run SOLQC

In the SOLQC web UI at `http://localhost:5000/`:

### Option A: Pre-matched data (our datasets)

Use this when reads are already matched to ground truths (i.e. from `train.txt`).

1. **Design**: upload `design.csv`
2. **NGS Files**: check the **"matching" checkbox**, then upload `reads_matching.csv`
3. **Library Configuration**:
   - **Prefix/Suffix**: leave empty (primers already stripped)
   - **Length**: design variant length (e.g. 110 for Microsoft)
   - **Barcode start/end**: 0
   - **Barcode tolerance**: 0

### Option B: Raw data with barcodes

Use this when you have raw FASTQ files with barcodes (e.g. directly from sequencing).

1. **Design**: upload a CSV with `barcode,sequence` columns
2. **NGS Files**: uncheck the "matching" checkbox, upload `.fastq` file
3. **Library Configuration**: fill in prefix, suffix, barcode positions, and length as appropriate

## 4. Goal

Estimate error models from the Microsoft dataset using different proportions of training data (5%, 10%, 25%, 50%, 75%, 100%) to determine when the estimated error rates (insertion, deletion, substitution) converge.
