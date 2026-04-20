# Chandak Nanopore DNA Storage Dataset

Processing pipeline to convert the raw [Chandak et al.](https://github.com/shubhamchandak94/nanopore_dna_storage_data) nanopore DNA storage data into TReconLM format.

## 1. Download raw data

Clone the data repository (contains ground truth and noisy reads). Run from the project root:

```bash
cd data/chandak
git clone https://github.com/shubhamchandak94/nanopore_dna_storage_data.git
```

The ground truth files (`oligo_files/oligos_0.fa` ... `oligos_12.fa`) are included in the repo. The noisy reads (`fastq/merged.fastq`) need to be downloaded and decompressed separately.

Download the compressed reads (continuing from `data/chandak/`):

```bash
sudo apt install -y wget
cd nanopore_dna_storage_data/fastq
wget "https://drive.usercontent.google.com/download?id=1yFOChP7qlOvS29llTD7WdhTNHaR9BySy&export=download&confirm=t" -O merged.fastq.spring
```

Build [SPRING](https://github.com/shubhamchandak94/Spring) and decompress (continuing from `data/chandak/nanopore_dna_storage_data/fastq/`):

```bash
git clone https://github.com/shubhamchandak94/Spring.git
cd Spring && mkdir build && cd build && cmake .. && make && cd ../..
./Spring/build/spring -d -i merged.fastq.spring -o merged.fastq
```

## 2. Extract primers and sort reads by experiment

The raw `merged.fastq` contains reads from all 13 experiments mixed together. This step identifies which experiment each read belongs to (by matching primer sequences), strips the primers, and saves the reads into per-experiment folders.

Return to the project root and run:

```bash
cd ../../../..
python data/chandak/extract_primers.py \
    --oligo-dir data/chandak/nanopore_dna_storage_data/oligo_files \
    --fastq-file data/chandak/nanopore_dna_storage_data/fastq/merged.fastq \
    --output-dir data/chandak/processed_data \
    --num-workers 50
```

By default uses all CPU cores. Adjust `--num-workers` as needed. Produces `processed_data/experiment_{0..12}/`.

## 3. Cluster reads to ground truth

```bash
python data/chandak/cluster_reads.py \
    --processed-dir data/chandak/processed_data
```

This clusters noisy reads to their ground truth sequences via prefix matching and produces `clustered_data_edit_dist_0/`.

## 4. Format for TReconLM

Run `process_chandak_data.ipynb` to create train/val/test splits in TReconLM format. This produces `final_data/` containing:

- `train.txt`, `val.txt`, `test.txt` — formatted text splits
- `ground_truth.txt`, `reads.txt` — raw sequences
- `test_x.pt`, `test_y.pt` — encoded tensors for inference
