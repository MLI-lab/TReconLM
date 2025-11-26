# TReconLM

TReconLM is a decoder-only transformer model for trace reconstruction of noisy DNA sequences. It is trained to reconstruct a ground-truth sequence from multiple noisy copies (traces), each independently corrupted by insertions, deletions, and substitutions.  
(For more details, see [our paper](https://arxiv.org/abs/2507.12927).)

---

## Installation

Tested on `Ubuntu 22.04.4 LTS`.

Create the conda environment:

```bash
conda env create -f treconlm.yml
```

Install `build-essential` (required for compiling extensions):

```bash
sudo apt update && sudo apt install -y build-essential
```

Set the Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/treconlm"
```

---

## Inference

Pretrained and fine-tuned models, as well as synthetic test datasets, are available on Hugging Face:

- [Models](https://huggingface.co/mli-lab/TReconLM)
- [Test datasets](https://huggingface.co/datasets/mli-lab/TReconLM_datasets)

### Getting Started

Start with the tutorial notebooks in `tutorial/`:

- [`quick_start.ipynb`](tutorial/quick_start.ipynb): Download models from HuggingFace and run inference on synthetic datasets
- [`custom_data.ipynb`](tutorial/custom_data.ipynb): Run inference on your own data or use the Microsoft/Noisy DNA datasets

### Data Format

For custom data, provide two files:
- `ground_truth.txt`: one DNA sequence per line (ACGT only)
- `reads.txt`:clusters of 2-10 noisy reads separated by `===============================`

See `tutorial/custom_data.ipynb` for details.

---

## Training

### FlashAttention (optional)

To run with FlashAttention for faster training (see [PyTorch issue](https://github.com/pytorch/pytorch/issues/119054)):

```bash
pip install nvidia-cuda-nvcc-cu11
export TRITON_PTXAS_PATH=/opt/conda/envs/treconlm/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas
```

### Pretraining

In `src`, you can run:

```bash
torchrun --nproc_per_node=<available_gpus> pretrain.py exps=...
```

and choose a pretraining experiment from  
`src/hydra/config/train_config/exps` (contains all configurations used in our paper).

Example:

```bash
python pretrain.py exps=test/pretrain_scratch
```

> Use `torchrun` for multi-GPU runs. For single-GPU, `python` also works.  
> Pretraining data is generated on the fly during training.

Example cluster scripts can be found in `/slurm_pkg`.

### Fine-tuning

In `src`, run:

```bash
torchrun --nproc_per_node=<available_gpus> finetune.py exps=...
```

and choose a fine-tuning experiment from `src/hydra/config/train_config/exps`.  
Available experiments include `microsoft_data` and `noisy_dna`.

---

## Example Training Times

- **Pretraining**:  
  Training a ~38M parameter model on ~300M examples (sequence length `L = 110`, cluster sizes uniformly sampled between 2 and 10, totaling ~440B tokens) on 4 NVIDIA H100 GPUs takes approximately 71.1 hours.

- **Fine-tuning**:  
  Fine-tuning a ~38M parameter model on ~5.5M examples (sequence length `L = 60`, cluster sizes between 2 and 10, totaling ~4.39B tokens) takes approximately 20.6 hours.

---

## Data

Configuration files for our synthetic data generation are in:

```text
src/hydra/data_config
```

To generate new test datasets, run from `src`:

```bash
python data_pkg/data_generation.py
```

---

## Baselines

### Non–deep learning baselines

To run inference with non–deep learning baselines:

```bash
python src/eval_pkg/eval_all_baselines.py --alg <algorithm>
```

Available algorithms:

```python
ALGS = {
    'bmala': BMALA,
    'itr': Iterative,
    'muscle': MuscleAlgorithm,
    'trellisbma': TrellisBMAAlgorithm,
    'vs': VSAlgorithm,
}
```

An example cluster script is available in `src/slurm_pkg/baselines`.

### Deep learning baselines

To pretrain, fine-tune, or run inference with our deep learning baselines, see:

- `DeepLearningBaselines/DNAFormer/slurm_pkg`  
- `DeepLearningBaselines/RobuSeqNet/slurm_pkg`

These contain example SLURM execution scripts.

---

### Source Implementations

The original implementations of the baselines were taken from:

- **VS**, **BMALA**, **ITR**: [github.com/omersabary/Reconstruction](https://github.com/omersabary/Reconstruction)  
- **MUSCLE**: [github.com/rcedgar/muscle](https://github.com/rcedgar/muscle)  
- **TrellisBMA**: [github.com/orenht/DNA-trellis-reconstruction](https://github.com/orenht/DNA-trellis-reconstruction)
- **RobuSeqNet**: [github.com/qinyunnn/RobuSeqNet](https://github.com/qinyunnn/RobuSeqNet)  
- **DNAformer**: [github.com/itaiorr/Deep-DNA-based-storage](https://github.com/itaiorr/Deep-DNA-based-storage)  

