"""
Tutorial utility functions for TReconLM custom data inference.
"""
import pickle
from pathlib import Path

def load_vocabulary():
    """Load vocabulary from pickle file"""
    project_root = Path(__file__).parent.parent
    meta_path = project_root / 'src' / 'data_pkg' / 'meta_nuc.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

def get_model_info(sequence_length, variant='pretrained'):
    """Get model information based on sequence length and variant

    For variable length model, use sequence_length='var_50_120' or 'variable'.
    """
    models = {
        60: {
            'pretrained': {
                'model_name': 'model_seq_len_60.pt',
                'repo_id': 'tracereconstruction2026/TReconLM',
                'block_size': 800,
                'description': 'Pretrained on synthetic IDS data (60nt)',
                'dataset_repo_id': 'tracereconstruction2026/TReconLM_datasets',
                'dataset': 'L60_ctx800_ds50000'
            },
            'noisy_dna': {
                'model_name': 'finetuned_noisy_dna_len60.pt',
                'repo_id': 'tracereconstruction2026/TReconLM',
                'block_size': 800,
                'description': 'Fine-tuned on Noisy DNA dataset (60nt)',
                'dataset_repo_id': 'tracereconstruction2026/TReconLM_datasets',
                'dataset': 'L60_ctx800_ds50000'
            }
        },
        110: {
            'pretrained': {
                'model_name': 'model_seq_len_110.pt',
                'repo_id': 'tracereconstruction2026/TReconLM',
                'block_size': 1500,
                'description': 'Pretrained on synthetic IDS data (110nt)',
                'dataset_repo_id': 'tracereconstruction2026/TReconLM_datasets',
                'dataset': 'L110_ctx1500_ds50000'
            },
            'microsoft': {
                'model_name': 'finetuned_microsoft_dna_len110.pt',
                'repo_id': 'tracereconstruction2026/TReconLM',
                'block_size': 1500,
                'description': 'Fine-tuned on Microsoft DNA dataset (110nt)',
                'dataset_repo_id': 'tracereconstruction2026/TReconLM_datasets',
                'dataset': 'L110_ctx1500_ds50000'
            }
        },
        117: {
            'pretrained': {
                'model_name': 'model_seq_len_110.pt',
                'repo_id': 'tracereconstruction2026/TReconLM',
                'block_size': 1500,
                'description': 'Pretrained on synthetic IDS data (110nt) - closest to 117nt',
                'dataset_repo_id': 'tracereconstruction2026/TReconLM_datasets',
                'dataset': 'L110_ctx1500_ds50000'
            },
            'chandak': {
                'model_name': 'finetuned_chandak_len117.pt',
                'repo_id': 'tracereconstruction2026/TReconLM',
                'block_size': 1500,
                'description': 'Fine-tuned on Chandak dataset (117nt)',
                'dataset_repo_id': 'tracereconstruction2026/TReconLM_datasets',
                'dataset': 'L110_ctx1500_ds50000'
            }
        },
        180: {
            'pretrained': {
                'model_name': 'model_seq_len_180.pt',
                'repo_id': 'tracereconstruction2026/TReconLM',
                'block_size': 2400,
                'description': 'Pretrained on synthetic IDS data (180nt)',
                'dataset_repo_id': 'tracereconstruction2026/TReconLM_datasets',
                'dataset': 'L180_ctx2400_ds50000'
            }
        },
        # Variable length model (50-120nt)
        'var_50_120': {
            'pretrained': {
                'model_name': 'model_var_len_50_120.pt',
                'repo_id': 'tracereconstruction2026/TReconLM',
                'block_size': 2400,
                'description': 'Pretrained on variable length sequences (50-120nt). Best for fine-tuning on custom real-world data.',
                'dataset_repo_id': 'tracereconstruction2026/TReconLM_datasets',
                'dataset': 'L110_ctx1500_ds50000'
            }
        }
    }

    # Handle 'variable' as alias for 'var_50_120'
    if sequence_length == 'variable':
        sequence_length = 'var_50_120'

    if sequence_length not in models:
        raise ValueError(f"No model available for sequence length {sequence_length}. "
                        f"Available: {list(models.keys())}")

    if variant not in models[sequence_length]:
        print(f"Warning: variant '{variant}' not available, using 'pretrained'")
        variant = 'pretrained'

    return models[sequence_length][variant]

def list_available_models():
    """Return markdown table of available models"""
    return """
### Available Models

#### Pretrained Models (Fixed Length)
| Length | Variant | Model Name | Description |
|--------|---------|------------|-------------|
| 60nt | pretrained | model_seq_len_60.pt | Pretrained on synthetic IDS data |
| 110nt | pretrained | model_seq_len_110.pt | Pretrained on synthetic IDS data |
| 180nt | pretrained | model_seq_len_180.pt | Pretrained on synthetic IDS data |

#### Pretrained Models (Variable Length)
| Length | Variant | Model Name | Description |
|--------|---------|------------|-------------|
| 50-120nt | pretrained | model_var_len_50_120.pt | Pretrained on variable length sequences. **Recommended for fine-tuning on custom real-world data.** |

#### Fine-tuned Models
| Length | Variant | Model Name | Description |
|--------|---------|------------|-------------|
| 60nt | noisy_dna | finetuned_noisy_dna_len60.pt | Fine-tuned on Noisy DNA dataset |
| 110nt | microsoft | finetuned_microsoft_dna_len110.pt | Fine-tuned on Microsoft DNA dataset |
| 117nt | chandak | finetuned_chandak_len117.pt | Fine-tuned on Chandak dataset |

**Block sizes:** 60nt=800, 110nt/117nt=1500, 180nt/variable=2400

To fine-tune on your own real-world data without pretraining from scratch, use the variable length pretrained model (`model_var_len_50_120.pt`) as a starting point.
"""

def validate_reads_file(filepath):
    """Validate reads.txt format"""
    valid_chars = set('ACGT\n')
    separator = '==============================='

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Check for invalid characters
        invalid_chars = set(content) - valid_chars - {separator[0], '=', ' ', '\t', '\r'}
        if invalid_chars:
            return {'valid': False, 'error': f'Invalid characters found: {invalid_chars}'}

        # Parse clusters
        clusters = [c.strip().split('\n') for c in content.split(separator) if c.strip()]

        if len(clusters) == 0:
            return {'valid': False, 'error': 'No clusters found'}

        # Validate cluster sizes
        invalid_clusters = [i for i, c in enumerate(clusters) if not (2 <= len(c) <= 10)]

        return {
            'valid': len(invalid_clusters) == 0,
            'num_clusters': len(clusters),
            'invalid_clusters': invalid_clusters,
            'error': None if len(invalid_clusters) == 0 else f'{len(invalid_clusters)} clusters with invalid size'
        }

    except FileNotFoundError:
        return {'valid': False, 'error': 'File not found'}
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def validate_ground_truth_file(filepath, expected_count=None):
    """Validate ground_truth.txt format"""
    valid_chars = set('ACGT')

    try:
        with open(filepath, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]

        # Check count
        if expected_count and len(sequences) != expected_count:
            return {
                'valid': False,
                'error': f'Expected {expected_count} sequences, found {len(sequences)}'
            }

        # Check each sequence
        for i, seq in enumerate(sequences):
            invalid_chars = set(seq) - valid_chars
            if invalid_chars:
                return {
                    'valid': False,
                    'error': f'Invalid characters in sequence {i+1}: {invalid_chars}'
                }

        return {
            'valid': True,
            'num_sequences': len(sequences),
            'sequence_lengths': [len(s) for s in sequences],
            'error': None
        }

    except FileNotFoundError:
        return {'valid': False, 'error': 'File not found'}
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def print_validation_results(validation, file_type):
    """Print validation results"""
    if validation['valid']:
        print(f"[PASS] {file_type}.txt validation passed")
        if 'num_clusters' in validation:
            print(f"  Found {validation['num_clusters']} clusters")
        if 'num_sequences' in validation:
            print(f"  Found {validation['num_sequences']} sequences")
    else:
        print(f"[FAIL] {file_type}.txt validation failed:")
        print(f"  Error: {validation['error']}")
