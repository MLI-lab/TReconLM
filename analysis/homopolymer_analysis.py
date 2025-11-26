import csv
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.stats import binomtest

try:
    from .helpers import find_homopolymers, get_edit_operations
except ImportError:
    from helpers import find_homopolymers, get_edit_operations

def classify_hp_operation(operation, homopolymers):
    """
    Classify how an edit operation relates to homopolymers.
    
    Args:
        operation: Tuple like ('ins', pos, base) or ('del', pos, base) or ('sub', pos, old, new)
        homopolymers: List of (start, end, base) tuples for HPs in ground truth
    
    Returns:
        Classification string:
        
        'hp_expansion': Model incorrectly made the homopolymer longer
            Example: GT:   "...TAAAT..."  (3 A's)
                    Pred: "...TAAAAT..." (4 A's)
                    Operation: ('ins', pos=4, 'A')
                    The model inserted an extra 'A' into or adjacent to the HP
        
        'hp_contraction': Model incorrectly made the homopolymer shorter
            Example: GT:   "...TAAAT..."  (3 A's)
                    Pred: "...TAAT..."   (2 A's)
                    Operation: ('del', pos=3, 'A')
                    The model deleted an 'A' from the HP
        
        'hp_interruption': Model interrupted the homopolymer with a different base
            Example: GT:   "...TAAAT..."  (3 A's)
                    Pred: "...TACAT..."  (A→C in middle, or inserted C)
                    Operation: ('sub', pos=3, 'A', 'C') or ('ins', pos=3, 'C')
                    The model broke up the HP with a different base
        
        'hp_boundary': Error at the edge of a homopolymer
            Example: GT:   "...TAAACG..."  (AAA followed by C)
                    Pred: "...TAAATG..."  (C→T right after HP)
                    Operation: ('sub', pos=4, 'C', 'T')
                    Error immediately before/after HP (often alignment artifacts)
        
        'non_hp': Operation doesn't affect any homopolymer
            Example: GT:   "...TACGAT..."  (no homopolymers)
                    Pred: "...TACGCT..."  (A→C in non-HP region)
                    Operation: ('sub', pos=4, 'A', 'C')
                    Error in normal sequence region
    """
    op_type = operation[0] # error type 
    pos = operation[1] # error position 
    
    for hp_start, hp_end, hp_base in homopolymers:
        
        if op_type == 'ins':
            inserted_base = operation[2]
            # Insertion positions: "before" the GT position
            # hp_end is EXCLUSIVE (one past the last HP position)

            # Within HP (but not at the end boundary)
            if hp_start <= pos < hp_end:
                if inserted_base == hp_base:
                    return 'hp_expansion'  # Incorrectly expanded the HP
                else:
                    return 'hp_interruption'  # Interrupted HP with different base
            # Just before HP
            elif pos == hp_start - 1 and inserted_base == hp_base:
                return 'hp_expansion'  # Extended HP at the start
            # Just after HP (hp_end is already exclusive, so this is the position right after)
            elif pos == hp_end and inserted_base == hp_base:
                return 'hp_expansion'  # Extended HP at the end
            # Near HP but with different base (boundary errors)
            elif (pos == hp_start - 1 or pos == hp_end) and inserted_base != hp_base:
                return 'hp_boundary'
                    
        elif op_type == 'del':
            deleted_base = operation[2]
            # Deletion strictly within HP
            # hp_end is EXCLUSIVE
            if hp_start <= pos < hp_end:
                if deleted_base == hp_base:
                    return 'hp_contraction'  # Incorrectly shortened the HP
                else:
                    # This shouldn't happen if HP is correctly identified in GT (otherwise would not be a HP)
                    print(f"Something is wrong")
                    return 'hp_interruption'  # Strange case - different base was in HP
                    
        elif op_type == 'sub':
            old_base, new_base = operation[2], operation[3]
            # Substitution within HP
            # hp_end is EXCLUSIVE
            if hp_start <= pos < hp_end:
                if old_base == hp_base:
                    return 'hp_interruption'  # Changed HP base to something else
                # else: shouldn't happen if HP correctly identified
            # Substitution at boundaries (before HP or right after - hp_end is exclusive)
            elif pos == hp_start - 1 or pos == hp_end:
                return 'hp_boundary'
    
    return 'non_hp'  # Operation doesn't affect any homopolymer

def analyze_edit_operations(filepath):
    """
    Analyze errors using actual edit operations rather than position-by-position.
    """
    # Detailed counters
    total_operations = 0
    hp_operations = 0
    hp_expansions = 0
    hp_contractions = 0
    hp_interruptions = 0  # Changed from hp_substitutions
    hp_boundary = 0
    operation_types = defaultdict(int)
    
    # Track HP error patterns by HP length
    hp_errors_by_length = defaultdict(lambda: defaultdict(int))
    
    # For statistical test
    all_sequence_lengths = []
    all_hp_lengths = []
    
    # Track sequences with perfect HP reconstruction
    total_sequences = 0
    sequences_with_hp_errors = 0 # sequences that had at least one HP error
    sequences_with_homopolymers = 0 # sequences that contain at least one HP (≥3 bases)
    
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gt = row["ground_truth"]
            pred = row["prediction"]
            total_sequences += 1
            
            # Find homopolymers in ground truth
            homopolymers = find_homopolymers(gt)

            # Track if this sequence contains any homopolymers
            if homopolymers:
                sequences_with_homopolymers += 1

            # Get actual edit operations
            operations = get_edit_operations(gt, pred)
            
            # Track sequence composition
            all_sequence_lengths.append(len(gt))
            hp_length = sum(hp[1] - hp[0] for hp in homopolymers)
            all_hp_lengths.append(hp_length)
            
            # Track if this sequence has any HP errors
            has_hp_error = False
            
            # Analyze each operation
            for op in operations:
                total_operations += 1
                operation_types[op[0]] += 1
                
                # Classify the operation
                classification = classify_hp_operation(op, homopolymers)
                
                if classification != 'non_hp':
                    hp_operations += 1
                    has_hp_error = True
                    
                    # Track specific HP error type
                    if classification == 'hp_expansion':
                        hp_expansions += 1
                    elif classification == 'hp_contraction':
                        hp_contractions += 1
                    elif classification == 'hp_interruption':
                        hp_interruptions += 1
                    elif classification == 'hp_boundary':
                        hp_boundary += 1
                    
                    # Track by HP length (find which HP was affected)
                    for hp_start, hp_end, hp_base in homopolymers:
                        pos = op[1]
                        if hp_start - 1 <= pos <= hp_end:
                            hp_len = hp_end - hp_start
                            hp_errors_by_length[hp_len][classification] += 1
                            break
            
            if has_hp_error:
                sequences_with_hp_errors += 1
    
    # Calculate statistics
    total_seq_length = sum(all_sequence_lengths)
    total_hp_length = sum(all_hp_lengths)
    hp_fraction = total_hp_length / total_seq_length if total_seq_length > 0 else 0
    print(f"HP fraction: {hp_fraction:.2%} ({total_hp_length}/{total_seq_length})")

    hp_operation_rate = hp_operations / total_operations if total_operations > 0 else 0
    enrichment = hp_operation_rate / hp_fraction if hp_fraction > 0 else 0

    # Statistical test: Are homopolymers significantly more error-prone?
    # Null hypothesis: errors occur uniformly across sequence (HP error rate = hp_fraction)
    if total_operations > 0 and hp_fraction > 0:
        stat_test = binomtest(
            k=hp_operations,           # observed HP errors
            n=total_operations,        # total errors
            p=hp_fraction,             # expected rate under null hypothesis
            alternative='greater'      # testing if HP error rate is HIGHER than expected
        )
        p_value = stat_test.pvalue
    else:
        p_value = 1.0
    
    # Print results
    print("="*60)
    print("Edit operation analysis for homopolymers")
    print("="*60)

    print(f"\nDataset overview:")
    print(f"  Total sequences: {total_sequences:,}")
    #print(f"  Sequences containing homopolymers: {sequences_with_homopolymers:,} ({sequences_with_homopolymers/total_sequences:.1%})")
    #print(f"  Sequences with HP errors: {sequences_with_hp_errors:,} ({sequences_with_hp_errors/total_sequences:.1%} of all sequences)")

    # Show HP error rate among sequences that actually have HPs
    #if sequences_with_homopolymers > 0:
    #    hp_error_rate_among_hp_seqs = sequences_with_hp_errors / sequences_with_homopolymers
    #    sequences_with_perfect_hp = sequences_with_homopolymers - sequences_with_hp_errors
    #    print(f"  Sequences with perfect HP reconstruction: {sequences_with_perfect_hp:,} ({sequences_with_perfect_hp/sequences_with_homopolymers:.1%} of HP-containing sequences)")
    #    print(f"  → HP error rate among HP-containing sequences: {hp_error_rate_among_hp_seqs:.1%}")
    
    print(f"\nSequence composition:")
    print(f"  Homopolymer regions: {hp_fraction:.1%} of sequence positions")
    print(f"  How many nucleotides are part of a HP on average in a sequenece: {np.mean(all_hp_lengths):.1f}")

    print(f"\nEdit operations:")
    print(f"  Total operations: {total_operations:,}")
    for op_type in ['sub', 'del', 'ins']:
        if op_type in operation_types:
            count = operation_types[op_type]
            print(f"    {op_type:12s}: {count:,} ({count/total_operations:.1%})")

    print(f"\nHomopolymer-related operations:")
    print(f"  Total HP operations: {hp_operations:,} ({hp_operation_rate:.1%} of all operations)")
    print(f"  Enrichment factor: {enrichment:.2f}x")
    print(f"  Statistical significance: p = {p_value:.4e} {'(*)' if p_value < 0.05 else '(ns)'}")

    if p_value < 0.05:
        if enrichment > 1.2:
            print(f"    Homopolymers are significantly {enrichment:.1f}x more error-prone")
        elif enrichment < 0.8:
            print(f"    Homopolymers are significantly {1/enrichment:.1f}x less error-prone")
    else:
        print(f"    Not statistically significant (could be due to chance)")

    print(f"\nHP error type breakdown:")
    if hp_operations > 0:
        print(f"  Expansions:     {hp_expansions:,} ({hp_expansions/hp_operations:.1%}) - model adds bases")
        print(f"  Contractions:   {hp_contractions:,} ({hp_contractions/hp_operations:.1%}) - model removes bases")
        print(f"  Interruptions:  {hp_interruptions:,} ({hp_interruptions/hp_operations:.1%}) - model breaks up HPs")
        print(f"  Boundary:       {hp_boundary:,} ({hp_boundary/hp_operations:.1%}) - errors at HP edges")

        # Determine primary failure mode
        if hp_expansions > hp_contractions * 1.5:
            print(f"\n  Primary failure mode: model over-extends homopolymers")
        elif hp_contractions > hp_expansions * 1.5:
            print(f"\n  Primary failure mode: model under-counts homopolymer length")
        elif hp_interruptions > (hp_expansions + hp_contractions):
            print(f"\n  Primary failure mode: model interrupts homopolymers with wrong bases")

    # Show errors by HP length
    if hp_errors_by_length:
        print(f"\nErrors by homopolymer length:")
        for length in sorted(hp_errors_by_length.keys()):
            errors = hp_errors_by_length[length]
            total = sum(errors.values())
            print(f"  Length {length}: {total} errors", end="")
            if total > 0:
                main_type = max(errors.items(), key=lambda x: x[1])[0]
                print(f" (mainly {main_type})")
            else:
                print()
    
    return {
        'total_operations': total_operations,
        'hp_operations': hp_operations,
        'enrichment': enrichment,
        'p_value': p_value,
        'hp_expansions': hp_expansions,
        'hp_contractions': hp_contractions,
        'hp_interruptions': hp_interruptions,
        'hp_boundary': hp_boundary,
        'hp_errors_by_length': dict(hp_errors_by_length)
    }

# Run the analysis
if __name__ == "__main__":
    # Set your prediction output file path
    all_predictions_file = ""  # path to predictions_output.tsv (e.g., '/mnt/.../synthetic_L110/predictions_output.tsv')
    results = analyze_edit_operations(all_predictions_file)