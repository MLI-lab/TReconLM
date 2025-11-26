import torch

#Load model checkpoints
path_dynamic = '/workspaces/TReconLM/Multi-Read-Reconstruction/examples/train_para_dynamic_1.pth'
path_fixed   = '/workspaces/TReconLM/Multi-Read-Reconstruction/examples/train_para_fixed_1.pth'

dynamic_state = torch.load(path_dynamic, map_location='cpu')
fixed_state   = torch.load(path_fixed,   map_location='cpu')

#Load train losses
loss_path_dynamic = '/workspaces/TReconLM/Multi-Read-Reconstruction/examples/train_loss_dynamic.pt'
loss_path_fixed   = '/workspaces/TReconLM/Multi-Read-Reconstruction/examples/train_loss_fixed.pt'

dynamic_loss = torch.load(loss_path_dynamic, map_location='cpu')
fixed_loss   = torch.load(loss_path_fixed,   map_location='cpu')

def compare_state_dicts(state_dict1, state_dict2, atol=1e-6):
    """Compare two state dicts with a given absolute tolerance, reporting max difference if mismatch."""
    # Check if both contain the same keys
    if state_dict1.keys() != state_dict2.keys():
        print("Different parameter keys found!")
        return

    all_match = True
    for key in state_dict1:
        p1 = state_dict1[key]
        p2 = state_dict2[key]
        # Compare parameters with torch.allclose
        if not torch.allclose(p1, p2, atol=atol):
            # If mismatch, compute the max difference for debugging
            diff = (p1 - p2).abs().max()
            print(f"Mismatch found at layer: {key}, max difference = {diff.item():.6g}")
            all_match = False

    if all_match:
        print(f"All model parameters match (within atol={atol})!")

def compare_losses(loss1, loss2, atol=1e-6):
    """Compare two dictionaries of losses (per epoch) with a given absolute tolerance, reporting max difference if mismatch."""
    # Check if both store the same epoch keys
    if loss1.keys() != loss2.keys():
        print("Different epochs stored in loss dictionaries!")
        return

    all_match = True
    for epoch in loss1:
        loss_list1 = loss1[epoch]
        loss_list2 = loss2[epoch]

        if len(loss_list1) != len(loss_list2):
            print(f"Different number of loss entries at epoch {epoch}!")
            all_match = False
            continue

        for i, (l1, l2) in enumerate(zip(loss_list1, loss_list2)):
            # Convert Python floats/lists to tensors for comparison
            t1 = torch.tensor(l1, dtype=torch.float32)
            t2 = torch.tensor(l2, dtype=torch.float32)
            if not torch.allclose(t1, t2, atol=atol):
                diff = (t1 - t2).abs().max()
                print(f"Loss mismatch at epoch {epoch}, batch {i}: "
                      f"{l1} vs {l2}, max diff={diff.item():.6g}")
                all_match = False

    if all_match:
        print(f"All training losses match (within atol={atol})!")

# Run comparisons
print("\nComparing model parameters:")
compare_state_dicts(dynamic_state, fixed_state, atol=1e-6)

print("\nComparing training losses:")
compare_losses(dynamic_loss, fixed_loss, atol=1e-6)
