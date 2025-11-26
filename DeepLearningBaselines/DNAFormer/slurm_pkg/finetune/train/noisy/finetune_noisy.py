from imports import *
import helper
#import utils as utils
from loss import loss_function
from finet_DNAformer import finetune


#####################################################################

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#####################################################################
# Load model
from model_DNAFormer_siamese import net

class config:

    #####################################################################
    # Define data generator   
    read_padding    = 'end'      # [end ,symmetric] #Whether sequences are padded at the end or symmetrically,i.e. add half the padding at the beginning and half at the end.
    filter_index    = False  #  If True, part of the sequence (index) is excluded from loss calculation and model output.
    model_config    = 'siamese'  # [single, siamese] # Type of model (single = normal, siamese = split and fuse).

    #####################################################################
    # Data parameters
    label_length           = 60
    noisy_copies_length = 80
    max_number_per_cluster = 10 

    #####################################################################
    #Paths to load val and train data from
    data_folder = ''  # set to your data directory
    filename_train = 'train.txt'
    filename_val = 'val.txt'

    #####################################################################
    # Define loss
    loss_type = 'ce_consistency' # [ce_consistency] #Defines what kind of loss is used (ce_consistency: cross-entropy + consistency loss).

    # Loss coefficients: ce + consistency
    ce_const_coeff_ce  = 1.0 # Weight for cross-entropy (classification) part of the loss.
    ce_const_coeff_const = 0.25  # Weight for consistency loss (forces similar outputs across noisy copies).

    #####################################################################
    # DNAFormer parameters
    n_head             = 32
    activation         = 'gelu'
    num_layers         = 12
    d_model            = 1024
    alignment_filters = 128
    dim_feedforward    = 2048
    output_ch          = 4 # for one hot encoding? 
    enc_filters        = 4
    p_dropout          = 0.3
    class_token        = 0
    use_input_scaling  = False
    weight_decay       = 0

    model_checkpoint=''  # set to your pretrained model checkpoint path

    #####################################################################
    # Define training parameters
    lrMax            = 1e-5
    lrMin            = 1e-8     # 
    lrCycles         = 1        # odd number # controls how often the learning rate goes from max -> min -> max -> min, etc.
    batch_size       = 8       # batch size
    device           = 'cuda' if torch.cuda.is_available() else 'cpu'

    #################################Path to save model checkpoints ##############
    out_dir= ''  # set to your output directory
    init_from = 'scratch'  #'resume', scratch
    #train_time = '20250614_123917' # Set if we want to resume
    always_save_checkpoint = True  # or True if you want
    train_seed = 100 
    test = False

    ###########################Set more configs##############################
    max_iters= 685307  
    log_interval= 10
    eval_interval = 500
    eval_iters= 1000

    ################WAND configs############################################

    wandb_log = True              # to enable wandb logging
    wandb_project = "Baselines"    # your wandb project name
    wandb_entity = ""  # your wandb username or team name
    test_project = "TRACE_RECONSTRUCTION"




#####################################################################
# Load model
model = net(config)
# Load weights if model_checkpoint
if getattr(config, "model_checkpoint", None) and os.path.isfile(config.model_checkpoint):
    print(f"Loading weights from '{config.model_checkpoint}'")
    ckpt = torch.load(config.model_checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    # strip unwanted prefixes
    unwanted_prefix = "module._orig_mod."
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            cleaned_state_dict[k[len(unwanted_prefix):]] = v
        else:
            cleaned_state_dict[k] = v

    # load weights
    #missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    #print(f" loaded with {len(cleaned_state)} tensors "
    #      f"(missing={len(missing)}, unexpected={len(unexpected)})")

    model.load_state_dict(cleaned_state_dict)

helper.print_network(model) # prints number of model parameters

#####################################################################
# Define loss
config.loss = loss_function(config) # Custom loss module based on your config settings.
config.loss_items = helper.get_loss_items(config) # When training, will call this to compute the loss

#####################################################################
# Run training
finetune(config, model)
