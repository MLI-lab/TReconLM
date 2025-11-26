from imports import *
import utils
#import utils as utils
from loss import loss_function
import helper

from train_DNAformer import run_train_ddp


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
    train_data_source= 'gen_sim'
    val_data_source = 'gen_sim'
    label_length           = 180
    noisy_copies_length = 240
    min_number_per_cluster = 2 
    max_number_per_cluster = 10 
    min_cluster_size_for_false_copies = 0 

    insertion_probability_lb = 0.01
    deletion_probability_lb = 0.01
    substitution_probability_lb = 0.01

    insertion_probability_ub=0.1
    deletion_probability_ub= 0.1
    substitution_probability_ub = 0.1

    #####################################################################
    # Define loss
    loss_type = 'ce_consistency' # [ce_consistency] #Defines what kind of loss is used (ce_consistency: cross-entropy + consistency loss).

    # Loss coefficients: ce + consistency
    ce_const_coeff_ce  = 1.0 # Weight for cross-entropy (classification) part of the loss.
    ce_const_coeff_const = 0.25 # Weight for consistency loss (forces similar outputs across noisy copies).

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
    p_dropout          = 0
    class_token        = 0
    use_input_scaling  = False
    weight_decay       = 0

    #####################################################################
    # Define training parameters
    nEpochs          = 1        # number of epochs
    lrMax            = 3e-5 #9.185586535436918e-05 #3e-5  # 
    lrMin            = 1e-7 # 3.0618621784789724e-07   # 
    lrCycles         = 1        # odd number # controls how often the learning rate goes from max -> min -> max -> min, etc.
    batch_size       = 64 #600 #64       # batch size
    nThreads         = 12       # number of threads for data loader to use

    #################################Path to save model checkpoints ##############
    out_dir= ''  # set to your output directory
    init_from = 'scratch'
    #train_time = '' # Set if wanna resume from checkpoint
    always_save_checkpoint = True  
    train_seed = 100 

    ###########################Set more configs##############################
    max_iters= 2867988 #305919 # 2867988 
    log_interval= 10
    eval_interval = 500

    ################WAND configs############################################

    wandb_log = True              # to enable wandb logging
    wandb_project = "Baselines"    # your wandb project name
    wandb_entity = "" # your wandb username or team name
    test = True
    device = 'cuda:0' # dummy device




#####################################################################
#####################################################################
# Load model
model = net(config)
helper.print_network(model) # prints number of model parameters

#####################################################################
# Define loss
config.loss = loss_function(config) # Custom loss module based on your config settings.
config.loss_items = helper.get_loss_items(config) # When training, will call this to compute the loss

#####################################################################
# Run training
run_train_ddp(config, model)
