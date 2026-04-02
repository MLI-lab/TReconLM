from imports import *
import helper
from inference_DNAformer import run_inference


#####################################################################
# Set deterministic conditions
seed = 1 
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

#####################################################################
# Load model
from model_DNAFormer_siamese import net

class config:

    #####################################################################
    # Define data generator       
    read_padding    = 'end'      # [end ,symmetric]
    model_config    = 'siamese'  # [single, siamese]

    #####################################################################
    # Data parameters
    label_length           = 60
    corrupt_max_deviation  = 0
    noisy_copies_length    = 80
    min_number_per_cluster = 2
    max_number_per_cluster = 10
    filter_index=False


    #####################################################################
    # DNAFormer parameters
    n_head             = 32
    activation         = 'gelu'
    num_layers         = 12
    d_model            = 1024
    alignment_filters  = 128
    dim_feedforward    = 2048
    output_ch          = 4
    enc_filters        = 4
    p_dropout          = 0
    class_token        = 0
    use_input_scaling  = False
    train_seed = 100

    #####################################################################
    # Data parameters
    train_data_source= 'gen_sim'
    val_data_source = 'gen_sim'
    label_length           = 60
    noisy_copies_length = 80
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
    # Define inference parameters parameters
    test_batch_size   = 100      # batch size for inference
    nThreads         = 1       # number of threads for data loader to use
    device           = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Device:', device)
    if not torch.cuda.is_available():
            print('Slow inference time, it is recommended to use a GPU')

    #####################################################################
    # Set paths 
    save_flag       = False
    save_path       = './results/'
    data_path       = None

    ################################### Loading best model & Artifact for inference #####################
    pretrained_path= ''  # set to your model checkpoint path
    load_wandb = True  # or False
    test_artifact_name = "test_dataset_seed34721_gl60_bs800_ds50000"
    download_dir = "./artifact_data/"
    entity = ""  # your wandb username or team name
    test_project = "TRACE_RECONSTRUCTION"
    wandb_log = True
    project = "Baselines"
    out_dir= ''  # set to your output directory
    val_dataset_size=1 # dummy
    num_workers=1



def inference_run():
    #####################################################################
    # Load model
    model = net(config)
    helper.print_network(model)

    #####################################################################
    # Run training
    run_inference(config, model)
  
if __name__ == "__main__":
    inference_run()