import os
import sys
import argparse
import pandas as pd
import datetime
import torch
from src.embed_deepNGS import *
from src.utils_deepNGS import *
from src.clusterLabel_deepNGS import *



def initialize_gpu_devices(num_devices):
    """Initialize GPU devices."""
    devices = [torch.device(f'cuda:{i}') for i in range(num_devices)]
    torch.cuda.synchronize()
    return devices


def setup_output_directory(directory):
    """Create the output directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)
    return directory


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DeepNGS Navigator Pipeline")
    
    # General parameters
    parser.add_argument("--desc", default="test", type=str, help="Description for the run")
    parser.add_argument("--fileName", required=True, type=str, help="Path to input data file")
    parser.add_argument("--len_group", default='', type=str, help="Length group configuration:'length_cdr3length'")
    parser.add_argument("--length", default=-1, type=int)
    parser.add_argument("--wandb_project", default="deepngs_training", type=str, help="WandB project name")
    parser.add_argument("--wandb_entity", default="", type=str, help="WandB entity name")

    # Model and training parameters
    parser.add_argument("--max_nn", default=5000, type=float)
    parser.add_argument("--d_max", default=-1, type=int)
    parser.add_argument("--d_max_cdr3", default=-1, type=int)
    parser.add_argument("--nn_estimate",default=10,type=int) # the average of neighbours we expect each sequnces to have (if edit distance cut off is not defined(-1), it is configured based on this metric)

    parser.add_argument("--model_type", default="bert", type=str, help="Model type (e.g., bert)")
    parser.add_argument("--pretrained_model_path", default="", type=str, help="Path to pretrained model")
    parser.add_argument("--bs", default=256, type=int, help="Batch size")
    parser.add_argument("--d_embed", default=256, type=int, help="Embedding dimension")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--patience", default=10, type=int, help="Patience for early stopping")
    parser.add_argument("--max_epochs", default=50, type=int, help="Maximum number of epochs")

    # Clustering parameters
    parser.add_argument("--clustering_method", default="LM-deepngs", type=str, help="Clustering method to use. Options: 'LM-tsne', 'LM-umap',  'ablang-umap'. Default is 'LM-deepngs'.")
    parser.add_argument("--loss_function", default="diagonal_positive_sum_negative", type=str, help="Contrastive Loss function, Options: diagonal_positive_sum_negative, weighted_by_mask_size_pos_neg")
    parser.add_argument("--pretrained_deepngs_model_path", default="", type=str, help="Path to pretrained DeepNGS model")
    parser.add_argument("--assign_cluster_label", default=False, type=bool, help="Assign cluster labels based on Leiden")
    
    # Additional parameters
    parser.add_argument("--num_devices", default=1, type=int, help="Number of GPU devices to use")
    parser.add_argument("--min_distance", default=1, type=float, help="Minimum distance for clustering")

    return parser.parse_args()


def load_dataframe(file_path):
    """Load data from a CSV file."""
    dataframe = pd.read_csv(file_path)
    print(f"Loaded dataframe with shape: {dataframe.shape}")
    print(dataframe.head())
    return dataframe


def initialize_main(args, dataframe):
    """Initialize the main processing class."""
    return Main(desc=args.desc, len_group=args.len_group, dataframe=dataframe)


def run_clustering(main, args):
    """Run clustering based on the specified method."""
    if args.clustering_method == 'ablang-umap':
        return main.train_embedding_projecting_model_ablang()
    elif args.clustering_method == 'LM-umap':
        return main.train_embedding_projecting_model_umap(args.pretrained_model_path, args.max_nn, 1, 2)
    elif args.clustering_method == 'LM-tsne':
        return main.train_embedding_projecting_model_tSNE(args.pretrained_model_path, 1, 2)
    
    else:  # Default: Run deepNGS with contrastive loss
        return main.train_embedding_projecting_model(
            desc=args.desc,
            length=args.length,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            fileName=args.fileName,
            max_nn=args.max_nn,
            d_max=args.d_max,
            d_max_cdr3=args.d_max_cdr3,
            nn_estimate=args.nn_estimate,
            pretrained_model_path=args.pretrained_model_path,
            bs=args.bs,
            d_embed=args.d_embed,
            lr=args.lr,
            model_type=args.model_type,
            patience=args.patience,
            len_group=args.len_group,
            max_epochs=args.max_epochs,
            clustering_method=args.clustering_method,
            output_name=args.output_name,
            pretrained_deepngs_model_path=args.pretrained_deepngs_model_path,
            loss_function=args.loss_function,
            num_devices=args.num_devices
        )


def save_results(df, output_dir, file_name):
    """Save clustering results to a compressed CSV file."""
    output_path = os.path.join(output_dir, f"{file_name}.csv.gz")
    df.to_csv(output_path, compression="gzip", index=False)
    print(f"Results saved to {output_path}")


def main():
    # Parse arguments
    args = parse_arguments()
    args.output_name = args.desc + '-' + datetime.datetime.now().strftime("%Y%m%d")
    
    # Setup directories and devices
    store_dir = setup_output_directory("outputs/deepngs/models/")
    devices = initialize_gpu_devices(args.num_devices)

    # Load data
    dataframe = load_dataframe(args.fileName)

    # Initialize main processing class
    main_process = initialize_main(args, dataframe)

    # Pretrain model if required
    if not args.pretrained_model_path and 'ablang' not in args.clustering_method:
        print(f"Starting training Bert LM.... ")
        
        args.pretrained_model_path = str(main_process.pretrain_model(
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            bs=args.bs,
            lr=args.lr,
            patience=args.patience,
            d_embed=args.d_embed,
            num_devices=args.num_devices
        ))
        print("Bert LM Training completed successfully.")
    print(f"Using pretrained model: {args.pretrained_model_path}")

    # Run clustering
    print("Starting 2D Training...")
    result_df = run_clustering(main_process, args)
    print("2D Training completed successfully.")
    if args.assign_cluster_label:
        print("Starting assigning cluster labels...")
        result_df=LeidenClustering(result_df).fit_leiden()
    # Save results
    print("Saving results...")
    save_results(result_df, "outputs/deepngs", args.output_name)
    time.sleep(2.5)  # Wait for the file to be saved
    # Generate and save plots
    print("Generating and saving plots...")
    utils = Utils()
    utils.set_path_and_arguments(args.output_name)
    utils.preprocess_data(args.output_name, args.desc)
    final_df = utils.plot_analysis_figures()


if __name__ == "__main__":
    main()
