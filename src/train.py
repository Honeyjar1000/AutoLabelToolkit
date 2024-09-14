import os
import ultralytics
from IPython import display
import torch
import argparse

def main():

    parser = argparse.ArgumentParser(description='Process command line arguments for YOLO training and dataset management.')
    parser.add_argument('--model_id', type=str, required=False, default='steven',
                        help='Identifier for the model.')
    parser.add_argument('--dataset_id', type=str, required=False, default='fruit_big',
                        help='Identifier for the dataset.')
    parser.add_argument('--batch_size', type=int, required=False, default=50, 
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, required=False, default=20, 
                        help='Number of epochs for training.')
    args = parser.parse_args()

    # Determine the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Define relative paths
    dataset_id = args.dataset_id
    # Specify a folder name for the run
    run_name = args.model_id
    batch_size = args.batch_size
    epochs = args.epochs


    train_data_path = os.path.join(base_dir, 'data','labeled', f'{dataset_id}', 'data.yaml')

    # Initialize the YOLO model
    model = ultralytics.YOLO("yolov8s.pt")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", str(device))

    # Train the model
    results = model.train(
        data=train_data_path, 
        batch=batch_size, 
        epochs=epochs, 
        imgsz=320, 
        plots=True, 
        device=device,
        name=run_name,  # Set the custom name for the run
        workers=2
    )

    # Print the folder where the final weights are stored
    print(f"Saved in: {results.save_dir}")

if __name__ == '__main__':
    main()