import os
import sys
import shutil
from ultralytics import YOLO

def main():
    print("Starting YOLOv8 training on local aircraft surface damage dataset...")
    
    # Check if dataset yaml exists
    yaml_path = os.path.abspath("data_local.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: Dataset configuration file not found at {yaml_path}")
        sys.exit(1)
        
    print(f"Loading dataset configuration from: {yaml_path}")
    
    # Load pretrained YOLOv8n model
    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # Train the model
    # We use 5 epochs to complete training quickly on CPU.
    # User can modify this value for better accuracy.
    epochs = 5
    print(f"Training YOLOv8n for {epochs} epochs on CPU...")
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=320,
            fraction=0.1, # Train on 3% of the dataset for rapid execution on CPU
            workers=0,     # Prevent overhead of multi-threaded data loading on CPU
            device="cpu",  # Force CPU usage
            project="aircraft_damage",
            name="yolov8_train"
        )
        print("Training completed successfully!")
        
        # Get path of best weights
        best_weights = os.path.join("aircraft_damage", "yolov8_train", "weights", "best.pt")
        if os.path.exists(best_weights):
            print(f"Best weights saved to: {best_weights}")
            # Copy to root directory for easy access
            shutil.copy(best_weights, "best.pt")
            print("Copied best weights to root directory as 'best.pt'")
        else:
            print("Warning: Could not locate trained weights file 'best.pt'.")
            
    except Exception as e:
        print(f"An error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
