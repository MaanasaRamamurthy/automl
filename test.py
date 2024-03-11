import os

def download_model(model_path, target_path):
  try:
    with open(model_path, 'rb') as f:
      model_data = f.read()
    with open(target_path, 'wb') as f:
      f.write(model_data)
    print("Model downloaded successfully to project directory.")
  except Exception as e:
    print(f"Error downloading model: {e}")

model_path = "best_model.pkl"
current_dir_path = os.getcwd()
target_path = os.path.join(current_dir_path, "trained_model.pkl")
download_model(model_path, target_path)