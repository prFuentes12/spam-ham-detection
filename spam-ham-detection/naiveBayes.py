import kagglehub

# Download latest version
path = kagglehub.dataset_download("wanderfj/enron-spam")

print("Path to dataset files:", path)
