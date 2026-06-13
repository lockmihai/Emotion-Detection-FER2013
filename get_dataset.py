import kagglehub

# Download the image-based dataset (train/test folders)
path_images = kagglehub.dataset_download("msambare/fer2013")
print("Path to image dataset files:", path_images)

# Download the CSV-based dataset (fer2013.csv)
path_csv = kagglehub.dataset_download("deadskull7/fer2013")
print("Path to CSV dataset files:", path_csv)
