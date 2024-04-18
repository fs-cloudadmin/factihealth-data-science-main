# -*- coding: utf-8 -*-

import tarfile

# Path to your .pkl file
# pkl_file_path = 'Models/model_30.pkl'

import zipfile
import tarfile
import os

# Path to your existing .zip file
zip_file_path = 'Models/model_30.zip'

# Path to output .tar.gz file
output_tar_file = 'Models/model_30.tar.gz'

# Temporary directory to extract ZIP contents
temp_dir = 'temp_directory'

# Extract contents of the .zip file to a temporary directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Create a .tar.gz file from the extracted contents
with tarfile.open(output_tar_file, 'w:gz') as tar:
    tar.add(temp_dir, arcname=os.path.basename(temp_dir))

# Clean up temporary directory if needed
# os.rmdir(temp_dir)