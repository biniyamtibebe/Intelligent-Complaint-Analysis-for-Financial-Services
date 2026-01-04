import os
import shutil

# Source file (your Downloads folder)
source_path = r"C:\Users\hp\Downloads\complaints.csv\complaints.csv"

# Destination folder inside your project
dest_dir = r"C:\Users\hp\Pictures\financial intelligence\Intelligent-Complaint-Analysis-for-Financial-Services\rag-complaint-chatbot\data\raw"
dest_path = os.path.join(dest_dir, "complaints.csv")

# Create data folder if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Copy file
shutil.copy(source_path, dest_path)

print(f"File copied successfully to {dest_path}")
