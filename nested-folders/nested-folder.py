import os
import shutil
from time import sleep


def create_folders_with_files(base_dir, num_folders=5):
    """
    Create multiple folders, each containing a txt file with a message.

    :param base_dir: The parent directory where folders will be created
    :param num_folders: Number of folders (and txt files) to create
    """
    os.makedirs(base_dir, exist_ok=True)  # Create base directory if it doesn't exist

    for i in range(1, num_folders + 1):
        # Create folder
        folder_name = f"folder_{i}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Create text file inside the folder
        file_path = os.path.join(folder_path, f"text_file_{i}.txt")
        with open(file_path, "w") as f:
            f.write(f"This is Text file {i}")

        print(f"Created: {file_path}")
        sleep(120)
    # Compress the entire base directory
    sleep(50)
    archive_path = os.path.join(base_dir, "archive")
    archive_file = shutil.make_archive(archive_path, 'zip', base_dir)
    print(f"\nCompressed all folders into: {archive_file}")

create_folders_with_files("/data/outputs")