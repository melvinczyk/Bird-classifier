import os


def delete(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
                print(f"Deleted folder: {dir_path}")
            except OSError as e:
                print(f"Error deleting folder {dir_path}: {e}")

    try:
        os.rmdir(folder_path)
        print(f"Deleted root folder: {folder_path}")
    except OSError as e:
        print(f"Error deleting root folder {folder_path}: {e}")


if __name__ == "__main__":
    folder_to_delete = 'dataset'
    delete(folder_to_delete)