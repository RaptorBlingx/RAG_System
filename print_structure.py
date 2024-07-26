import os

def print_project_structure(root_dir, indent=""):
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item not in (".git", "venv"):  # Exclude both
            print(f"{indent}├── {item}/")
            print_project_structure(item_path, indent + "│   ")
        elif item not in (".git", "venv"):
            print(f"{indent}├── {item}")

root_directory = "."  # Current directory
print_project_structure(root_directory)
