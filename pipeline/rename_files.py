import os
import sys


def rename_files(directory):
    existing_filenames = set(os.listdir(directory))
    for filename in os.listdir(directory):
        new_filename = filename.replace(" ", "_")
        if new_filename != filename:
            original_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            if new_filename in existing_filenames:
                # Handle collision by appending a unique number
                base, ext = os.path.splitext(new_filename)
                counter = 1
                while new_filename in existing_filenames:
                    new_filename = f"{base}_{counter}{ext}"
                    new_path = os.path.join(directory, new_filename)
                    counter += 1
                existing_filenames.add(new_filename)
            else:
                existing_filenames.remove(filename)
                existing_filenames.add(new_filename)
            print(f"Renaming '{filename}' to '{new_filename}'")
            try:
                os.rename(original_path, new_path)
            except Exception as e:
                print(f"Error renaming '{filename}': {e}")
                sys.exit(1)
    print("File renaming completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_files.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    rename_files(directory)
