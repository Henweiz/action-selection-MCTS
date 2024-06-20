import os

def create_file_list(root_dir, file_list_file='files.txt'):
    with open(file_list_file, 'r') as file:
        file_content = file.read()

    file_list = []

    # Split the content into lines and parse
    current_dir = ""
    for line in file_content.splitlines():
        stripped_line = line.strip()  # Remove leading and trailing whitespaces
        if not stripped_line:  # Skip empty lines
            continue
        indent_level = len(line) - len(stripped_line)
        if indent_level == 0 and stripped_line.endswith('/'):  # It's a root level directory
            current_dir = stripped_line
        elif indent_level > 0:  # It's a file or subdirectory under the current directory
            file_list.append(f"{root_dir}{current_dir}{stripped_line}")
        else:
            # If the line is not indented but not a directory, it means it's a file in the root
            file_list.append(f"{root_dir}{stripped_line}")

    return file_list
