
from create_file_list import create_file_list
from sort_files import sort_files
import nbformat as nbf
import os
import re



def create_notebook_from_files(filepaths):
    # Create a new notebook object
    nb = nbf.v4.new_notebook()
    cells = []

    # Extract base filenames without extension for local import checks
    base_filenames = {os.path.basename(fp).replace('.py', '') for fp in filepaths}

    # Regular expression to detect import statements
    import_pattern = re.compile(r'from\s+([\.\w]+)\s+import\s+')

    # Iterate through each file in the given order
    for filepath in filepaths:
        # Read the content of the file
        with open(filepath, 'r') as file:
            content = file.readlines()

        # Filter out local imports
        new_content = []
        for line in content:
            if "from action_selection_rules import action_selection" in line:
                continue
            if "      action_selection.custom_action_selection,\n" in line:
                line = "      custom_action_selection,\n"

            match = import_pattern.match(line)
            if match:
                # Check if the import is local by examining all parts of the import path
                imported_module_parts = match.group(1).split('.')
                if any(part in base_filenames for part in imported_module_parts):
                    continue
            new_content.append(line)

        # Create a comment indicating the file name
        comment = f"# {os.path.basename(filepath)}"

        # Create a new cell with the content
        cell_content = f"{comment}\n{''.join(new_content)}"
        cell = nbf.v4.new_code_cell(cell_content)
        cells.append(cell)

    # Add cells to the notebook
    nb['cells'] = cells

    # first cell should be pip intalls
    pip_installs = nbf.v4.new_code_cell("!pip install mctx\n!pip install jumanji\n!pip install flashbax")
    cells.insert(0, pip_installs)

    # Write the notebook to a file
    with open('output_notebook.ipynb', 'w') as f:
        nbf.write(nb, f)

# Define the root directory
root_dir = './'
file_paths = create_file_list(root_dir)
sorted_files = sort_files(file_paths)
# put '../TUDM/IDMP/Codebase/tree_policies.py' second to last
sorted_files.remove('./tree_policies.py')
sorted_files.remove('./Main.py')
sorted_files.append('./tree_policies.py')
sorted_files.append('./Main.py')

create_notebook_from_files(sorted_files)
