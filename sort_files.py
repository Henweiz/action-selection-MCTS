"""Methods used for turning the codebase into a python notebook"""

import ast
import os
from collections import defaultdict, deque


def extract_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split('.')[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split('.')[-1])
    return imports


def get_python_files(root_dir):
    python_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                python_files.append(os.path.join(dirpath, filename))
    return python_files


def match_local_imports(file_imports, python_files):
    file_names = {os.path.splitext(os.path.basename(file))[0]: file for file in python_files}
    local_dependencies = defaultdict(list)

    for file_path in python_files:
        imports = file_imports.get(file_path, [])
        for imp in imports:
            module = imp.split('.')[-1]
            if module in file_names:
                local_dependencies[file_path].append(file_names[module])
        if file_path not in local_dependencies:
            local_dependencies[file_path] = []  # Ensure every file is included

    return local_dependencies

def build_dependency_graph(local_dependencies):
    graph = defaultdict(list)
    for file in local_dependencies:
        for dep in local_dependencies[file]:
            graph[dep].append(file)
        if file not in graph:
            graph[file] = []
    return graph


def topological_sort(graph):
    in_degree = {u: 0 for u in graph}
    print(in_degree)
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in graph if in_degree[u] == 0])
    sorted_list = []

    while queue:
        u = queue.popleft()
        sorted_list.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_list) == len(graph):
        return sorted_list
    else:
        raise Exception("The graph has at least one cycle.")

def sort_files(python_files):
    file_imports = {file_path: extract_imports(file_path) for file_path in python_files}

    local_dependencies = match_local_imports(file_imports, python_files)

    dependency_graph = build_dependency_graph(local_dependencies)

    sorted_files = topological_sort(dependency_graph)

    # Output the sorted files
    return sorted_files