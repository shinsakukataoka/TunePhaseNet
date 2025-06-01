#!/bin/bash

# Check if a directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Check if the provided path is a valid directory
if [ ! -d "$1" ]; then
    echo "Error: '$1' is not a valid directory"
    exit 1
fi

# Function to print directory structure using tree (if available)
print_tree() {
    if command -v tree >/dev/null 2>&1; then
        tree -a "$1"
    else
        echo "Note: 'tree' command not found. Using find command instead."
        find "$1" -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g'
    fi
}

# Print the directory structure
echo "Directory structure for: $1"
print_tree "$1"
