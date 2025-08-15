

#!/bin/bash

# Check arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: bash replace_all.sh <directory> <find_string> <replace_string>"
    exit 1
fi

# Inputs
target_dir="$1"
find_str="$2"
replace_str="$3"

# Validate directory
if [ ! -d "$target_dir" ]; then
    echo "Error: Directory '$target_dir' does not exist."
    exit 1
fi

# Get absolute paths
target_dir="$(realpath "$target_dir")"
parent_dir="$(dirname "$target_dir")"
base_name="$(basename "$target_dir")"

# Backup outside the target
timestamp=$(date +%Y%m%d%H%M%S)
backup_dir="${parent_dir}/${base_name}_backup_${timestamp}"

echo "Creating backup at: $backup_dir"
cp -r "$target_dir" "$backup_dir"

echo "Backup created at $backup_dir"
echo "Starting replacements..."

# Replace inside files
grep -rl --exclude-dir=".git" "$find_str" "$target_dir" | while IFS= read -r file; do
    sed -i "s/$find_str/$replace_str/g" "$file"
done

# Rename files
find "$target_dir" -depth -type f -name "*$find_str*" | while IFS= read -r file; do
    newfile="$(dirname "$file")/$(basename "$file" | sed "s/$find_str/$replace_str/g")"
    mv "$file" "$newfile"
done

# Rename directories
find "$target_dir" -depth -type d -name "*$find_str*" | while IFS= read -r dir; do
    newdir="$(dirname "$dir")/$(basename "$dir" | sed "s/$find_str/$replace_str/g")"
    mv "$dir" "$newdir"
done

echo "All done!"

