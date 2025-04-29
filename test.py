import os
import re

def sanitize_name(name):
    # Replace problematic characters
    name = name.replace("&", "and")
    name = name.replace(";", "")  # remove semicolon
    name = re.sub(r"[^\w\s.-]", "", name)  # remove anything else weird
    return name

root_dir = "mo_db"

for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
    # Rename files
    for filename in filenames:
        new_filename = sanitize_name(filename)
        if new_filename != filename:
            os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, new_filename))
            print(f"Renamed file.........................: {filename} to {new_filename}")

    # Rename directories
    for dirname in dirnames:
        new_dirname = sanitize_name(dirname)
        if new_dirname != dirname:
            os.rename(os.path.join(dirpath, dirname), os.path.join(dirpath, new_dirname))
            print(f"Renamed directory======================: {dirname} to {new_dirname}")

            
            
# Abstract_Paisley_Fit_&_Flare_Dress