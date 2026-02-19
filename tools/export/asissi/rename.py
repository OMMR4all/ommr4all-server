import os
from glob import glob
import re
def extract_number(filename):
    # Find all number sequences in the filename
    matches = re.findall(r'\d+', filename)
    if matches:
        # Return the first number found as an integer
        return int(matches[0])
    return 0
folder_path = "/tmp/images/"
print(os.listdir(folder_path))
files = sorted(glob(folder_path + "*.png"), key=extract_number)
i = -1
v = False
for filename in files:

    if filename == "rename.py":
        continue
    print(f"renaming: {filename}")
    file_extension = os.path.splitext(filename)[1]

    if v == False:
        new_name = f"folio_{i:04d}{file_extension}"
    else:
        new_name = f"folio_{i:04d}v{file_extension}"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)

    os.rename(src, dst)
    print(f"Renamed: {filename} -> {new_name}")
    if v == True:
        i += 1
        v = False
    else:
        v = True