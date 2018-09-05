import os
import subprocess
from os.path import dirname, abspath, join

if __name__ == "__main__":
    DATA_FOLDER = join(dirname(dirname(abspath(__file__))), 'data', 'speaker-train', 'dev-clean')
    ORIGINAL_FORMAT = "flac"
    i=0

    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            file_name_list = file.split(".")
            if file_name_list[-1] == ORIGINAL_FORMAT:
                i += 1
                print(i, ":", file)
                cmd='sox "{0}" -r 16000 "{1}"'.format(join(root, file),join(root, file_name_list[0]+".wav"))
                subprocess.run(cmd, shell=True)
    print("*********conversion done************")
