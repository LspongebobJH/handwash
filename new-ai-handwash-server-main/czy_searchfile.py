from pathlib import Path
import numpy as np
def find_file(directory, filename):
    for file in Path(directory).rglob('*.mp4'):
        if file.name == filename:
            return file
aa = np.load('norm_results_test.npy',allow_pickle=True)
print(len(aa.item().keys()))
# file_path = find_file('/home/hhyg/handwash/new-ai-handwash-server-main/dataset', '20221007_523_1_washstep4_repeat0.mp4')
# if file_path:
#     print('Found: ', file_path)
# else:
#     print('File not found')