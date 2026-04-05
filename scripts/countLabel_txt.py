import os
import numpy as np
import json
from tqdm import tqdm

dir_label = r"D:\A_myData\RC26-Vision\dataset\corner7\txts"
list_label_name = ["corner", "trash"]

amount_label = len(list_label_name)
list_labels = os.listdir(dir_label)
count_amount_per_label = np.zeros(amount_label + 1,'int')   # 加一是不在标签列表中的标签

# for label in list_labels:
#     if label == "classes.txt" :
#         continue
#     path_label = dir_label + label
#     with open(path_label,"r") as f :
#         for line in f :
#             idx = line.split(' ')[0]
#             count[int(idx)] += 1
#
# for idx,num in enumerate(count):
#     print("{}: {}".format(idx,num))

for file_label in tqdm(list_labels):
    if not file_label.endswith(".txt"):
        print('note:',file_label,'not a txt.')
        continue
    path_label = os.path.join(dir_label,file_label)
    with open(path_label,'r') as txt_file:
        for line in txt_file:
            idx = int(line.split(' ')[0])
            if idx < amount_label:
                count_amount_per_label[idx] += 1
            else:
                count_amount_per_label[amount_label] += 1
                print('warning:', file_label, 'label', idx, 'not exists.')

print("--------------------------------------------------------")
print("Label Count:")
for idx,count in enumerate(count_amount_per_label):
    if idx == amount_label:
        print('other labels :',count)
        break
    print(list_label_name[idx],':',count)