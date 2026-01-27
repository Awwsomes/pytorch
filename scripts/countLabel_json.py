import os
import numpy as np
import json
from tqdm import tqdm

dir_label = "D:\\A_myData\\dataset\\juanZhou4\\json2\\"
list_label_name = ['1','2','3','4','5','6','7','8','9','10',
                   '11','12','13','14','15','16','17','18',
                   '19','20','21','22','23','24','25','26',
                   '27','28','29','30','31']

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
    if not file_label.endswith(".json"):
        print('note:',file_label,'not a json.')
        continue
    path_label = os.path.join(dir_label,file_label)
    with open(path_label,'r') as json_file:
        data = json.load(json_file)
        for shape in data["shapes"]:
            label = shape["label"]
            if label in list_label_name:
                idx = list_label_name.index(label)
                count_amount_per_label[idx] += 1
            else:
                count_amount_per_label[amount_label] += 1
                print('warning:',file_label,'label',label,'not exists.')

print("--------------------------------------------------------")
print("Label Count:")
for idx,count in enumerate(count_amount_per_label):
    if idx == amount_label:
        print('other labels :',count)
        break
    print(list_label_name[idx],':',count)