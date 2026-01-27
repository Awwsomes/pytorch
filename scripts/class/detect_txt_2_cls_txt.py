import os
from tqdm import tqdm

def convert_to_cls_txt(detect_labels_path,output_path):
    list_labels = os.listdir(detect_labels_path)
    list_labels = [x for x in list_labels if os.path.isfile(os.path.join(detect_labels_path,x))]
    list_labels = [x for x in list_labels if x.endswith(".txt")]

    for label in tqdm(list_labels):
        path_old_label = os.path.join(detect_labels_path,label)
        path_new_label = os.path.join(output_path,label)
        with open(path_old_label,'r') as old_label, \
             open(path_new_label,'x') as new_label:
            label_idx = old_label.readline().split(" ")[0]
            new_label.write(label_idx)

if __name__ == "__main__":
    detect_txt_path = r"D:\A_myData\dataset\juanZhou_gazebo2\labels-detect"
    cls_txt_path = r"D:\A_myData\dataset\juanZhou_gazebo2\labels"

    if not os.path.exists(cls_txt_path):
        os.mkdir(cls_txt_path)

    convert_to_cls_txt(detect_txt_path,cls_txt_path)