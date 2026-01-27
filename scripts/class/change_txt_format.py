import os

def change_txt_format(input_path,output_path):
    """
    从 conf idx 格式 变为 idx conf
    e.g. 1.00 22 -> 22 1.00
    """
    list_txt = os.listdir(input_path)
    list_txt = [x for x in list_txt if os.path.isfile(os.path.join(input_path,x))]
    list_txt = [x for x in list_txt if x.endswith(".txt")]

    for txt in list_txt:
        old_path = os.path.join(input_path,txt)
        new_path = os.path.join(output_path,txt)
        with open(old_path,'r') as old_txt, \
             open(new_path,'x') as new_txt:
            for line in old_txt:
                conf,idx = line.split(" ")
                new_line = f"{idx.strip()} {conf}\n"
                new_txt.write(new_line)

if __name__ == "__main__":
    intput_path = r"D:\A_myData\Pytorch\yolo11_cls\runs\classify\predict2\labels"
    output_path = r"D:\A_myData\Pytorch\yolo11_cls\runs\classify\predict2\labels_new"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    change_txt_format(intput_path,output_path)