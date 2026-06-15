import os
from tqdm import tqdm

# 根据文件夹名创建标签，框为整个图像
def generate_labels_through_dirname(input_path,output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 列文件
    list_imgs = os.listdir(input_path)

    # dir name
    input_path = os.path.normpath(input_path)
    dir_name = os.path.basename(input_path)

    for img in list_imgs:
        img_name,_ = os.path.splitext(img)
        txt_name = img_name + ".txt"
        txt_path = os.path.join(output_path,txt_name)

        # open同名txt
        with open(txt_path,'x') as txt:
            # 文件夹名写入类别，左上右下点
            line = f"{dir_name} 0.0 0.0 1.0 1.0"
            txt.write(line)
            print(f"Write {txt_name}.")

# 根据文件名创建标签，框为整个图像
def generate_labels_through_filename(input_path,output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 列文件
    list_imgs = os.listdir(input_path)
    list_imgs = [x for x in list_imgs if os.path.isfile(os.path.join(input_path,x))]

    for img in list_imgs:
        img_name,_ = os.path.splitext(img)
        txt_name = img_name + ".txt"
        txt_path = os.path.join(output_path,txt_name)

        # 获取类别
        label_idx = img.split("_")[1]

        # open同名txt
        with open(txt_path,'x') as txt:
            # 文件夹名写入类别，左上右下点
            line = f"{int(label_idx)-1} 0.5 0.5 1.0 1.0"
            txt.write(line)
            print(f"Write {txt_name}.")

if __name__ == "__main__":
    # root_path = r"D:\A_myData\dataset\juanZhou_gazebo"
    # output_root_path = r"D:\A_myData\dataset\juanZhou_gazebo\labels"
    #
    # if not os.path.exists(output_root_path):
    #     os.mkdir(output_root_path)
    #
    # list_dir = os.listdir(root_path)
    # list_dir = [x for x in list_dir if os.path.isdir(os.path.join(root_path,x))]
    # for dir in tqdm(list_dir):
    #     dir_path = os.path.join(root_path,dir)
    #     output_dir = os.path.join(output_root_path,dir)
    #     generate_labels_through_dirname(dir_path,output_dir)

    root_path = "/home/awwsome/datasets/juanZhou_gazebo10/"

    dir_list = os.listdir(root_path)

    for dir_name in dir_list:
        input_path = os.path.join(root_path,dir_name)
        output_path = r"/home/awwsome/datasets/juanZhou_gazebo11/labels"

        os.makedirs(output_path,exist_ok=True)

        generate_labels_through_dirname(input_path,output_path)