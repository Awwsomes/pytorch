import os
import random
import shutil
from tqdm import tqdm

def random_take_some_files(input_path:str, output_path:str, n:int, mode:str):
    orin_file_list = os.listdir(input_path)
    orin_file_list = [file for file in orin_file_list if os.path.isfile(os.path.join(input_path, file))]

    if len(orin_file_list) < n:
        print("Not enough orin files,skip the dir...")
        return -1
    file_list = random.sample(orin_file_list, n)

    if mode == "copy":
        os.makedirs(output_path, exist_ok=True)
        for file in file_list:
            input_file = os.path.join(input_path, file)
            output_file = os.path.join(output_path, file)
            shutil.copy(input_file, output_file)
    elif mode == "delete":
        others_file_list = [file for file in orin_file_list if file not in file_list]
        for file in others_file_list:
            other_files_path = os.path.join(input_path, file)
            os.remove(other_files_path)

    return 0


if __name__ == "__main__":
    process_list = ["1"]
    root_dir = r"D:\A_myData\RC26-Vision\dataset\juanZhou_cls_gazebo_real_rotate1"
    output_root_dir = r"D:\A_myData\RC26-Vision\dataset\juanZhou_cls_mix5"
    for process_dir in tqdm(process_list):
        dir_path = os.path.join(root_dir, process_dir)
        output_dir = os.path.join(output_root_dir, process_dir)
        need_files_num = 2850-1452

        random_take_some_files(dir_path, output_dir, need_files_num,"copy")