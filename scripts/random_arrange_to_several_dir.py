import os
import shutil
import random

def arrange_images(input_path,output_root_path):
    list_dir = os.listdir(output_root_path)
    amount_dir = len(list_dir)
    if amount_dir == 0:

    save_dir_idx = 0

    extension_img = [".jpg",".png",".bmp"]
    list_img = os.listdir(input_path)
    list_img = [x for x in list_img if os.path.isfile(os.path.join(input_path,x))]
    list_img = [x for x in list_img if os.path.splitext(x)[1] in extension_img]

    random.shuffle(list_img)
    amount_img = len(list_img)
    amount_each_dir = int(amount_img / amount_dir)

    for idx,img in enumerate(list_img):
        origin_path = os.path.join(input_path,img)
        save_path = os.path.join(os.path.join(output_root_path,list_dir[save_dir_idx]),img)
        shutil.copy(origin_path,save_path)
        print(f"{save_dir_idx + 1}/{amount_dir} copy to {save_path}")
        if (idx + 1) % amount_each_dir == 0:
            save_dir_idx += 1

if __name__ == "__main__":
    input_dir = r"D:\A_myData\dataset\corner3\images"
    output_root_dir = r"D:\A_myData\dataset\corner3\split"
    arrange_images(input_dir,output_root_dir)