import os
import cv2
import shutil

'''
    @brief 直接通过展示图片来选择图片，并把图片，标签，mat同步转存在输出文件夹
           输入文件夹目录要求：---root
                            |---images  存图片
                            |---image   存mat
                            |---labels  存txt
'''
def choose_and_copy(input_path,output_path):

    output_img = os.path.join(output_path,"images")
    output_txt = os.path.join(output_path,"labels")
    output_mat = os.path.join(output_path, "image")
    os.makedirs(output_img,exist_ok=True)
    os.makedirs(output_txt,exist_ok=True)
    os.makedirs(output_mat,exist_ok=True)

    imgs_path = os.path.join(input_path,"images")
    txt_path = os.path.join(input_path,"labels")
    mat_path = os.path.join(input_path,"image")

    # 列出并筛选图片
    list_img = os.listdir(imgs_path)
    list_img = [x for x in list_img if os.path.isfile(os.path.join(imgs_path,x))]

    # 展示图片
    for img_file in list_img:
        img_path = os.path.join(imgs_path,img_file)
        img = cv2.imread(img_path)
        cv2.imshow("img",img)
        key_code = cv2.waitKey(0)
        print(key_code)
        # y 要 n 不要
        if key_code == 121 or key_code == 89: # Y or y
            # copy图片
            new_img_path = os.path.join(output_img,img_file)
            shutil.copy(img_path,new_img_path)

            # copy txt
            img_name,_ = os.path.splitext(img_file)
            txt = img_name + ".txt"
            old_txt_path = os.path.join(txt_path,txt)
            new_txt_path = os.path.join(output_txt,txt)
            shutil.copy(old_txt_path,new_txt_path)

            # copy mat
            mat = img_name + ".mat"
            old_mat_path = os.path.join(mat_path,mat)
            new_mat_path = os.path.join(output_mat,mat)
            shutil.copy(old_mat_path,new_mat_path)

            print(f"Copy {img_file} and its txt & mat.")
        elif key_code == 110 or key_code == 78:  # n or N
            print(f"No choose {img_file}.")
            continue
        elif key_code == 113 or key_code == 81:  # q or Q
            exit(0)

if __name__ == "__main__":
    input_path = r"D:\A_myData\dataset\hands2"
    output_path = r"D:\A_myData\dataset\hand5_only_heng_ping"
    choose_and_copy(input_path,output_path)