import rename_data
import os
import shutil
import json

def rename_data_and_generate_jsons(root_dir:str,output_root_path:str,start_idx:int):
    """
    复制并重命名ext后缀名列表中的文件，会处理根目录底下所有子目录的文件。
    输出文件全部拷贝到输出目录下（一个文件夹）
    文件夹格式
    ----dataset
        ----images
   	        ----image_1.png
      	    ----image_2.png
      	    ...
    ----roi_images
        ----roi_1
            ----1.png
       	    .
       	    .
       	    ----12.png
	    ----roi_2
    ----labels
   	    ----label_1.json
   	    ----label_2.json
   	    ...
    label格式
	{
		"rvec": [],
		"tvec": [],
		"labels": [],
		"point_size": []
	}
    """
    print("拷贝并重命名文件...")
    os.makedirs(os.path.join(output_root_path,"images"), exist_ok=True)
    os.makedirs(os.path.join(output_root_path,"labels"), exist_ok=True)

    idx = start_idx
    for root,_,files in os.walk(root_dir):
        # print(len(files))
        if len(files) == 0:
            continue
        else:
            global_imgs_list = [x for x in files if os.path.splitext(os.path.join(root_dir,x))[1] in [".png", ".jpg", ".jpeg", ".bmp"]]
            txts_list = [x for x in files if os.path.splitext(os.path.join(root_dir,x))[1] == ".txt"]
            print(len(global_imgs_list), len(txts_list))
            if len(global_imgs_list) == 0 or len(txts_list) == 0:
                continue

            # 只处理第一个文件，因为按照格式只会有一个文件
            # 拷贝图片
            old_img_path = os.path.join(root,global_imgs_list[0])
            _,img_ext = os.path.splitext(global_imgs_list[0])
            img_new_name = f"image_{idx}{img_ext}"
            new_img_path = os.path.join(output_root_path, "images", img_new_name)
            # print(new_img_path)
            shutil.copy(old_img_path,new_img_path)

            # 读取txt，生成json
            # 拼接路径
            old_txt_path = os.path.join(root, txts_list[0])
            new_json_path = os.path.join(output_root_path, "labels", f"label_{idx}.json")
            # print(new_json_path)

            # 读取txt，生成json
            with open(old_txt_path, 'r') as txt_file, \
                 open(new_json_path, 'w') as json_file:

                # 读取txt内容
                lines = txt_file.readlines()
                if not len(lines) == 3:
                    print(f"[ERROR]")
                rvec = [float(x.strip()) for x in lines[0].strip().split(",")]
                tvec = [float(x.strip()) for x in lines[1].strip().split(",")]
                labels = [int(x) for x in lines[2].strip().split()]

                # 生成字典，写入json文件
                json_content = {
                    "rvec": rvec,
                    "tvec": tvec,
                    "labels": labels
                }
                json.dump(json_content, json_file)

            idx += 1
    print("拷贝并重命名文件完成.")

if __name__ == "__main__":
    input_dir = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_26"
    output_root_dir = r"D:\A_myData\RC26-Vision\dataset\A_car\2026_3_26_output"
    start_idx = 1

    rename_data_and_generate_jsons(input_dir, output_root_dir, start_idx)