import os
import shutil

def copy_dirs(root_dir:str,output_dir:str):
    """
    复制根文件夹的所有层级的子目录到一个新的目录
    """
    print("拷贝目录...")
    if not os.path.exists(root_dir):
        print("根目录不存在！")
        return -1
    for root,dirs,_ in os.walk(root_dir):
        if len(dirs) == 0:
            rela_path = os.path.relpath(root,root_dir)
            new_path = os.path.join(output_dir,rela_path)
            os.makedirs(new_path,exist_ok=True)
    print("目录拷贝完成.")
    return 0

def rename_data(root_dir:str,output_path:str,img_root_name:str,start_idx:int,list_ext:list):
    """
    复制并重命名ext后缀名列表中的文件，会处理根目录底下所有子目录的文件。
    输出文件会按照原文件夹的层级储存
    注意：不会同步拷贝标签
    """
    print("拷贝并重命名文件...")
    idx = start_idx
    for root,_,files in os.walk(root_dir):
        files = [x for x in files if os.path.splitext(x)[1] in list_ext]
        if len(files) == 0:
            continue
        else:
            rela_path = os.path.relpath(root,root_dir)
            new_dir = os.path.join(output_path,rela_path)
            for file in files:
                old_file_path = os.path.join(root,file)
                _,file_ext = os.path.splitext(file)
                file_new_name = f"{img_root_name}{idx}{file_ext}"
                new_file_path = os.path.join(new_dir,file_new_name)
                shutil.copy(old_file_path,new_file_path)
                idx += 1
    print("拷贝并重命名文件完成.")

def rename_data_one_dir(root_dir:str,output_path:str,img_root_name:str,start_idx:int,list_ext:list):
    """
    复制并重命名ext后缀名列表中的文件，会处理根目录底下所有子目录的文件。
    输出文件全部拷贝到输出目录下（一个文件夹）
    注意：不会同步拷贝标签
    """
    print("拷贝并重命名文件...")
    idx = start_idx
    for root,_,files in os.walk(root_dir):
        files = [x for x in files if os.path.splitext(x)[1] in list_ext]
        if len(files) == 0:
            continue
        else:
            for file in files:
                old_file_path = os.path.join(root,file)
                _,file_ext = os.path.splitext(file)
                file_new_name = f"{img_root_name}{idx}{file_ext}"
                new_file_path = os.path.join(output_path,file_new_name)
                shutil.copy(old_file_path,new_file_path)
                idx += 1
    print("拷贝并重命名文件完成.")

if __name__ == "__main__":
    root_path = r"D:\A_myData\dataset\juanZhou_log\2026_1_9"
    output_path = r"D:\A_myData\dataset\juanZhou_log\2026_1_9_new"
    img_root_name = "juanZhou_multi_color"
    start_idx = 0
    ext = [".png",".jpg",".mat"]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # copy_dirs(root_path,output_path)
    rename_data_one_dir(root_path,output_path,img_root_name,start_idx,ext)