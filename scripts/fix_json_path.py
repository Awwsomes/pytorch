import os
import json

def batch_fix_image_path(json_dir, image_dir):
    print("=" * 60)
    print("Labelme 批量修改 imagePath 为相对路径工具")
    print("=" * 60)

    # 验证目录合法性
    if not os.path.isdir(json_dir):
        print(f"❌ 错误：JSON 文件夹不存在 - {json_dir}")
        return
    if not os.path.isdir(image_dir):
        print(f"❌ 错误：图片文件夹不存在 - {image_dir}")
        return

    # 统一转为绝对路径，避免相对路径计算偏差
    json_dir = os.path.abspath(json_dir)
    image_dir = os.path.abspath(image_dir)

    print(f"JSON 文件夹: {json_dir}")
    print(f"图片文件夹: {image_dir}")
    print("-" * 60)

    # 统计信息
    total_files = 0
    success_files = 0
    skipped_files = 0

    # 遍历所有 JSON 文件
    for filename in os.listdir(json_dir):
        if filename.lower().endswith('.json'):
            total_files += 1
            json_file_path = os.path.join(json_dir, filename)

            try:
                # 读取原始 JSON
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取无后缀的文件名
                base_name = os.path.splitext(filename)[0]

                # 构建图片绝对路径（默认png格式，需其他后缀可自行修改）
                image_abs_path = os.path.join(image_dir, f"{base_name}.png")

                # 计算图片相对于 JSON 目录的相对路径
                relative_path = os.path.relpath(image_abs_path, start=json_dir)
                # 统一转为正斜杠，兼容 Windows/Linux 和 Labelme 读取
                relative_path = relative_path.replace(os.sep, '/')

                old_path = data.get('imagePath', '未找到')
                data['imagePath'] = relative_path

                # 覆盖保存
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"✅ 成功: {filename}")
                print(f"   旧路径: {old_path}")
                print(f"   新路径: {relative_path}")
                success_files += 1

            except Exception as e:
                print(f"❌ 失败: {filename} - 错误: {str(e)}")
                skipped_files += 1

    # 输出统计
    print("-" * 60)
    print(f"处理完成！总计: {total_files} 个文件")
    print(f"成功: {success_files} 个")
    print(f"失败: {skipped_files} 个")
    print("=" * 60)


if __name__ == "__main__":
    # ====================== 配置路径 ======================
    json_dir = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det1\jsons_fix"  # JSON 所在文件夹
    image_dir = r"D:\A_myData\RC26-Vision\dataset\juanZhou_det_blue2\image"  # 图片所在文件夹
    # =====================================================
    batch_fix_image_path(json_dir, image_dir)