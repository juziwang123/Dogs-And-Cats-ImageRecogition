import os
import shutil

# 定义源文件夹和目标文件夹
source_directory = r'C:\Users\chiju\PycharmProjects\RecogitionImage\train\train'
destination_directory = r'C:\Users\chiju\PycharmProjects\RecogitionImage\train\train_organized'


def organize_images():
    """
    将猫和狗的图片从源文件夹移动到按类别分类的目标文件夹中。
    """

    # 确保目标文件夹存在
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
        print(f"创建了目标目录: {destination_directory}")

    # 定义猫和狗的子文件夹路径
    cats_dir = os.path.join(destination_directory, 'cats')
    dogs_dir = os.path.join(destination_directory, 'dogs')

    # 确保子文件夹存在
    if not os.path.exists(cats_dir):
        os.makedirs(cats_dir)
        print(f"创建了猫的目录: {cats_dir}")
    if not os.path.exists(dogs_dir):
        os.makedirs(dogs_dir)
        print(f"创建了狗的目录: {dogs_dir}")

    # 遍历源文件夹中的所有文件
    print("开始移动图片...")
    for filename in os.listdir(source_directory):
        # 构建完整的源文件路径
        source_path = os.path.join(source_directory, filename)

        # 检查文件是否为图片（跳过文件夹等）
        if os.path.isfile(source_path) and (filename.endswith('.jpg') or filename.endswith('.png')):
            # 根据文件名判断类别并移动文件
            if filename.startswith('cat.'):
                destination_path = os.path.join(cats_dir, filename)
                shutil.move(source_path, destination_path)
            elif filename.startswith('dog.'):
                destination_path = os.path.join(dogs_dir, filename)
                shutil.move(source_path, destination_path)

    print("图片移动完成。")
    print(f"所有图片现在都已组织在 {destination_directory} 目录下的 'cats' 和 'dogs' 文件夹中。")


if __name__ == "__main__":
    organize_images()