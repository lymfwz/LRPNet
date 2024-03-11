import os
import cv2

# 前置处理
def ori_reshape(input_folder, output_folder):
    # 打开文件夹遍历图片process/ori
    files = os.listdir(input_folder)
    for file in files:
        # 读取图片
        img = cv2.imread(input_folder + file)
        # 将图片的大小重新定义为(x, x)
        width, height = img.shape[1], img.shape[0]
        x = max(width, height)
        if x > 2816:
            img = cv2.resize(img, (2816, 2816))
        else:
            x = x // 256 * 256
            img = cv2.resize(img, (x, x))
        # 将图片保存为new_1.png
        cv2.imwrite(output_folder + file, img)
        print(file)

# 后置处理
def img_reshape_to_ori(folderOri, folderRes, outputFolder):
    # 打开文件夹遍历图片 process/ori
    files_ori = os.listdir(folderOri)
    files_re = os.listdir(folderRes)

    for file_re in files_re:
        # 检查文件是否存在于 process_re 文件夹中
        if file_re in files_ori:
            # 读取原始图片
            img_ori = cv2.imread(os.path.join(folderOri, file_re))
            # 获取原始图片的大小
            height_ori, width_ori, _ = img_ori.shape

            # 读取 process_re 文件夹中的对应图片
            img_re = cv2.imread(os.path.join(folderRes, file_re))
            # 调整 process_re 中的图片大小为原始图片的大小
            img_re_resized = cv2.resize(img_re, (width_ori, height_ori))
            cv2.imwrite(os.path.join(outputFolder, file_re), img_re_resized)
            print(f"Resized and saved: {file_re}")
        else:
            print(f"No corresponding image found for: {file_re}")


def image_process():
    from PIL import Image
    import os
    from tqdm import tqdm
    # 文件夹路径
    folder1 = r'E:\projects\python\Machine-Learning-Collection\ML\Pytorch\GANs\Pix2Pix\data\myAnime\train'
    folder2 = r'E:\projects\python\Machine-Learning-Collection\ML\Pytorch\GANs\Pix2Pix\data\myAnime\train_line\anime_style'
    folder3 = r'E:\projects\python\Machine-Learning-Collection\ML\Pytorch\GANs\Pix2Pix\data\myAnime\myTrain'

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder3):
        os.makedirs(folder3)

    # 获取文件夹1和2中的图像文件列表
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # 遍历文件夹1中的图像文件
    for file1 in tqdm(files1, desc="Processing images"):
        # 检查是否存在相应的文件夹2中的图像文件
        corresponding_file2 = file1  # 假设文件名一致
        if corresponding_file2 in files2:
            # 打开文件夹1中的图像
            image1_path = os.path.join(folder1, file1)
            img1 = Image.open(image1_path)

            # 打开文件夹2中的图像
            image2_path = os.path.join(folder2, corresponding_file2)
            img2 = Image.open(image2_path)

            # 确保图像尺寸相同（512x512）
            if img1.size == img2.size == (512, 512):
                # 创建新图像，水平合并
                new_img = Image.new('RGB', (1024, 512))
                new_img.paste(img1, (0, 0))
                new_img.paste(img2, (512, 0))

                # 保存合并后的图像到文件夹3
                output_path = os.path.join(folder3, f"{file1}")
                new_img.save(output_path)
            else:
                print(f"图像尺寸不匹配: {file1}")

    print("合并完成！")


if __name__ == '__main__':
    # ori_reshape()
    img_reshape_to_ori("process/imgr/ori",
                       "process/imgr/process_re",
                       "process/imgr/process_reshape")

    # files = os.listdir("process/ori")
    # for file in files:
    #     # 读取图片
    #     img = cv2.imread('process/ori/' + file)
    #     # 将图片的大小重新定义为(x, x)
    #     img = cv2.resize(img, (3072, 3072))
    #     cv2.imwrite('process/ori_reshape/' + file, img)
    #     print(file)
    
    # image_process()

    # drawImg()
    print("Done")