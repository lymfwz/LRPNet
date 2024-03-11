from PIL import Image
import os
from tqdm import tqdm
import concurrent.futures

def process_image(file1, folder1, folder2, folder3):
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

def image_process():
    # 文件夹路径
    folder1 = r'E:\projects\python\Machine-Learning-Collection\ML\Pytorch\GANs\Pix2Pix\data\myAnime\train'
    folder2 = r'E:\projects\python\Machine-Learning-Collection\ML\Pytorch\GANs\Pix2Pix\data\myAnime\train_line\anime_style'
    folder3 = r'E:\projects\python\Machine-Learning-Collection\ML\Pytorch\GANs\Pix2Pix\data\myAnime\myTrain2'

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder3):
        os.makedirs(folder3)

    # 获取文件夹1和2中的图像文件列表
    files1 = os.listdir(folder1)
    global files2  # 将files2变成全局变量，以便在process_image函数中使用
    files2 = os.listdir(folder2)

    # 使用多线程处理图像
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file1 in files1:
            futures.append(executor.submit(process_image, file1, folder1, folder2, folder3))

        # 等待所有线程完成
        concurrent.futures.wait(futures)

    print("合并完成！")

# 调用函数进行图像处理
# image_process()
import networkx as nx
import matplotlib.pyplot as plt

def draw_patch_gan():
    # 创建一个空的有向图
    G = nx.DiGraph()

    # 添加节点
    G.add_node('input', pos=(0, 0), color='blue')
    G.add_node('patch_0', pos=(1, 1), color='green')
    G.add_node('patch_1', pos=(1, 0), color='green')
    G.add_node('patch_2', pos=(1, -1), color='green')
    G.add_node('patch_3', pos=(1, -2), color='green')
    G.add_node('output', pos=(2, -1), color='red')

    # 添加边
    G.add_edge('input', 'patch_0')
    G.add_edge('input', 'patch_1')
    G.add_edge('input', 'patch_2')
    G.add_edge('input', 'patch_3')
    G.add_edge('patch_0', 'output')
    G.add_edge('patch_1', 'output')
    G.add_edge('patch_2', 'output')
    G.add_edge('patch_3', 'output')

    # 提取节点颜色
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]

    # 提取节点位置
    node_positions = nx.get_node_attributes(G, 'pos')

    # 绘制图形
    nx.draw(G, pos=node_positions, node_color=node_colors, with_labels=True, node_size=1000)

    # 显示图形
    plt.show()



if __name__ == '__main__':
    draw_patch_gan()
