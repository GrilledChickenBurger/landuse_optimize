# I/O 工具：文件系统操作、TIFF 保存、图片序列转视频
# 合并自 file_utils.py 和 video_utils.py

import os
from typing import List, Optional


def ensure_folder(folder: str) -> None:
    """确保文件夹存在，若不存在则创建"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def find_valid_file(file_paths: List[str], file_description: str = "文件") -> Optional[str]:
    """
    从文件路径列表中查找第一个存在的有效文件路径。

    Parameters:
    -----------
    - file_paths: 文件路径列表，按优先级排序
    - file_description: 文件描述，用于日志输出

    Returns:
    --------
    - 第一个存在的文件路径，如果都不存在则返回 None
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            return file_path
    print(f"警告: 找不到{file_description}文件，尝试的路径有: {', '.join(file_paths)}")
    return None


def save_tif(plan, output_folder: str, output_filename: str):
    """
    保存张量为 TIFF 文件。

    Parameters:
    ----------
    - plan: 编辑后的张量，需转换为 numpy 数组
    - output_folder: 保存文件的文件夹位置
    - output_filename: 输出文件名
    """
    import rasterio
    from rasterio.transform import from_origin
    from .tools import convert_to_nparray

    ensure_folder(output_folder)
    output_path = os.path.join(output_folder, output_filename)

    plan_np = convert_to_nparray(plan)
    _, _, height, width = plan_np.shape

    transform = from_origin(0, 0, 1, 1)

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=plan_np.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(plan_np[0, 0, :, :], 1)


def get_video_from_imgs(src_folder: str, video_folder: str):
    """
    从 src_folder 中读取所有图片，生成视频并保存到 video_folder。
    视频文件名格式为: output_video.mp4

    Parameters:
    ----------
    - src_folder: 图片源文件夹路径
    - video_folder: 视频输出文件夹路径
    """
    import cv2

    print("================= 生成视频中... =================")

    allvalimg_path = [os.path.join(src_folder, filename) for filename in os.listdir(src_folder)]
    frame_num = len(allvalimg_path)
    print(f"共{frame_num}帧图片, 生成视频中...")

    def epoch_sort_key(file_path):
        """Extract the step number from the file path."""
        parts = file_path.split('/')
        filename_parts = parts[-1].replace('.png', '').split('_')
        step_num = int(filename_parts[-1])
        return step_num

    img_path = sorted([f for f in allvalimg_path], key=epoch_sort_key)
    assert len(img_path) == frame_num

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    frame0 = cv2.imread(img_path[0])
    height, width, layers = frame0.shape

    fps = frame_num // 10
    ensure_folder(video_folder)
    video_name = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{video_folder}/{video_name}', fourcc, fps, (width, height))

    for img_file in img_path:
        frame = cv2.imread(img_file)
        video.write(frame)
        del frame

    video.release()
    print(f"Done. The video has been saved to {video_folder}.\n")
