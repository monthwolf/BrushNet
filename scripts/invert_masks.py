import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def invert_mask(image_path, output_path):
    """
    读取图像，进行反转处理（0->255, 255->0），并保存。
    """
    try:
        with Image.open(image_path) as img:
            # 如果不是灰度图，转换为灰度图 (L模式)
            if img.mode != 'L':
                img = img.convert('L')
            
            # 转为 numpy 数组
            img_np = np.array(img)
            
            # 反转颜色：255 - 原像素值
            inverted_np = 255 - img_np
            
            # 保存结果
            inverted_img = Image.fromarray(inverted_np)
            inverted_img.save(output_path)
            
    except Exception as e:
        print(f"处理文件 {image_path} 时发生错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="遮罩图像黑白反转工具 (Mask Inverter)")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入文件路径或文件夹路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出文件路径或保存目录")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # 判断输入是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入路径 '{input_path}' 不存在。")
        return

    # 情况1: 输入是文件夹
    if os.path.isdir(input_path):
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"创建输出目录: {output_path}")
        
        # 获取支持的图像文件列表
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_exts)]
        
        if not files:
            print(f"在 '{input_path}' 中未找到支持的图像文件。")
            return
            
        print(f"开始批量处理 {len(files)} 张图像...")
        print(f"输入目录: {input_path}")
        print(f"输出目录: {output_path}")
        
        for filename in tqdm(files, desc="Processing"):
            in_file = os.path.join(input_path, filename)
            out_file = os.path.join(output_path, filename)
            invert_mask(in_file, out_file)
            
        print("所有图像处理完成。")

    # 情况2: 输入是单个文件
    elif os.path.isfile(input_path):
        # 确定输出路径
        # 如果 output_path 看起来像目录（没有扩展名，或者以分隔符结尾，或者是已存在的目录），则将文件保存其中
        is_output_dir = os.path.isdir(output_path) or output_path.endswith(os.sep) or (os.path.basename(output_path) == '' or '.' not in os.path.basename(output_path))
        
        if is_output_dir:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # 使用原文件名
            filename = os.path.basename(input_path)
            final_output_path = os.path.join(output_path, filename)
        else:
            # output_path 是文件路径
            final_output_path = output_path
            # 如果父目录不存在则创建
            parent_dir = os.path.dirname(os.path.abspath(final_output_path))
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

        print(f"处理单张图像: {input_path} -> {final_output_path}")
        invert_mask(input_path, final_output_path)
        print("完成。")

if __name__ == "__main__":
    main()
