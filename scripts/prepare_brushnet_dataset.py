
import os
import argparse
import json
import random
import numpy as np
from PIL import Image, ImageDraw
import tqdm
import shutil
import warnings

# 尝试导入 transformers 和 torch，如果不可用则稍后处理
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="将图像文件夹转换为 BrushBench 数据集格式 (images 文件夹 + mapping_file.json)。")
    parser.add_argument("--input_dir", type=str, required=True, help="包含源图像的文件夹路径。")
    parser.add_argument("--output_dir", type=str, required=True, help="保存输出数据集的根目录。")
    parser.add_argument("--resolution", type=int, default=512, help="调整大小的目标分辨率（图像将被调整大小以适应）。")
    parser.add_argument("--caption_model", type=str, default="Salesforce/blip-image-captioning-base", help="指定用于生成描述的 AI 模型 (例如 'Salesforce/blip-image-captioning-base')。如果未指定，将使用默认的虚拟描述。")
    parser.add_argument("--model_dir", type=str, default="data/model", help="模型下载/加载的本地目录。")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备 (cuda 或 cpu)。")
    return parser.parse_args()

def mask2rle(img):
    '''
    img: numpy format binary mask image, shape (h, w)
    order: 'F' (Fortran-style)
    Returns RLE encoding as a list of integers [start, length, start, length...]
    '''
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()

def generate_random_mask(height, width, mask_type='box'):
    """生成随机掩码 (0=保留, 1=遮罩/修复区域)"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if mask_type == 'box':
        # 随机矩形遮罩 (20% - 50% 面积)
        box_w = random.randint(width // 4, width // 2)
        box_h = random.randint(height // 4, height // 2)
        x1 = random.randint(0, width - box_w)
        y1 = random.randint(0, height - box_h)
        draw.rectangle((x1, y1, x1 + box_w, y1 + box_h), fill=1)
    else:
        # 随机线条
        for _ in range(5):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            w_line = random.randint(width // 20, width // 10)
            draw.line((x1, y1, x2, y2), fill=1, width=w_line)
            
    return np.array(mask).astype(np.uint8)

def process_image(image_path, resolution):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # 简单的调整大小
            img = img.resize((resolution, resolution), Image.BICUBIC)
            return img
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return None

class CaptionGenerator:
    def __init__(self, model_name, device, model_dir):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装 `transformers` 和 `torch` 以使用 AI 字幕生成功能: `pip install transformers torch`")
        
        # 确定本地保存路径
        if os.path.exists(model_name):
             local_path = model_name
        else:
             repo_name = model_name.split('/')[-1]
             local_path = os.path.join(model_dir, repo_name)
        
        if not os.path.exists(local_path):
            print(f"本地未找到模型，正在下载 {model_name} 到 {local_path} ...")
            try:
                os.makedirs(local_path, exist_ok=True)
                processor = BlipProcessor.from_pretrained(model_name)
                model = BlipForConditionalGeneration.from_pretrained(model_name)
                processor.save_pretrained(local_path)
                model.save_pretrained(local_path)
                print("模型下载并保存完毕。")
            except Exception as e:
                 # Clean up empty dir if failed
                 if os.path.exists(local_path) and not os.listdir(local_path):
                     os.rmdir(local_path)
                 raise RuntimeError(f"下载模型失败: {e}")
        else:
            print(f"发现本地模型: {local_path}")

        print(f"正在加载模型 到 {device}...")
        self.device = device
        self.processor = BlipProcessor.from_pretrained(local_path)
        self.model = BlipForConditionalGeneration.from_pretrained(local_path).to(device)
        print("模型加载完成。")

    def generate(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

def main():
    args = parse_args()

    # 如果指定了模型，初始化生成器
    caption_generator = None
    if args.caption_model:
        try:
            caption_generator = CaptionGenerator(args.caption_model, args.device, args.model_dir)
        except Exception as e:
            print(f"警告: 无法加载字幕模型 ({e})。将回退到默认虚拟字幕。")

    images_dir = os.path.join(args.output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    masks_dir = os.path.join(args.output_dir, "masks")
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"找到 {len(image_files)} 张图像。")
    image_files.sort()

    mapping_data = {}

    print("开始处理...")
    for idx, filename in enumerate(tqdm.tqdm(image_files)):
        file_path = os.path.join(args.input_dir, filename)
        img = process_image(file_path, args.resolution)
        
        if img is None:
            continue

        # 生成新的文件名 (ID.jpg)
        image_id = f"{idx:09d}" # Matches 000000000 format
        new_filename = f"{image_id}.jpg"
        save_path = os.path.join(images_dir, new_filename)
        
        img.save(save_path, quality=95)

        # 生成随机掩码
        current_mask_np = generate_random_mask(args.resolution, args.resolution, mask_type='box' if random.random() > 0.5 else 'random')
        
        # 保存遮罩图像
        mask_save_path = os.path.join(masks_dir, f"{image_id}.png")
        Image.fromarray((1 - current_mask_np) * 255).save(mask_save_path)

        inpainting_rle = mask2rle(current_mask_np)
        
        outpainting_mask_np = generate_random_mask(args.resolution, args.resolution, mask_type='box')
        outpainting_rle = mask2rle(outpainting_mask_np)

        # 生成描述 (Caption)
        caption = f"an image of {os.path.splitext(filename)[0]}"
        if caption_generator:
            try:
                caption = caption_generator.generate(img)
            except Exception as e:
                print(f"生成字幕失败: {e}，使用默认字幕。")

        mapping_data[image_id] = {
            "image": f"images/{new_filename}",
            "caption": caption, 
            "inpainting_mask": inpainting_rle,
            "outpainting_mask": outpainting_rle
        }

    # 保存 mapping_file.json
    json_path = os.path.join(args.output_dir, "mapping_file.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"完成！数据集已保存到 {args.output_dir}")
    print(f"包含 mapping_file.json 和 images 文件夹。")

if __name__ == "__main__":
    main()
