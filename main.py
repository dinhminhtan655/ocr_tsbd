# import os
# from typing import List
# import numpy as np
# import torch
# import torchvision.transforms as T
# from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModel, AutoTokenizer
# import io
# import json

# # --- Thư viện cho API ---
# # THAY ĐỔI: Import thêm 'Form' để nhận dữ liệu từ form
# from fastapi import FastAPI, File, HTTPException, UploadFile, Form
# import uvicorn

# # =================================================================================
# # PHẦN 1: CÁC HÀM XỬ LÝ ẢNH VÀ MODEL (Không thay đổi)
# # =================================================================================

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# def build_transform(input_size):
#     MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
#     transform = T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=MEAN, std=STD)
#     ])
#     return transform

# # ... (Các hàm xử lý ảnh khác giữ nguyên như cũ) ...
# def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
#     best_ratio_diff = float('inf')
#     best_ratio = (1, 1)
#     area = width * height
#     for ratio in target_ratios:
#         target_aspect_ratio = ratio[0] / ratio[1]
#         ratio_diff = abs(aspect_ratio - target_aspect_ratio)
#         if ratio_diff < best_ratio_diff:
#             best_ratio_diff = ratio_diff
#             best_ratio = ratio
#         elif ratio_diff == best_ratio_diff:
#             if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
#                 best_ratio = ratio
#     return best_ratio

# def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
#     orig_width, orig_height = image.size
#     aspect_ratio = orig_width / orig_height
#     target_ratios = set(
#         (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
#         i * j <= max_num and i * j >= min_num)
#     target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
#     target_aspect_ratio = find_closest_aspect_ratio(
#         aspect_ratio, target_ratios, orig_width, orig_height, image_size)
#     target_width = image_size * target_aspect_ratio[0]
#     target_height = image_size * target_aspect_ratio[1]
#     blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
#     resized_img = image.resize((target_width, target_height))
#     processed_images = []
#     for i in range(blocks):
#         box = (
#             (i % (target_width // image_size)) * image_size,
#             (i // (target_width // image_size)) * image_size,
#             ((i % (target_width // image_size)) + 1) * image_size,
#             ((i // (target_width // image_size)) + 1) * image_size
#         )
#         split_img = resized_img.crop(box)
#         processed_images.append(split_img)
#     assert len(processed_images) == blocks
#     if use_thumbnail and len(processed_images) != 1:
#         thumbnail_img = image.resize((image_size, image_size))
#         processed_images.append(thumbnail_img)
#     return processed_images

# def load_image(image_file, input_size=448, max_num=12):
#     image = image_file.convert('RGB')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values


# # =================================================================================
# # PHẦN 2: TẢI MODEL VÀ ĐỊNH NGHĨA API
# # =================================================================================

# app = FastAPI(title="Sổ Hồng OCR API", description="API để trích xuất thông tin từ ảnh sổ hồng.")

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Sử dụng GPU (MPS) của Apple Silicon.")
# else:
#     device = torch.device("cpu")
#     print("Không tìm thấy MPS, sử dụng CPU. Quá trình xử lý có thể chậm.")

# model_name = "5CD-AI/Vintern-1B-v3_5"
# print("Đang tải model...")
# try:
#   model = AutoModel.from_pretrained(
#       model_name,
#       torch_dtype=torch.bfloat16,
#       low_cpu_mem_usage=True,
#       trust_remote_code=True,
#       use_flash_attn=False,
#   ).eval().to(device)
# except Exception:
#   model = AutoModel.from_pretrained(
#       model_name,
#       torch_dtype=torch.bfloat16,
#       low_cpu_mem_usage=True,
#       trust_remote_code=True
#   ).eval().to(device)

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
# print("Model đã tải xong.")

# # --- Định nghĩa API Endpoint ---
# # THAY ĐỔI QUAN TRỌNG: Thêm `prompt: str = Form(...)` để nhận cả prompt từ client
# @app.post("/extract-land-title-info/")
# async def extract_info(files: List[UploadFile] = File(...), prompt: str = Form(...)):
#     """
#     Nhận một DANH SÁCH file ảnh và một chuỗi prompt, xử lý và trả về thông tin.
#     """
#     if not files:
#         raise HTTPException(status_code=400, detail="No files sent.")

#     print(f"Nhận được {len(files)} file(s).")
    
#     list_of_pixel_values = []
#     # Lặp qua từng file ảnh được gửi lên
#     for file in files:
#         print(f"Đang xử lý file: {file.filename}")
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents))

#         # Sử dụng lại hàm load_image để xử lý từng ảnh một
#         # Hàm này sẽ trả về một tensor các "mảnh vá" (patches) cho mỗi ảnh
#         pixel_values_single_image = load_image(image, max_num=6)
#         list_of_pixel_values.append(pixel_values_single_image)

#     # Nối tất cả các tensor từ các ảnh lại thành một tensor lớn duy nhất
#     # Đây là bước quan trọng để model "nhìn" thấy tất cả các ảnh cùng lúc
#     combined_pixel_values = torch.cat(list_of_pixel_values, dim=0).to(torch.bfloat16).to(device)
    
#     print(f"Đã gộp các ảnh, tổng số tensor patches: {combined_pixel_values.shape[0]}")
    
#     # Câu hỏi prompt không thay đổi
#     question = prompt

#     generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)

#     print("Bắt đầu xử lý với model...")
#     # Truyền tensor đã được gộp vào model
#     response = model.chat(tokenizer, combined_pixel_values, question, generation_config)
#     print(f"Model trả về: {response}")

#     try:
#         start_index = response.find('{')
#         end_index = response.rfind('}') + 1
#         json_str = response[start_index:end_index]
#         json_response = json.loads(json_str)
#         return json_response
#     except Exception as e:
#         print(f"Không thể parse JSON từ response của model. Lỗi: {e}")
#         return {"error": "Could not parse model response to JSON", "raw_response": response}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
from typing import List
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import io
import json
import time # <-- Thêm thư viện time

# --- Thư viện cho API ---
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
import uvicorn

# =================================================================================
# PHẦN 1: CÁC HÀM XỬ LÝ ẢNH VÀ MODEL (Không thay đổi)
# =================================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def process_image_to_tensor(image_obj, input_size=448, max_num=12):
    image = image_obj.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# =================================================================================
# PHẦN 2: TẢI MODEL VÀ ĐỊNH NGHĨA API
# =================================================================================

app = FastAPI(title="Sổ Hồng OCR API", description="API để trích xuất thông tin từ ảnh sổ hồng.")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Đã phát hiện và đang sử dụng GPU NVIDIA (CUDA).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Đã phát hiện và đang sử dụng GPU Apple Silicon (MPS).")
else:
    device = torch.device("cpu")
    print("⚠️ Không tìm thấy GPU, đang sử dụng CPU. Quá trình xử lý sẽ rất chậm.")

model_name = "5CD-AI/Vintern-1B-v3_5"
print("Đang tải model...")
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
print("Model đã tải xong.")

# --- Định nghĩa API Endpoint với log gỡ lỗi ---
@app.post("/extract-land-title-info/")
async def extract_info(files: List[UploadFile] = File(...), prompt: str = Form(...)):
    start_time = time.time()
    print(f"\n{'='*20} BẮT ĐẦU REQUEST MỚI {'='*20}")
    print(f"DEBUG {time.time() - start_time:.2f}s: Bước 1 - Nhận được {len(files)} file(s).")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files sent.")
    
    list_of_pixel_values = []
    
    # Bắt đầu vòng lặp xử lý ảnh
    print(f"DEBUG {time.time() - start_time:.2f}s: Bước 2 - Bắt đầu vòng lặp tiền xử lý ảnh.")
    for file in files:
        print(f"DEBUG {time.time() - start_time:.2f}s:   Bước 2.1 - Đang xử lý file: {file.filename}")
        
        contents = await file.read()
        print(f"DEBUG {time.time() - start_time:.2f}s:   Bước 2.2 - Đã đọc {len(contents)} bytes từ file.")

        image = Image.open(io.BytesIO(contents))
        print(f"DEBUG {time.time() - start_time:.2f}s:   Bước 2.3 - Đã mở ảnh bằng PIL.")
        
        pixel_values_single_image = process_image_to_tensor(image, max_num=6)
        list_of_pixel_values.append(pixel_values_single_image)
        print(f"DEBUG {time.time() - start_time:.2f}s:   Bước 2.4 - Đã xử lý ảnh thành tensor.")

    print(f"DEBUG {time.time() - start_time:.2f}s: Bước 3 - Hoàn thành vòng lặp. Bắt đầu gộp tensor.")
    combined_pixel_values = torch.cat(list_of_pixel_values, dim=0).to(torch.bfloat16).to(device)
    print(f"DEBUG {time.time() - start_time:.2f}s: Bước 4 - Đã gộp các ảnh và chuyển sang GPU. Tổng số patches: {combined_pixel_values.shape[0]}")
    
    question = prompt
    generation_config = dict(max_new_tokens=2048, do_sample=False, num_beams=3, repetition_penalty=3.5)

    print(f"DEBUG {time.time() - start_time:.2f}s: Bước 5 - Bắt đầu xử lý với model.chat()...")
    response = model.chat(tokenizer, combined_pixel_values, question, generation_config)
    print(f"DEBUG {time.time() - start_time:.2f}s: Bước 6 - Model đã trả về response.")
    
    print(f"Model trả về: {response}")

    print(f"DEBUG {time.time() - start_time:.2f}s: Bước 7 - Bắt đầu parse JSON.")
    try:
        # ... (Khối code parse JSON giữ nguyên như cũ) ...
        start_bracket = response.find('[')
        start_curly = response.find('{')
        if start_bracket != -1 and (start_curly == -1 or start_bracket < start_curly):
            start_index = start_bracket
            end_char = ']'
        else:
            start_index = start_curly
            end_char = '}'
        if start_index == -1: raise ValueError("Không tìm thấy ký tự bắt đầu JSON")
        end_index = response.rfind(end_char)
        if end_index == -1 or end_index < start_index: raise ValueError("Không tìm thấy ký tự kết thúc JSON")
        json_str = response[start_index : end_index + 1]
        json_response = json.loads(json_str)
        print(f"DEBUG {time.time() - start_time:.2f}s: Bước 8 - Parse JSON thành công. Hoàn tất request.")
        return json_response
            
    except Exception as e:
        print(f"DEBUG {time.time() - start_time:.2f}s: Bước 8 FAILED - Không thể parse JSON. Lỗi: {e}")
        return {"error": "Could not parse model response to JSON", "raw_response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)