import os
import requests
from tqdm import tqdm
import zipfile

# 创建保存数据的目录
data_dir = 'coco_dataset'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 下载文件函数
def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

# 下载并解压数据集
def download_and_extract(url, dest):
    zip_path = os.path.join(data_dir, dest)
    download_file(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(zip_path)

# COCO 2017 数据集 URL
coco_urls = {
    # 'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    # 'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    # 'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

# 下载并解压所有文件
for key, url in coco_urls.items():
    print(f'Downloading {key}...')
    download_and_extract(url, f'{key}.zip')

print('COCO dataset download and extraction completed.')
