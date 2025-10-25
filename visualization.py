from PIL import Image
import matplotlib.pyplot as plt
# Proceeding with the same visualizations
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Load the images
image_paths = [
   "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/1016887272.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/7162685234.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/3000017878.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/4971484184.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/92679312.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/2853682342.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/5131842202.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/4859170265.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/2612125121.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/3134644844.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/316298162.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/3601843201.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/5086989679.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/1246863003.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/2193001254.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/4283472819.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/1117972841.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/48614561.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/6232601127.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/4439654945.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/269898428.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/2113592981.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/1542970158.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/4743795506.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/1313961775.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/4005756399.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/4844409798.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/4970590451.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/2536995789.jpg",
    "/mnt/my_disk/yuanhui/140server/data/f30k/Flickr30k_Images/images/2230134548.jpg",
]

file_path_latest = '/mnt/my_disk/yuanhui/140server/wwbl_dense_two_context_len/predictions_data-flicker-_th-0.85_min-2_model-MSCOCO_wwbl_dense_context_len_24/predictions_0-30.pickle'

# Ensure output dirs exist
out_root = '/mnt/my_disk/yuanhui/140server/results'
os.makedirs(out_root, exist_ok=True)
os.makedirs(os.path.join(out_root, 'originals'), exist_ok=True)
os.makedirs(os.path.join(out_root, 'overlays'), exist_ok=True)

# Load and display the images to ensure they are accessible
images = [Image.open(image_path).convert('RGB') for image_path in image_paths]

# Show the images
fig, axes = plt.subplots(1, 30, figsize=(30, 5))
for ax, img in zip(axes, images):
    ax.imshow(img)
    ax.axis('off')
fig.savefig(os.path.join(out_root, 'lucky_ten.png'))
plt.close(fig)

# Load predictions
with open(file_path_latest, 'rb') as file:
    predictions_data_latest = pickle.load(file)

# Visualize
for i in range(0, 30):
    # 1) 每张图片先保存一次“原图”（只保存一次，避免 j 循环重复）
    base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
    orig_out = os.path.join(out_root, 'originals', f'{i:02d}_{base_name}.jpg')
    images[i].save(orig_out, quality=95)

    # 2) 两个候选的热力图叠加
    for j in range(0, 2):
        description = predictions_data_latest[i][j][0]  # Description of the image
        heatmap_array = np.array(predictions_data_latest[i][j][1])  # Heatmap for the image

        image_array = np.array(images[i])  # RGB
        heatmap_resized = cv2.resize(heatmap_array, (image_array.shape[1], image_array.shape[0]))
        heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-6)
        heatmap_colored = plt.cm.jet(heatmap_resized)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

        overlayed_image = cv2.addWeighted(image_array, 0.6, heatmap_colored, 0.4, 0)

        fig, ax = plt.subplots()
        ax.imshow(overlayed_image)
        ax.axis('off')
        ax.set_title(str(description))
        overlay_out = os.path.join(out_root, 'overlays', f'{i:02d}_{base_name}_top{j+1}.jpg')
        fig.savefig(overlay_out, bbox_inches='tight', dpi=150)
        plt.close(fig)