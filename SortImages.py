import shutil
import re
from pathlib import Path
from PIL import Image

source_dir = Path('Coursework 2 Dataset/Car Dataset Altered')
target_dir = Path('Yolo Classification Dataset')

train_split, val_split = 10, 13
target_size = (224, 224)

def get_index_label(path):
    text = path.name
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def prepare_dataset():
    # Don't add duplicate sets of files
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # Our class labels are the names of the directories storing the images
    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    
    for cls in classes:
        for folder in ['train', 'val', 'test']:
            (target_dir / folder / cls).mkdir(parents=True, exist_ok=True)
        
        images = [f for f in (source_dir / cls).iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        images.sort(key=get_index_label)
        
        for i, img_path in enumerate(images):
            match = re.match(r"(.*\]).*(\..*)", img_path.name)
            new_name = (match.group(1) + match.group(2)) if match else img_path.name
            
            if i < train_split:
                subfolder = 'train'
            elif i < val_split:
                subfolder = 'val'
            else:
                subfolder = 'test'
            
            dest_path = target_dir / subfolder / cls / new_name
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    img.save(dest_path)
            except Exception as e:
                print(f"Error processing image {img_path.name}: {e}")

    print(f"The dataset has been saved to: {target_dir.absolute()}")

if __name__ == "__main__":
    prepare_dataset()