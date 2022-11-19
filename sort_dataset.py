import os
from absl import app, flags
from pathlib import Path
import threading
import random
import math
import shutil

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Directory containing datasets.')
flags.DEFINE_integer('workers', 3, 'Number of workers.')


def split_dataset(dataset: list, train_size: float) -> list:
    train_len = int(len(dataset) * train_size)
    val_len = int(len(dataset) - train_len)
    
    return dataset[:train_len], dataset[-val_len:]


def shuffle_dataset(dataset: list) -> list:
    return random.sample(dataset, len(dataset))


def worker_split(dataset: list, workers: int) -> list:
    datasets_list = []
    if len(dataset) / workers % 2 == 0:
        split_idx = int(len(dataset) / workers)
        for w in range(workers):
            datasets_list.append(dataset[split_idx * w: split_idx * (w + 1)])
    else:
        overrun = math.ceil(len(dataset) / workers % 2)
        split_idx = split_idx = int(len(dataset) / workers)
        for w in range(workers):
            datasets_list.append(dataset[split_idx * w: split_idx * (w + 1)])
        new_list = datasets_list[-1] + dataset[-overrun:]
        datasets_list.pop(-1)
        datasets_list.append(new_list)
        
    return datasets_list


def worker(dataset, output_dir_dict: dict, image_folder_in: str, label_folder_in: str, train=True):
    if train:
        img_folder = output_dir_dict['img_train']
        label_folder = output_dir_dict['label_train']
    else:
        img_folder = output_dir_dict['img_val']
        label_folder = output_dir_dict['label_val']
    
    for file in dataset:
        img_src = os.path.join(image_folder_in, file)
        img_dst = os.path.join(img_folder, file)
        label_src = os.path.join(label_folder_in, file.split('.')[0] + '.txt')
        label_dst = os.path.join(label_folder, file.split('.')[0] + '.txt')
        
        print(f'Copy file: {img_src}')
        shutil.copy(img_src, img_dst)
        print(f'Copy file: {label_src}')
        shutil.copy(label_src, label_dst)


def main(argv):
    print(f'no workers: {FLAGS.workers}')
    # List of data directories
    countries = os.listdir(FLAGS.data_dir)
    
    # Prepare training folder structure
    new_folder_list = []
    root = Path(os.path.join(FLAGS.data_dir, 'rdd_2022_dashboard'))
    root.mkdir(parents=True, exist_ok=True)
    new_folder_list.append(root)
    img_folder = Path(os.path.join(root, 'images'))
    img_folder.mkdir(parents=True, exist_ok=True)
    new_folder_list.append(img_folder)
    img_train_folder = Path(os.path.join(img_folder, 'train'))
    img_train_folder.mkdir(parents=True, exist_ok=True)
    new_folder_list.append(img_train_folder)
    img_val_folder = Path(os.path.join(img_folder, 'val'))
    img_val_folder.mkdir(parents=True, exist_ok=True)
    new_folder_list.append(img_val_folder)
    label_folder = Path(os.path.join(root, 'labels'))
    label_folder.mkdir(parents=True, exist_ok=True)
    new_folder_list.append(label_folder)
    label_train_folder = Path(os.path.join(label_folder, 'train'))
    label_train_folder.mkdir(parents=True, exist_ok=True)
    new_folder_list.append(label_train_folder)
    label_val_folder = Path(os.path.join(label_folder, 'val'))
    label_val_folder.mkdir(parents=True, exist_ok=True)
    new_folder_list.append(label_val_folder)
    
    # Print status update
    for i in new_folder_list:
        print(f'New folder created: {i}')
        
    del new_folder_list
    
    out_dict = {
        'img_train': img_train_folder,
        'img_val': img_val_folder,
        'label_train': label_train_folder,
        'label_val': label_val_folder,
    }
        
    for c in countries:
        image_folder = os.path.join(FLAGS.data_dir, c, 'train', 'images')
        annotation_folder = os.path.join(FLAGS.data_dir, c, 'train', 'annotations', 'yolo')
        imgs_list = os.listdir(image_folder)
        
        # Shuffle images
        shuffled_images = shuffle_dataset(imgs_list)
        
        # Split in train and validation set
        train, val = split_dataset(shuffled_images, train_size=.8)
        
        train_splits, val_splits = worker_split(train, workers=FLAGS.workers), worker_split(val, workers=FLAGS.workers)
        threads = list()
        for w in range(FLAGS.workers):
            thread = threading.Thread(target=worker, args=(train_splits[w], out_dict, image_folder, annotation_folder,))
            threads.append(thread)
            thread.start()
            
        for index, thread in enumerate(threads):
            thread.join()
            
        threads = list()
        for w in range(FLAGS.workers):
            thread = threading.Thread(target=worker, args=(val_splits[w], out_dict, image_folder, annotation_folder, False,))
            threads.append(thread)
            thread.start()
            
        for index, thread in enumerate(threads):
            thread.join()
    
    
    
if __name__ == '__main__':
    app.run(main)