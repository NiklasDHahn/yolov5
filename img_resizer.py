import os
from PIL import Image
from absl import app, flags
import threading
import math

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Directory containing datasets.')
flags.DEFINE_integer('workers', 3, 'Number of workers.')
flags.DEFINE_integer('target_size', 640, 'Target size of the images.')


def worker_split(dataset: list, workers: int) -> list:
    datasets_list = []
    if len(dataset) / workers % 2 == 0 or workers == 1:
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


def worker(img_list: list, size: int):
    for i in img_list:
        print(f'Processing {i}')
        img = Image.open(i)
        width, height = img.size
        
        if width == height:
            new_img = img.resize((size, size))
        elif width < height:
            scale_factor = size / width
            width = size
            height = int(height * scale_factor)
            new_img = img.resize((width, height))
        elif height < width:
            scale_factor = size / height
            height = size
            width = int(width * scale_factor)
            new_img = img.resize((width, height))
            
        new_img.save(i)


def main(argv):
    train_dir = os.path.join(FLAGS.data_dir, 'train')
    val_dir = os.path.join(FLAGS.data_dir, 'val')
    
    train_list = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
    val_list = [os.path.join(val_dir, img) for img in os.listdir(val_dir)]
    
    final_list = train_list + val_list
    
    split_list = worker_split(final_list, workers=FLAGS.workers)
    
    del train_list, val_list, final_list
    
    threads = list()
    for w in range(FLAGS.workers):
            thread = threading.Thread(target=worker, args=(split_list[w], FLAGS.target_size))
            threads.append(thread)
            thread.start()
            
    for index, thread in enumerate(threads):
        thread.join()
    
    
    
if __name__ == '__main__':
    app.run(main)