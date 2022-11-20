import os
from pathlib import Path
from absl import app, flags
import cv2
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Directory containing datasets.')

def get_img_size(img_dir: str) -> tuple:
    imgs_list = os.listdir(img_dir)
    test_img = os.path.join(img_dir, imgs_list[0])
    
    img = cv2.imread(test_img)
    
    return img.shape[1], img.shape[0]
    

def get_no_imgs(img_dir: str) -> int:
    imgs_list = os.listdir(img_dir)
    
    return len(imgs_list)
    

def get_no_labels(label_dir: str) -> int:
    label_list = os.listdir(label_dir)
    
    cnt = 0
    for label in label_list:
        with open(os.path.join(label_dir, label), 'r') as annotation:
            lines = annotation.readlines()
            cnt += len(lines)
    
    return cnt


def dump_results(dump_file: str, country: str, img_shape: tuple, nr_imgs: int, nr_labels: int):
    with open(dump_file, 'a') as d_file:
        string = '|'
        string = string + '{c:<20}'.format(c=country)
        string = string + '|'
        string = string + '{i_s_1:<4}'.format(i_s_1=img_shape[0])
        string = string + ','
        string = string + ' {i_s_2:<9}'.format(i_s_2=img_shape[1])
        string = string + '|'
        string = string + '{i_n:<15}'.format(i_n=nr_imgs)
        string = string + '|'
        string = string + '{l_n:<9}'.format(l_n=nr_labels)
        string = string + '|\n'
        d_file.write(string)


def main(argv):
    # List of data directories
    countries = os.listdir(FLAGS.data_dir)
    
    stats_path = Path(os.path.join(FLAGS.data_dir, 'stats'))
    stats_path.mkdir(parents=True, exist_ok=True)
    dump_file = os.path.join(stats_path, 'restuls.txt')
    with open(dump_file, 'w') as d_file:
        string = '|'
        string = string + '{c:<20}'.format(c='Country')
        string = string + '|'
        string = string + '{i_s:<15}'.format(i_s='Image Size')
        string = string + '|'
        string = string + '{i_n:<15}'.format(i_n='No Images')
        string = string + '|'
        string = string + '{l_n:<9}'.format(l_n='No Labels')
        string = string + '|\n'
        string = string + '-' * 64 + '\n'
        d_file.write(string)
    
    for c in tqdm(countries):
        if not c == 'stats':
            img_path = os.path.join(FLAGS.data_dir, c, 'train', 'images')
            label_path = os.path.join(FLAGS.data_dir, c, 'train', 'annotations', 'yolo')
            img_size = get_img_size(img_path)
            no_imgs = get_no_imgs(img_path)
            no_labels = get_no_labels(label_path)
            
            dump_results(dump_file, c, img_size, no_imgs, no_labels)
    

if __name__ == '__main__':
    app.run(main)