from absl import app, flags
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import threading

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Directory containing datasets.')
flags.DEFINE_integer('workers', 3, 'Number of workers.')


classes_dict = {
    'D00': '0',
    'D10': '1',
    'D20': '2',
    'D40': '3',
}


def xml_parser(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    name_list = []
    bbox_list = []
    for child in root:
        if child.tag == 'size':
            img_w, img_h = int(child[0].text), int(child[1].text)
        if child.tag == 'object':
            if child[0].text in classes_dict:
                name_list.append(child[0].text)
                for sub in child:
                    if sub.tag == 'bndbox':
                        xmin, ymin = int(round(float(sub[0].text), 0)), int(round(float(sub[1].text), 0))
                        xmax, ymax = int(round(float(sub[2].text), 0)), int(round(float(sub[3].text), 0))
                        bbox_list.append((xmin, ymin, xmax, ymax))
            
    return [(img_w, img_h), name_list, bbox_list]


def yolo_style(xml_style):
    yolo_classes = [classes_dict[y_cls] for y_cls in xml_style[1]]
    img_w, img_h = xml_style[0][0], xml_style[0][1]
    bbox_list = []
    for coord_list in xml_style[2]:
        norm_xmin, norm_ymin = coord_list[0] / img_w, coord_list[1] / img_h
        norm_xmax, norm_ymax = coord_list[2] / img_w, coord_list[3] / img_h
        width_norm, height_norm = (norm_xmax - norm_xmin) / 2, (norm_ymax - norm_ymin) / 2
        x_center_norm, y_center_norm = norm_xmin + width_norm, norm_ymin + height_norm
        bbox_list.append((round(x_center_norm, 6), round(y_center_norm, 6), round(width_norm * 2, 6) , round(height_norm * 2, 6)))
        
    return [yolo_classes, bbox_list]
            

def yolo_writer(yolo_label, f_name, output_dir):
    if not os.path.exists(os.path.join(output_dir, f_name + '.txt')):
        with open(os.path.join(output_dir, f_name + '.txt'), 'a') as f:
            for line in range(len(yolo_label[0])):
                f.write(f'{yolo_label[0][line]} {yolo_label[1][line][0]} {yolo_label[1][line][1]} {yolo_label[1][line][2]} {yolo_label[1][line][3]}\n')


def yolo_converter(xml_file_list, input_dir, output_dir):
    for i in xml_file_list:
        print(f'Write YOLO label for: {i}')
        file_name = i.split('.')[0]
        xml_file = os.path.join(input_dir, i)
        xml_label = xml_parser(xml_file)
        yolo_label = yolo_style(xml_label)
        yolo_writer(yolo_label, file_name, output_dir)


def main(argv):
    countries = os.listdir(FLAGS.data_dir)
    
    for c in countries:
        xml_path = Path(os.path.join(FLAGS.data_dir, c, 'train', 'annotations', 'xmls'))
        yolo_path = Path(os.path.join(FLAGS.data_dir, c, 'train', 'annotations', 'yolo'))
        yolo_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of xml files
        xml_files = os.listdir(xml_path)
        slice_size = int(len(xml_files) / FLAGS.workers)
        rest_size = len(xml_files) - slice_size * FLAGS.workers
        
        # Slice the list of xml files by no workers and add them to threads list
        threads = list()
        for w in range(FLAGS.workers):
            xml_slice = xml_files[slice_size * w: slice_size * (w + 1)]
            thread = threading.Thread(target=yolo_converter, args=(xml_slice, xml_path, yolo_path,))
            threads.append(thread)
            thread.start()
        
        # Start thread with overrun
        if rest_size > 0:
            rest_slice = xml_files[-rest_size:]
            thread = threading.Thread(target=yolo_converter, args=(rest_slice, xml_path, yolo_path))
            threads.append(thread)
            thread.start()
        
        for index, thread in enumerate(threads):
            thread.join()      
    
    
if __name__ == '__main__':
    app.run(main)