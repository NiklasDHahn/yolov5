from data_stats import get_img_size, get_no_imgs, get_no_labels
import os

test_img_path_list = [r'D:\yolov5\dataset\RDD2022_all_countries\China_Drone\train\images',
                      r'D:\yolov5\dataset\RDD2022_all_countries\India\train\images',
                      r'D:\yolov5\dataset\RDD2022_all_countries\United_States\train\images']

test_label_path = r'D:\yolov5\dataset\RDD2022_all_countries\China_Drone\train\annotations\yolo'

china_drone_size = (512, 512)
india_size = (720, 720)
united_states_size = (640, 640)

assert get_img_size(test_img_path_list[0]) == china_drone_size
assert get_img_size(test_img_path_list[1]) == india_size
assert get_img_size(test_img_path_list[2]) == united_states_size
print('get_img_size test completed: Successful')


no_images_china_drone = 2401
no_labels_china_drone = 3068

assert get_no_imgs(test_img_path_list[0]) == no_images_china_drone
print('get_no_imgs test completed: Successful')

assert get_no_labels(test_label_path) == no_labels_china_drone
print('get_no_labels test completed: Successful')