import cv2
from PIL import Image
from shapely.geometry import Polygon
import csv
import os
import os.path
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import sys
torch.set_grad_enabled(False)


CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    image_displayed = False
    for i in range(len(boxes)):
        if prob[i][1] > 0.5 and not image_displayed:
            plt.figure(figsize=(16,10))
            plt.imshow(pil_img)
            ax = plt.gca()
            colors = COLORS * 100
            for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
                if p[1] > 0.8:
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                              fill=False, color=c, linewidth=3))
                    cl = 1
                    text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                    ax.text(xmin, ymin, text, fontsize=15,
                            bbox=dict(facecolor='yellow', alpha=0.5))
            plt.axis('off')
            plt.show()
            image_displayed = True


model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()


def opredelenie_procenta_vhoda(poligon, chilovik_koordinita):
    rectangle_coords = chilovik_koordinita
    polygon_coords = poligon
    print(polygon_coords[0])
    rectangle = Polygon(rectangle_coords)
    polygon = Polygon(polygon_coords)
    intersection_area = rectangle.intersection(polygon).area
    rectangle_area = rectangle.area
    percentage_inside = intersection_area / rectangle_area * 100
    return percentage_inside


# im = Image.open('img.png') # ------------------ картинку сюда ---------------
# img = transform(im).unsqueeze(0)
# outputs = model(img)
# probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
# keep = probas.max(-1).values > 0.8
# bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
# print(int(bboxes_scaled[0][0]))
# plot_results(im, probas[keep], bboxes_scaled)
# xyx1 = int(bboxes_scaled[0][0])
# xyx2 = int(bboxes_scaled[0][1])
# yxy1 = int(bboxes_scaled[0][2])
# yxy2 = int(bboxes_scaled[0][3])
# print(xyx1)


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def save_image(image, directory, filename):
    cv2.imwrite(os.path.join(directory, filename), image)


my_dict = {
    'DpR-Csp-uipv-ShV-V1': [(534, 288), (834, 219), (1365, 580), (1124, 806)],
    'Pgp-com2-K-1-0-9-36': [(511, 214), (776, 265), (788, 367), (445, 720), (225, 717), (195, 597), (591, 315), (468, 265)],  # Добавьте свои значения
    'Pgp-lpc2-K-0-1-38': [(181, 321), (378, 310), (379, 360), (553, 334), (544, 274), (907, 227), (996, 363), (895, 390), (881, 435), (582, 491), (570, 435), (375, 459), (371, 541), (170, 551)],
    'Phl-com3-Shv2-9-K34': [(1335, 640), (1505, 662), (1491, 776), (1290, 752)],
    'Php-Angc-K3-1': [(471, 717), (1434, 737), (1460, 894), (1224, 896), (1223, 761), (692, 754), (680, 916), (444, 906)],
    'Php-Angc-K3-8': [(1036, 831), (480, 475), (614, 421), (1171, 691)],
    'Php-Ctm-K-1-12-56': [(516, 261), (1344, 580), (452, 1078), (84, 352)],
    'Php-Ctm-Shv1-2-K3': [(172, 108), (115, 745), (441, 669), (422, 540), (864, 421), (864, 259), (1363, 151), (1881, 421), (1593, 529), (1824, 723), (1094, 1080)],
    'Php-nta4-shv016309-k2-1-7': [(0, 1080), (0, 712), (192, 518), (384, 518), (825, 97), (902, 97), (1132, 367), (1132, 583), (1555, 572), (1574, 475), (1920, 475), (1920, 1080)],
    'Spp-210-K1-3-3-5': [(718, 204), (1128, 340), (1128, 720), (541, 720), (345, 607)],
    'Spp-210-K1-3-3-6': [(223, 345), (639, 193), (951, 477), (494, 707)]
    # Добавьте другие ключи и значения по необходимости
}

if __name__ == "__main__":
    name = sys.argv[1]  # Получаем первый аргумент из командной строки - НАЗВАНИЕ КАМЕРЫ
    name2 = sys.argv[2] # НАЗВАНИЕ ПАПКИ ДЛЯ ТЕСТА (ДОЛЖНА БЫТЬ НА УРОВНЕ ПРОЕКТА)

poligon = my_dict[name] # полигон камеры

with open('data.csv', 'a', newline='') as w_file:
    file_writer = csv.writer(w_file)
    file_writer.writerow(('camera_name','frame_filename','in_danger_zone','percent'))

folder_path = name
ffff =0
# Проходим по всем файлам в папке
for filename in os.listdir(folder_path):
    # Проверяем, что файл является картинкой
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        im = Image.open(folder_path+'/'+filename)  # ------------------ картинку сюда ---------------
        img = transform(im).unsqueeze(0)
        outputs = model(img)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.8
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

        for i in range(len(bboxes_scaled)):
            xyx1 = int(bboxes_scaled[i][0])
            xyx2 = int(bboxes_scaled[i][1])
            yxy1 = int(bboxes_scaled[i][2])
            yxy2 = int(bboxes_scaled[i][3])
            poligon2 = [(xyx1, xyx2), (yxy1, xyx2), (yxy1, yxy2), (xyx1, yxy2)]
            print(str(opredelenie_procenta_vhoda(poligon, poligon2))+' - процент входа')
            with open('data.csv', 'a', newline='') as w_file:
                procent = opredelenie_procenta_vhoda(poligon, poligon2)
                hook = False
                if procent > 15:
                    hook = True
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)
                    x1, y1 = xyx1, xyx2
                    x2, y2 = yxy1, yxy2
                    cropped_image = image[y1:y2, x1:x2]

                    # Увеличьте размер обрезанного изображения
                    scale_factor = 2  # Примерный масштабный коэффициент
                    cropped_image = cv2.resize(cropped_image, (0, 0), fx=scale_factor, fy=scale_factor)
                    output_path = 'violators' + '\\' + name + '\\' + str(ffff) + '.jpg'
                    cv2.imwrite(output_path, cropped_image)
                    ffff+=1
                file_writer = csv.writer(w_file, quoting=csv.QUOTE_ALL)
                file_writer.writerow((
                    folder_path,
                    str(filename),
                    str(hook),
                    str(round(procent, 2))
                ))
