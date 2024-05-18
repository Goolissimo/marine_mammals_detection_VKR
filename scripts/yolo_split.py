# Импорт библиотек
import os
import shutil

# Создание каталогов
os.mkdir(f'C:/Users/Савелий/Desktop/dataset/yolo/train/labels')
os.mkdir(f'C:/Users/Савелий/Desktop/dataset/yolo/train/images')

os.mkdir(f'C:/Users/Савелий/Desktop/ДИПЛОМ/dataset/yolo/valid/labels')
os.mkdir(f'C:/Users/Савелий/Desktop/ДИПЛОМ/dataset/yolo/valid/images')

os.mkdir(f'C:/Users/Савелий/Desktop/dataset/yolo/test/labels')
os.mkdir(f'C:/Users/Савелий/Desktop/dataset/yolo/test/images')

# Раздение изображений по каталогам
train = 'C:/Users/Савелий/Desktop/dataset/yolo/train'
valid = 'C:/Users/Савелий/Desktop/dataset/yolo/valid'

for folder, _, files in os.walk(train):
    for file in files:
        if file[-4:] == '.jpg':
            shutil.move(os.path.join(folder, file), f'{train}/images')
        elif file[-4:] == '.txt':
            shutil.move(os.path.join(folder, file), f'{train}/labels')

for folder, _, files in os.walk(valid):
    for file in files:
        if file[-4:] == '.jpg':
            shutil.move(os.path.join(folder, file), f'{valid}/images')
        elif file[-4:] == '.txt':
            shutil.move(os.path.join(folder, file), f'{valid}/labels')
