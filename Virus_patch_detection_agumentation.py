# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import PIL.ImageDraw
import xml.etree.ElementTree as ET

# %%

#!pip install matplotlib
#!pip install numpy --upgrade
#pip install matplotlib
#import torch
#!pip install tensorflow==1.14
#!pip install --ignore-installed --upgrade tensorflow 

# %%
import imgaug as ia

# %%
import imgaug.augmenters as iaa

# %%
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

# %%
#!pip install 'h5py<3.0.0'

# %%
#!python /home/amritasingh/Amrita_Project/3X3_5x5/voc2coco/voc2coco.py \--ann_dir /home/amritasingh/Downloads/Shalini/dipak/test_c/c \--ann_ids /home/amritasingh/Downloads/Shalini/dipak/test_c/test.txt \--labels /home/amritasingh/Downloads/Shalini/dipak/test_c/labels.txt \--output /home/amritasingh/Downloads/Shalini/dipak/test_c/c.json \--ext xml


# %%
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)  # init TF ...
config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)  # w/o taking ...
with tf.compat.v1.Session(config=config): pass            # all GPU memory



# %%
#dataset_location = '/home/amritasingh/Downloads/WCEBleedGen/bleeding'

# %%
dataset_location = '/home/amritasingh/Downloads/Downloads/ABC/ABCD1/'

# %%
anchors = ((0.57273, 0.677385),  # anchor 1, width & height, unit: cell_size
           (1.87446, 2.06253),   # anchor 2,
           (3.33843, 5.47434),   # ...
           (7.88282, 3.52778),
           (9.77052, 9.16828))

# %%
classes = ['cell']
colors = [(255,0,255)]
class_weights = [1.0]
lambda_coord = 1.0
lambda_noobj = 1.0
lambda_obj = 5.0
lambda_class = 1.0


# %%
images_location = os.path.join(dataset_location, 'JPEGImages')
annotations_location = os.path.join(dataset_location, 'Annotations')


# %%
#images_location = os.path.join(dataset_location, 'cell_Pic1')
#annotations_location = os.path.join(dataset_location, 'Xml_cell')



# %%
#images_location = os.path.join(dataset_location, 'cell_Pic1')
#annotations_location = os.path.join(dataset_location, 'Xml_cell')

#filename_list_xml = sorted(os.listdir(annotations_location))
#display(filename_list_xml[:3])

# %%
filename_list_xml = sorted(os.listdir(annotations_location))
display(filename_list_xml[:3])

# %%
class ImageWrapper:
    def __init__(self, filepath, width, height, depth):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.depth = depth
        self.objects = []
    def __str__(self):
        return f'{self.filepath}\n' \
               f'w:{self.width} h:{self.height} d:{self.depth}'

# %%
class BBoxWrapper:
    def __init__(self, classid, score, xmin ,ymin, xmax, ymax):
        self.classid = classid
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    def __str__(self):
        return f'{self.classid} {self.score} ' \
               f'{self.xmin} {self.ymin} {self.xmax} {self.ymax}'

# %%
import os
import xml.etree.ElementTree as ET
import numpy as np
import PIL
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the classes and their indices
classes = ['virus']
colors = [(255, 255, 255)]
class_weights = [1.0]
lambda_coord = 1.0
lambda_noobj = 1.0
lambda_obj = 5.0
lambda_class = 1.0
images_location = os.path.join(dataset_location, 'JPEGImages')
annotations_location = os.path.join(dataset_location, 'Annotations')
# Paths to the dataset (you need to define dataset_location and filename_list_xml)
dataset_location = dataset_location = '/home/amritasingh/Downloads/Downloads/ABC/ABCD1/'# Replace with actual path
filename_list_xml = filename_list_xml = sorted(os.listdir(annotations_location))  # Replace with actual list of XML filenames


# Define the augmentation pipeline
seq = iaa.Sequential([
    iaa.Multiply(2.0 / 255.0),  # scale pixel values
    iaa.Crop(px=(0, 16)),       # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),            # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
    iaa.Rot90(1)                # rotate images by 90 degrees
])
def draw_bboxes(image, bboxes, class_labels, title='Image'):
    """Draw bounding boxes on the image."""
    fig, ax = plt.subplots(1, figsize=(12, 6))
    ax.imshow(image)
    for bbox, class_label in zip(bboxes, class_labels):
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin, f'{classes[class_label]}', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.title(title)
    plt.axis('off')
    plt.show()

class ImageWrapper:
    def __init__(self, filepath, width, height, depth):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.depth = depth
        self.image = None
        self.objects = []

class BBoxWrapper:
    def __init__(self, classid, score, xmin, ymin, xmax, ymax):
        self.classid = classid
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

image_wrapper_list = []

for filename_xml in filename_list_xml:
    filepath_xml = os.path.join(annotations_location, filename_xml)
    tree = ET.parse(filepath_xml)

    filename = tree.find('./filename').text
    w = tree.find('./size/width').text
    h = tree.find('./size/height').text
    d = tree.find('./size/depth').text

    filepath_jpg = os.path.join(images_location, filename)
    assert os.path.isfile(filepath_jpg)

    iw_original = ImageWrapper(filepath=filepath_jpg, width=int(w), height=int(h), depth=int(d))

    object_elements = tree.findall('./object')
    bboxes = []
    class_labels = []

    for obj_el in object_elements:
        name = obj_el.find('./name').text
        xmin = int(obj_el.find('./bndbox/xmin').text)
        ymin = int(obj_el.find('./bndbox/ymin').text)
        xmax = int(obj_el.find('./bndbox/xmax').text)
        ymax = int(obj_el.find('./bndbox/ymax').text)

        if name in classes:
            classid = classes.index(name)
            bboxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(classid)
        else:
            raise ValueError(f"Unknown class name: {name}")

    image = np.array(PIL.Image.open(filepath_jpg))
    iw_original.image = image
    iw_original.objects.clear()
    for bbox, classid in zip(bboxes, class_labels):
        xmin, ymin, xmax, ymax = bbox
        bbw = BBoxWrapper(classid=classid, score=1.0,
                          xmin=int(xmin), ymin=int(ymin),
                          xmax=int(xmax), ymax=int(ymax))
        iw_original.objects.append(bbw)

    iw_original.image = iw_original.image.astype(np.float32)
    image_wrapper_list.append(iw_original)

    # Prepare bounding boxes for augmentation
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
        for xmin, ymin, xmax, ymax in bboxes
    ], shape=image.shape)

    # Apply augmentations
    augmented = seq(image=image, bounding_boxes=bbs)

    # Extract augmented image and bounding boxes
    augmented_image = augmented[0]  # image is the first item
    augmented_bboxes = [
        [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]
        for b in augmented[1].bounding_boxes  # bounding_boxes is the second item
    ]
    augmented_class_labels = class_labels

    print(f"Original image shape: {image.shape}")
    print(f"Augmented image shape: {augmented_image.shape}")

    augmented_image = augmented_image.astype(np.float32)

    iw_augmented = ImageWrapper(filepath=filepath_jpg, width=int(w), height=int(h), depth=int(d))
    iw_augmented.image = augmented_image
    iw_augmented.objects.clear()
    for bbox, classid in zip(augmented_bboxes, augmented_class_labels):
        xmin, ymin, xmax, ymax = bbox
        bbw = BBoxWrapper(classid=classid, score=1.0,
                          xmin=int(xmin), ymin=int(ymin),
                          xmax=int(xmax), ymax=int(ymax))
        iw_augmented.objects.append(bbw)

    image_wrapper_list.append(iw_augmented)

    # Draw and display both original and augmented images
    draw_bboxes(image, bboxes, class_labels, title='Original Image')
    draw_bboxes(augmented_image, augmented_bboxes, augmented_class_labels, title='Augmented Image')


# %%
import matplotlib.pyplot as plt
import numpy as np
import PIL

# Assuming augmented_image is a numpy array of the augmented image

# Display the original and augmented image
def show_images(original_image, augmented_image):
    # Create a figure and axis
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')  # Hide the axis

    # Display the augmented image
    axs[1].imshow(augmented_image)
    axs[1].set_title('Augmented Image')
    axs[1].axis('off')  # Hide the axis

    # Show the plot
    plt.show()

# Load original image
original_image = np.array(PIL.Image.open(filepath_jpg))

# Call the function to show the images
show_images(original_image, augmented_image)


# %%
import os
import xml.etree.ElementTree as ET
import numpy as np
import PIL
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the classes and their indices
classes = ['virus']
colors = [(255, 255, 255)]
class_weights = [1.0]
lambda_coord = 1.0
lambda_noobj = 1.0
lambda_obj = 5.0
lambda_class = 1.0

# Paths to the dataset
images_location = os.path.join(dataset_location, 'JPEGImages')
annotations_location = os.path.join(dataset_location, 'Annotations')

# Define the augmentation pipeline
seq = iaa.Sequential([
    iaa.Multiply(2.0 / 255.0),  # scale pixel values
    iaa.Crop(px=(0, 16)),       # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),            # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
    iaa.Rot90(1)                # rotate images by 90 degrees
])
def draw_bboxes(image, bboxes, class_labels, title='Image'):
    """ Draw bounding boxes on the image. """
    fig, ax = plt.subplots(1, figsize=(12, 6))
    ax.imshow(image)
    for bbox, class_label in zip(bboxes, class_labels):
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin, f'{classes[class_label]}', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.title(title)
    plt.axis('off')
    plt.show()

image_wrapper_list = []

for filename_xml in filename_list_xml:
    filepath_xml = os.path.join(annotations_location, filename_xml)
    tree = ET.parse(filepath_xml)

    filename = tree.find('./filename').text
    w = tree.find('./size/width').text
    h = tree.find('./size/height').text
    d = tree.find('./size/depth').text

    filepath_jpg = os.path.join(images_location, filename)
    assert os.path.isfile(filepath_jpg)

    iw_original = ImageWrapper(filepath=filepath_jpg, width=int(w), height=int(h), depth=int(d))

    object_elements = tree.findall('./object')
    bboxes = []
    class_labels = []

    for obj_el in object_elements:
        name = obj_el.find('./name').text
        xmin = obj_el.find('./bndbox/xmin').text
        ymin = obj_el.find('./bndbox/ymin').text
        xmax = obj_el.find('./bndbox/xmax').text
        ymax = obj_el.find('./bndbox/ymax').text

        if name in classes:
            classid = classes.index(name)
            bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            class_labels.append(classid)
        else:
            raise ValueError(f"Unknown class name: {name}")

    image = np.array(PIL.Image.open(filepath_jpg))
    iw_original.image = image
    iw_original.objects.clear()
    for bbox, classid in zip(bboxes, class_labels):
        xmin, ymin, xmax, ymax = bbox
        bbw = BBoxWrapper(classid=classid, score=1.0,
                          xmin=int(xmin), ymin=int(ymin),
                          xmax=int(xmax), ymax=int(ymax))
        iw_original.objects.append(bbw)

    iw_original.image = iw_original.image.astype(np.float32)
    image_wrapper_list.append(iw_original)

    # Prepare bounding boxes for augmentation
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
        for xmin, ymin, xmax, ymax in bboxes
    ], shape=image.shape)

    # Apply augmentations
    augmented = seq(image=image, bounding_boxes=bbs)

    # Extract augmented image and bounding boxes
    augmented_image = augmented[0]  # image is the first item
    augmented_bboxes = [
        [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]
        for b in augmented[1]  # bounding_boxes is the second item
    ]
    augmented_class_labels = class_labels

    print(f"Original image shape: {image.shape}")
    print(f"Augmented image shape: {augmented_image.shape}")

    augmented_image = augmented_image.astype(np.float32)

    iw_augmented = ImageWrapper(filepath=filepath_jpg, width=int(w), height=int(h), depth=int(d))
    iw_augmented.image = augmented_image
    iw_augmented.objects.clear()
    for bbox, classid in zip(augmented_bboxes, augmented_class_labels):
        xmin, ymin, xmax, ymax = bbox
        bbw = BBoxWrapper(classid=classid, score=1.0,
                          xmin=int(xmin), ymin=int(ymin),
                          xmax=int(xmax), ymax=int(ymax))
        iw_augmented.objects.append(bbw)

    image_wrapper_list.append(iw_augmented)

    # Draw and display both original and augmented images
    #draw_bboxes(image, bboxes, class_labels, title='Original Image')
    #draw_bboxes(augmented_image, augmented_bboxes, augmented_class_labels, title='Augmented Image')


# %%
len(image_wrapper_list)

# %%
for img_wrapper in image_wrapper_list:
    #print(img_wrapper)
    for bbox_wrapper in img_wrapper.objects:
       # print('  ', classes[bbox_wrapper.classid], bbox_wrapper)
        print(bbox_wrapper)
    break


# %%
import numpy as np
import PIL.Image
import tensorflow as tf

class VirusSequence(tf.keras.utils.Sequence):
    def __init__(self, image_wrapper_list, target_size, number_cells,
                 anchors, class_names, batch_size,
                 preprocess_images_function=None,
                 shuffle=False):
        """Keras dataloader
        
        Params:
            image_wrapper_list: list of ImageWrapper objects
            target_size (int): image size in pixels, e.g. 416
            number_cells (int): how many cells, e.g. 13
            anchors (list): list of anchors in format: [(w1,h1),(w2,h2),...]
            class_names (list): e.g. ['WBC', 'RBC', 'Platelets']
            batch_size (int): mini-batch size
            preprocess_images_function: funct. to normalize input image
            shuffle (bool): shuffle images
        """
        assert isinstance(image_wrapper_list, (list, tuple, np.ndarray))
        assert isinstance(target_size, int) and target_size > 0
        assert isinstance(number_cells, int) and number_cells > 0
        assert isinstance(anchors, (tuple, list))
        assert isinstance(anchors[0], (tuple, list)) and len(anchors[0]) == 2  # 2 = w,h
        assert isinstance(class_names, (tuple, list))
        assert isinstance(class_names[0], str)
        assert isinstance(batch_size, int) and batch_size > 0
        assert preprocess_images_function is None or callable(preprocess_images_function)
        assert isinstance(shuffle, bool)
        
        if target_size / number_cells != 16:
            raise ValueError('target_size and number_cells must be such that cell width is 16')
        
        self.cell_width = 16
        self.cell_height = 16
    
        self.image_wrapper_list = np.array(image_wrapper_list)  # for advanced indexing
        self.target_size = target_size        # int, e.g 416
        self.number_cells = number_cells      # int, e.g. 13
        self.anchors = anchors                # [[anchor_1_w, anchor_1_h], ...]
        self.class_names = class_names        # ['RBC', ...]
        self.class_ids = list(range(len(class_names)))  # [0, 1, ...]
        self.batch_size = batch_size
        self.preprocess_images_function = preprocess_images_function
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_wrapper_list) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Images format:
         - type: np.ndarray, dtype float
         - shape: (batch_size, target_size, target_size, 3)
        Targets format:
         - type: np.ndarray, dtype float
         - shape: (batch_size, nb_cells, nb_cells, nb_anchors, 5 + nb_classes)
         - where last dim is arranged as follows:
             [bbox_center_x, bbox_center_y, bbox_width, bbox_height, confidence, classes]
           + bbox_ params are expressed in cell_width/height as a unit
           + confidence is "objectness" from the paper
           + classes is a one-hot encoded object class
           + e.g. [1.5, 2.5, 2, 3, 1, 0, 0, 1]
                                      ^--^--^-- class
                                   ^----------- object present (if zero ignore other vals)    
                                ^------ bbox is 3x cells wide (32*3=96 pixels)
                             ^----- bbox is 2x cells tall
                          ^---- bbox center is in the 3rd row of cell grid
                    ^--- bbox center is in 2nd column of cell grid
         
        Returns:
            (images, targets): two np.ndarray with training mini-batch
        """
        batch_i = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_iw = self.image_wrapper_list[batch_i]    # [ImageWrapper, ...]
        
        images_shape = (
            len(batch_iw),      # batch size
            self.target_size,   # width, e.g 416
            self.target_size,   # height, e.g. 416
            3,                  # nb channels, RGB
        )
        
        targets_shape = (
            len(batch_iw),                  # batch size
            self.number_cells,              # nb cells x, 13
            self.number_cells,              # nb cells y, 13
            len(self.anchors),              # nb anchors
            4 + 1 + len(self.class_names))  # x,y,w,h, conf, clases
        
        images_arr = np.zeros(shape=images_shape, dtype=np.uint8)
        targets_arr = np.zeros(shape=targets_shape, dtype=float)
        
        for i, img_wrapper in enumerate(batch_iw):
            # Load Image
            image = PIL.Image.open(img_wrapper.filepath)
            image_w, image_h = image.size
            image_new = image.resize((self.target_size, self.target_size),
                                     resample=PIL.Image.LANCZOS)
            images_arr[i] = np.array(image_new)
            
            # Load Objects
            for obj_wrapper in img_wrapper.objects:
                if obj_wrapper.classid not in self.class_ids:
                    continue
                
                xmin, ymin = obj_wrapper.xmin, obj_wrapper.ymin  # unit: input img pixels
                xmax, ymax = obj_wrapper.xmax, obj_wrapper.ymax
                
                center_x_px = (xmin + xmax) / 2    # bounding box center
                center_y_px = (ymin + ymax) / 2    # unit: input img pixels, [0..image_h]
                size_w_px = (xmax - xmin)           # bounding box width & height
                size_h_px = (ymax - ymin)           # unit: input img pixels, [0..image_h]
                
                center_x_01 = (center_x_px / image_w)  # range: [0..1]
                center_y_01 = (center_y_px / image_h)
                size_w_01 = size_w_px / image_w
                size_h_01 = size_h_px / image_h
                
                center_x_cells = center_x_01 * self.number_cells  # range: [0..nb_cells]
                center_y_cells = center_y_01 * self.number_cells
                size_w_cells = size_w_01 * self.number_cells
                size_h_cells = size_h_01 * self.number_cells
                
                grid_x_loc = int(np.floor(center_x_cells))
                grid_y_loc = int(np.floor(center_y_cells))
                
                # Sanitize indices
                grid_x_loc = min(max(grid_x_loc, 0), self.number_cells - 1)
                grid_y_loc = min(max(grid_y_loc, 0), self.number_cells - 1)
                
                # Find highest IoU anchor
                best_anchor_loc, best_iou = 0, 0
                for anchor_loc, anchor_wh in enumerate(self.anchors):
                    (anchor_w_cells, anchor_h_cells) = anchor_wh
                    
                    intersect_w = min(size_w_cells, anchor_w_cells)
                    intersect_h = min(size_h_cells, anchor_h_cells)
                    intersect_area = intersect_w * intersect_h
                    union_w = max(size_w_cells, anchor_w_cells)
                    union_h = max(size_h_cells, anchor_h_cells)
                    union_area = union_w * union_h
                    
                    IoU = intersect_area / union_area
                    if IoU > best_iou:
                        best_iou = IoU
                        best_anchor_loc = anchor_loc
                
                # Ensure anchor index is within bounds
                best_anchor_loc = min(max(best_anchor_loc, 0), len(self.anchors) - 1)
                
                class_idx = obj_wrapper.classid
                target = np.zeros(shape=(4 + 1 + len(self.class_names)), dtype=float)
                target[0] = center_x_cells
                target[1] = center_y_cells
                target[2] = size_w_cells
                target[3] = size_h_cells
                target[4] = 1.0
                target[5 + class_idx] = 1.0
                
                targets_arr[i, grid_y_loc, grid_x_loc, best_anchor_loc] = target
        
        if self.preprocess_images_function is not None:
            images_arr = self.preprocess_images_function(images_arr)
        
        return images_arr, targets_arr
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_wrapper_list))
        if self.shuffle:
            np.random.shuffle(self.indices)


# %%
def preproc_images(images_arr):
    assert isinstance(images_arr, np.ndarray)
    assert images_arr.dtype == np.uint8
    return images_arr / 255


# %%
temp_generator = VirusSequence(image_wrapper_list, target_size=512, number_cells=32, 
                        anchors=anchors, class_names=classes, batch_size=16, shuffle=False,
                       preprocess_images_function=preproc_images)


# %%
x_batch, y_batch = temp_generator[0]
print('x_batch.shape:', x_batch.shape)
print('x_batch.dtype:', x_batch.dtype)
print('y_batch.shape:', y_batch.shape)
print('y_batch.dtype:', y_batch.dtype)

# %%
def decode_y_true(image_np, y_true_np):
    """
    Returns:
        [(x1, y1, x2, y2), ...] - where x1, y1, x2, y2 are coordinates of 
                                  top-left and bottom-right corners of bounding box
                                  in pixels in image, this can be passed to PIL.Draw
    """

    assert isinstance(image_np, np.ndarray)
    assert image_np.shape == (512, 512, 3)
    assert isinstance(y_true_np, np.ndarray)
    assert y_true_np.shape == (32, 32, 5, 6)

    img_h, img_w, _ = image_np.shape

    grid_w, grid_h, nb_anchors, _ = y_true_np.shape

    cell_w = img_w / grid_w  # 32
    cell_h = img_h / grid_h  # 32

    boundig_boxes = []


    for gx in range(grid_w):
        for gy in range(grid_h):
            for ai in range(nb_anchors):
                anchor = y_true_np[gx][gy][ai]
                
                classid = np.argmax(anchor[5:])
                
                if anchor.max() != 0:
                    bbox_center_x = cell_w * anchor[0]
                    bbox_center_y = cell_h * anchor[1]
                    bbox_width = cell_w * anchor[2]
                    bbox_height = cell_h * anchor[3]

                    x1 = bbox_center_x-bbox_width/2
                    y1 = bbox_center_y-bbox_height/2
                    x2 = bbox_center_x+bbox_width/2
                    y2 = bbox_center_y+bbox_height/2

                    boundig_boxes.append(BBoxWrapper(classid, 1.0, x1, y1, x2, y2))
                    
    return boundig_boxes

# %%
def plot_boxes(image_np, boundig_boxes, classes, colors1, width=1):
    image_h, image_w, _ = image_np.shape

    pil_img = PIL.Image.fromarray((image_np*255).astype(np.uint8))
    draw = PIL.ImageDraw.Draw(pil_img)
    i=1    
    for box in boundig_boxes:
        xmin, ymin = int(box.xmin), int(box.ymin)
        xmax, ymax = int(box.xmax), int(box.ymax)
        
        draw.rectangle([xmin, ymin, xmax, ymax], outline=colors[box.classid], width=width)
        draw.text([xmin+4,ymin+2],
                  f'{classes[box.classid]} {box.score:.2f} {i}', fill=colors[box.classid])
        i=i+1
       # print(box.classid)
        score =(box.score)
    
    del draw
    return pil_img 


# %%
idx = 6
boundig_boxes = decode_y_true(x_batch[idx], y_batch[idx])
pil_img = plot_boxes(x_batch[idx], boundig_boxes, classes, colors)
display(pil_img)


# %%
split = int(0.9*len(image_wrapper_list))
print('Train/Valid split:', split, '/', len(image_wrapper_list)-split)

# %%

train_generator = VirusSequence(image_wrapper_list[:split],
                                target_size=512, number_cells=32, 
                                anchors=anchors, class_names=classes,
                                batch_size=4, shuffle=False,
                                preprocess_images_function=preproc_images)
valid_generator = VirusSequence(image_wrapper_list[split:],
                                target_size=512, number_cells=32, 
                                anchors=anchors, class_names=classes,
                                batch_size=4, shuffle=False,
                                preprocess_images_function=preproc_images)


# %%
print(len(train_generator[3]))

# %%
from tensorflow.keras.layers import Input, Lambda, Reshape, concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.models import Model

# %%
class SpaceToDepth(tf.keras.layers.Layer):
    def __init__(self, block_size, **kwargs):
        super(SpaceToDepth, self).__init__(**kwargs)
        self.block_size = block_size
    
    def call(self, x):
        return tf.nn.space_to_depth(x, self.block_size)

# %%
def conv_block(X, filters, kernel_size, dilation_rate, suffix, max_pool=False):
    X = Conv2D(filters, kernel_size, strides=(1,1), padding='same',
               use_bias=False, name='conv'+suffix)(X)
    X = BatchNormalization(name='norm'+suffix)(X)
    X = LeakyReLU(alpha=0.1)(X)
    if max_pool:
        X = MaxPooling2D(pool_size=(2, 2))(X)
    return X

# %%
def switch(X,filters,max_pool=False):
    switch = Conv2D(filters, kernel_size=1, strides=(1,1), use_bias=True)(X)
    
    #switch.set_weights(1)
    #switch.bias.data.fill_(0)
    return switch

# %%
def create_yolov2(input_size, grid_size, number_anchors, number_classes):

    assert isinstance(input_size, (tuple, list)) and len(input_size) == 2
    assert isinstance(input_size[0], int) and input_size[0] > 0
    assert isinstance(input_size[1], int) and input_size[1] > 0
    assert isinstance(grid_size, (tuple, list)) and len(grid_size) == 2
    assert isinstance(grid_size[0], int) and grid_size[0] > 0
    assert isinstance(grid_size[1], int) and grid_size[1] > 0

    input_height, input_width = input_size
    grid_height, grid_width = grid_size
    


    IN = Input(shape=(input_height, input_width, 3))

    X1 = conv_block(IN, filters=32, kernel_size=(3,3),dilation_rate=1, suffix='_1', max_pool=True)
    #print(X1.shape)
    X2 = conv_block(IN, filters=32, kernel_size=(3,3),dilation_rate=3 ,suffix='_2',max_pool=True)
    #print(X2.shape)
    s= switch(X1, filters=32, max_pool=True)
    #print(s.shape)
    X =  s*X1+(1-s)*X2
    X1 = conv_block(X, filters=64, kernel_size=(3,3),dilation_rate=1 , suffix='_3', max_pool=True)
   # print(X.shape)
    X2 = conv_block(X, filters=64, kernel_size=(3,3),dilation_rate=3 ,suffix='_4',max_pool=True)
    #print(X2.shape)
    s=  switch(X2,filters=64,max_pool=True)
    X =  s*X1+(1-s)*X2
    X = conv_block(X, filters=128, kernel_size=(3,3),dilation_rate=1 , suffix='_5')
    X = conv_block(X, filters=64, kernel_size=(1,1),dilation_rate=1 , suffix='_6')
    X1 = conv_block(X, filters=128, kernel_size=(3,3),dilation_rate=1 , suffix='_7', max_pool=True)
    X2 = conv_block(X, filters=128, kernel_size=(3,3),dilation_rate=3, suffix='_8', max_pool=True)
    s=  switch(X2,filters=128,max_pool=True)
    X =  s*X1+(1-s)*X2
    X = conv_block(X, filters=256, kernel_size=(3,3),dilation_rate=1 , suffix='_9')
    X = conv_block(X, filters=128, kernel_size=(1,1),dilation_rate=1 , suffix='_10')
    X = conv_block(X, filters=256, kernel_size=(3,3),dilation_rate=1 , suffix='_11')
    X = conv_block(X, filters=512, kernel_size=(3,3),dilation_rate=1 , suffix='_12')
    X = conv_block(X, filters=256, kernel_size=(1,1),dilation_rate=1 , suffix='_13')
    X = conv_block(X, filters=512, kernel_size=(3,3),dilation_rate=1 , suffix='_14')
    X = conv_block(X, filters=256, kernel_size=(1,1),dilation_rate=1 , suffix='_15')
    X = conv_block(X, filters=512, kernel_size=(3,3),dilation_rate=1 , suffix='_16')

    SK = X  # skip connection

    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = conv_block(X, filters=1024, kernel_size=(3,3),dilation_rate=1 , suffix='_17')
    X = conv_block(X, filters=512, kernel_size=(1,1),dilation_rate=1 , suffix='_18')
    X = conv_block(X, filters=1024, kernel_size=(3,3),dilation_rate=1 , suffix='_19')
    X = conv_block(X, filters=512, kernel_size=(1,1),dilation_rate=1 , suffix='_20')
    X = conv_block(X, filters=1024, kernel_size=(3,3),dilation_rate=1 , suffix='_21')
    X = conv_block(X, filters=1024, kernel_size=(3,3),dilation_rate=1 , suffix='_22')
    X = conv_block(X, filters=1024, kernel_size=(3,3),dilation_rate=1 , suffix='_23')

    SK = conv_block(SK, filters=64, kernel_size=(1,1),dilation_rate=1 , suffix='_24')
    SK = SpaceToDepth(block_size=2)(SK)
    X = concatenate([SK, X])

    X = conv_block(X, filters=1024, kernel_size=(3,3),dilation_rate=1 , suffix='_25')

    X = Conv2D(filters=number_anchors * (4+1+number_classes),
               kernel_size=(1,1), strides=(1,1), padding='same', name='conv_26')(X)
    
    OUT = Reshape((grid_height, grid_width, number_anchors, 4+1+number_classes))(X)

    model1 = Model(IN, OUT)
    return model1

# %%
model1 = create_yolov2(input_size = (512, 512),  # (height, width)
                      grid_size = (32, 32),     # (height, width)
                      number_anchors = len(anchors),
                      number_classes = len(classes))

# %%
model1.summary()

# %%
logits = model1.predict(x_batch)

# %%
def load_yolov2_weights(model, filepath, last_layer='leave'):
    pointer = 4
    weights = np.fromfile(filepath, dtype='float32')

    for i in range(1, 26):

        #
        #   Norm layers 1..22
        #
        norm_layer = model.get_layer('norm_' + str(i))

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weights[pointer:pointer+size]; pointer += size;
        gamma = weights[pointer:pointer+size]; pointer += size;
        mean = weights[pointer:pointer+size]; pointer += size;
        var = weights[pointer:pointer+size]; pointer += size;

        norm_layer.set_weights([gamma, beta, mean, var])

        #
        #   Conv layers 1..22
        #
        conv_layer = model.get_layer('conv_' + str(i))

        size = np.prod(conv_layer.get_weights()[0].shape)
        kernel = weights[pointer:pointer+size]; pointer += size;
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])

    #
    #   Conv layer 23
    #
    if last_layer == 'leave':
        pass
    
    elif last_layer == 'load':
        conv_layer = model.get_layer('conv_26')

        size = np.prod(conv_layer.get_weights()[1].shape)
        bias   = weights[pointer:pointer+size]; pointer += size;

        size = np.prod(conv_layer.get_weights()[0].shape)
        kernel = weights[pointer:pointer+size]; pointer += size;
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])

        conv_layer.set_weights([kernel, bias])
        
    elif last_layer == 'rand':
        conv_layer = model.get_layer('conv_26') # the last convolutional layer
        weights = conv_layer.get_weights()
        
        output_shape = model.layers[-1].output_shape
        _, grid_w, grid_h, _, _ = output_shape
        
        new_kernel = np.random.normal(size=weights[0].shape) / (grid_w*grid_h)
        new_bias   = np.random.normal(size=weights[1].shape) / (grid_w*grid_h)

        conv_layer.set_weights([new_kernel, new_bias])
    
    else:
        raise ValueError("Parameter last_layer must be 'leave', 'load' or 'rand'.")



# %%
for layer in model1.layers:
    if layer.name == 'conv_20':
        layer.trainable = False

# %%
for layer in model1.layers:
    if layer.name == 'conv_21':
        layer.trainable = False

# %%
for layer in model1.layers:
    if layer.name == 'conv_22':
        layer.trainable = False

# %%
model1.summary()

# %%
#!./darknet detector test cfg/coco.data cfg/yolov2.cfg yolov2.weights data/dog.jpg

# %%
#!git clone https://github.com/pjreddie/darknet
#cd darknet
#!make

# %%
#!./darknet

# %%
#!wget https://pjreddie.com/media/files/yolov2.weights

# %%
np.random.seed(0)  # makes last layer deterministic for testing
load_yolov2_weights(model1, 'yolov2.weights', last_layer='rand')

# %%
logits = model1.predict(x_batch)

# %%
batch_idx = 0
cell_row = 1
cell_col = 8
print('rows are anochor boxes 0..4')
print('   -----------------   logits   -----------------')
print('    x     y     w     h    conf  cls0  cls1  cls2')
print(np.round(logits[batch_idx][cell_row][cell_col], 2))

# %%
def yolov2_loss_full(y_true, y_pred):
    """YOLOv2 loss. Note y_true, y_pred are tensors!"""
    global anchors, lambda_coord, lambda_noobj, lambda_obj, lambda_class, class_weights
    
    #
    #   Prepare empty masks
    #
    nb_batch = tf.shape(y_true)[0]
    nb_grid_w = tf.shape(y_true)[1]
    nb_grid_h = tf.shape(y_true)[2]
    nb_anchor = tf.shape(y_true)[3]
    nb_class = tf.shape(y_true)[4] - 5    # substract x,y,w,h,conf fields
    
    coord_mask = tf.zeros(shape=(nb_batch, nb_grid_w, nb_grid_h, nb_anchor))
    conf_mask  = tf.zeros(shape=(nb_batch, nb_grid_w, nb_grid_h, nb_anchor))
    class_mask = tf.zeros(shape=(nb_batch, nb_grid_w, nb_grid_h, nb_anchor))
    
    #
    #   Decode Predictions
    #
    
    # create grid_coord so we can offset predictions by grid cell index
    grid_tiles = tf.tile(tf.range(nb_grid_w), [nb_grid_h])
    grid_resh = tf.reshape(grid_tiles, (1, nb_grid_h, nb_grid_w, 1, 1))
    grid_x = tf.to_float(grid_resh)
    grid_y = tf.transpose(grid_x, (0,2,1,3,4))
    grid_concat = tf.concat([grid_x,grid_y], -1)
    grid_coord = tf.tile(grid_concat, [nb_batch, 1, 1, nb_anchor, 1])
    
    # transform logits to 0..1, then add cell offsets
    # shape [b, gw, gh, a, 2], range [0..gx, 0..gy], dtype float
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + grid_coord
       
    # transform logits to bounding box width and height
    # shape [b, gw, gh, a, 2], range [0..], dtype float
    # value of [1, 1] means bbox is same as anchor, [.5,.5] means half anchor
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(anchors, [1,1,1,len(anchors),2])
    
    # logits to confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
    # for class probabilites keep logits as (passed to cross-entropy later)
    pred_box_class = y_pred[..., 5:]
    
    #
    #   Decode targets
    #
    
    # target xywh are alredy correctly formatted
    # shape [b, gw, gh, a, 2], range [0..gx, 0..gy], dtype float
    # value [1.5, 2.5] means bbox center is aligned with center of cell [1, 2]
    true_box_xy = y_true[..., 0:2]
    true_box_wh = y_true[..., 2:4]
    
    # this whole section basically computes IoU(true_bbox, pred_bbox)
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    
    # target confidence is defined as: class_prob * IoU(true_bbox, pred_bbox)
    # note that in this dataset class_prob is always 0 or 1
    true_box_conf = iou_scores * y_true[..., 4]
    
    # convert class vector from one-hot to integer [0..]
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    
    #
    #   Compute 0/1 masks
    #
    
    # coordinate mask: the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * lambda_coord
    
    # confidence mask: the confidence of the ground truth boxes
    conf_mask = conf_mask + (1 - y_true[..., 4]) * lambda_noobj  # <- simplification #1
    conf_mask = conf_mask + y_true[..., 4] * lambda_obj
    
    # class mask: the class of the ground truth boxes
    class_mask = y_true[..., 4] * tf.gather(class_weights, true_box_class) * lambda_class       
    
    #
    #   Combine the loss
    #
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.

    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
    loss = loss_xy + loss_wh + loss_conf + loss_class
    
    return loss

# %%
temp_dataloader = VirusSequence(image_wrapper_list, target_size=512, number_cells=32, 
                        anchors=anchors, class_names=classes, batch_size=4, shuffle=False,
                        preprocess_images_function=preproc_images)

# %%
x_batch, y_batch = temp_dataloader[0]

# %%
y_hat = model1.predict(x_batch)

# %%
import tensorflow.compat.v1 as tf

# %%
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior

y_true_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 5, 6)) # Change dtype to float32
y_pred_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 5, 6)) # Change dtype to float32
loss = yolov2_loss_full(y_true_ph, y_pred_ph)


# %%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(loss, {y_true_ph: y_batch, y_pred_ph: y_hat})



# %%
print('result:', res)

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(loss=yolov2_loss_full, optimizer=optimizer, metrics=['accuracy'])

# %%
steps_per_epoch = len(train_generator)
validation_steps = len(valid_generator)


# %%
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'model_M_s_2.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')


# %%
hist = model1.fit_generator(generator=train_generator, 
                           steps_per_epoch=len(train_generator), 
                           epochs=100, 
                           validation_data=valid_generator,
                           validation_steps=len(valid_generator),
                          # callbacks=[checkpoint_cb],  # [earlystop_cb, checkpoint_cb]
                           max_queue_size=3)

# %%
tf.config.run_functions_eagerly(True)

# %%
plt.plot(hist.history['loss'], c='red')
plt.plot(hist.history['val_loss'], c='blue')
plt.ylim((0,1))

plt.legend(["Training loss", "Validation loss"], loc ="upper right")

# %%
def sigmoid(x):
    return 1 / (1+np.exp(-x))

# %%
def softmax(x):
    max_ = np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x - max_)
    ex_sum = np.sum(ex, axis=-1, keepdims=True)
    return ex / ex_sum

# %%
def iou(bboxA, bboxB):
    # intersect rectangle
    xmin = max(bboxA.xmin, bboxB.xmin)
    ymin = max(bboxA.ymin, bboxB.ymin)
    xmax = min(bboxA.xmax, bboxB.xmax)
    ymax = min(bboxA.ymax, bboxB.ymax)
    
    areaI = max(0, xmax-xmin) * max(0, ymax-ymin)
    areaA = (bboxA.xmax-bboxA.xmin) * (bboxA.ymax-bboxA.ymin)
    areaB = (bboxB.xmax-bboxB.xmin) * (bboxB.ymax-bboxB.ymin)
    
    IoU = areaI / (areaA + areaB - areaI)
    return IoU

# %%
def decode_y_pred(logits, img_shape, classes, anchors, obj_threshold=0.3):
    nb_grid_h, nb_grid_w, nb_anchor, _ = logits.shape
    nb_class = logits.shape[3] - 5  # remove x,y,w,h,conf.
    assert nb_class == len(classes)
    
    proc_logits = lambda xywh: (sigmoid(xywh[0]), sigmoid(xywh[1]),
                                np.exp(xywh[2]), np.exp(xywh[3]))
    
    logits_xywh = logits[...,:4]                        # shape (13, 13, 5, 4)
    bbox_xywh = np.apply_along_axis(                    # shape (13, 13, 5, 4)
        func1d=proc_logits, axis=-1, arr=logits_xywh)
    
    # all have shape: [grid_h, grid_w, nb_anchor, nb_class]
    confidences = np.expand_dims(sigmoid(logits[..., 4]),-1)
    probabilities = softmax(logits[..., 5:])
    scores = confidences * probabilities
    
    boxes = []
   
    for ri in range(nb_grid_h):
        for ci in range(nb_grid_w):
            for ai in range(nb_anchor):
                best_class_score = scores[ri, ci, ai].max()
                if best_class_score > obj_threshold:
                    
                    classid = np.argmax(scores[ri, ci, ai])     # int
                    img_h, img_w, _ = img_shape
                    
                    x, y, w, h = bbox_xywh[ri, ci, ai]
                    x = (ci + x) / nb_grid_w * img_w            # unit image [0..1]
                    y = (ri + y) / nb_grid_h * img_h            # unit image [0..1]
                    w = anchors[ai][0] * w / nb_grid_w * img_w  # unit image [0..1]
                    h = anchors[ai][1] * h / nb_grid_h * img_h  # unit image [0..1]
                    
                    box = BBoxWrapper(classid, best_class_score,
                                      xmin=x-w/2, ymin=y-h/2,
                                      xmax=x+w/2, ymax=y+h/2)
                    boxes.append(box)
                    #print(box)
    return boxes

# %%
def suppress(boxes, classes, nms_threshold):
    
    suppressed = []

    for classid in range(len(classes)):
        
        indices = np.argsort([box.score for box in boxes
                             if box.classid == classid])
        indices = indices[::-1]  # reverse
                
        for i in range(len(indices)):
            index_i = indices[i]
            
            if boxes[index_i] in suppressed: 
                continue
            
            for j in range(i+1, len(indices)):
                index_j = indices[j]


                iou2 = iou(boxes[index_i], boxes[index_j])
                if iou2 >= nms_threshold:
                    #boxes[index_j].classes[c] = 0
                    suppressed.append(boxes[index_j])

    boxes = [box for box in boxes if box not in suppressed]
    
    return boxes 

# %%
x_batch, y_batch = valid_generator[0]
image_np = x_batch[1]



# %%
y_hat = model1.predict(np.expand_dims(image_np, 0))[0]

# %%
y_hat=[]
for i in range(len(x_batch)):
    y_hat.append(model1.predict(np.expand_dims(x_batch[i], 0))[0])
    

# %%
boxes1 = decode_y_pred(y_hat[1], image_np.shape, classes=classes,
                obj_threshold=0.8, anchors=anchors)
boxes1 = suppress(boxes1, classes=classes, nms_threshold=0.03)


# %%
boxes1=[]
for i in range(len(x_batch)):
    y_hat=model1.predict(np.expand_dims(x_batch[i], 0))[0]
    boxes = decode_y_pred(y_hat, x_batch[i].shape, classes=classes,
                obj_threshold=0.08, anchors=anchors)
    boxes1.append(suppress(boxes, classes=classes, nms_threshold=0.25))
print(len(boxes1[1]))

# %%
image = plot_boxes(x_batch[2], boxes1[2], classes, colors, width=1)
display(image)
print(len(boxes1[2]))

# %%
idx = 1
boundig_boxes = decode_y_true(x_batch[idx], y_batch[idx])
pil_img = plot_boxes(x_batch[idx], boundig_boxes, classes, colors)
display(pil_img)


# %%
boundig_boxes = []
for i in range(0,4):
    boundig_boxes.append(decode_y_true(x_batch[i], y_batch[i]))
print(len(boundig_boxes[1]))

# %%
image = plot_boxes(x_batch[2], boxes1[2], classes, colors, width=1)
display(image)
print(len(boundig_boxes[1]))

# %%
#true_boxes=[]
#for j in range(0,10):
true_boxes=[]
j=0
for i in range(len(boundig_boxes[0])):
    #for j in range(0,10):
    true_boxes.append([boundig_boxes[j][i].classid, boundig_boxes[j][i].score, boundig_boxes[j][i].xmin, boundig_boxes[j][i].ymin,boundig_boxes[j][i].xmax,boundig_boxes[j][i].ymax])
       # true_boxes[i][j].insert(0,J+1)
j=1    
for i in range(len(boundig_boxes[1])):
    #for j in range(0,10):
        true_boxes.append([boundig_boxes[j][i].classid, boundig_boxes[j][i].score, boundig_boxes[j][i].xmin, boundig_boxes[j][i].ymin,boundig_boxes[j][i].xmax,boundig_boxes[j][i].ymax])
       # true_boxes[i][j].insert(0,J+1)  
j=2   
for i in range(len(boundig_boxes[j])):
    #for j in range(0,10):
    true_boxes.append([boundig_boxes[j][i].classid, boundig_boxes[j][i].score, boundig_boxes[j][i].xmin, boundig_boxes[j][i].ymin,boundig_boxes[j][i].xmax,boundig_boxes[j][i].ymax])
       # true_boxes[i][j].insert(0,J+1)
j=3   
for i in range(len(boundig_boxes[j])):
    #for j in range(0,10):
    true_boxes.append([boundig_boxes[j][i].classid, boundig_boxes[j][i].score, boundig_boxes[j][i].xmin, boundig_boxes[j][i].ymin,boundig_boxes[j][i].xmax,boundig_boxes[j][i].ymax])
#j=4   
#for i in range(len(boundig_boxes[j])):
    #for j in range(0,10):
 #   true_boxes.append([boundig_boxes[j][i].classid, boundig_boxes[j][i].score, boundig_boxes[j][i].xmin, boundig_boxes[j][i].ymin,boundig_boxes[j][i].xmax,boundig_boxes[j][i].ymax])
       # true_boxes[i][j].insert(0,J+1)

print(len(true_boxes)  ) 

# %%
#for j in range(0,10):
pred_boxes=[]
#for j in range(0,10):
j=0
for i in range(len(boxes1[j])):
    #for j in range(0,10):
    pred_boxes.append([boxes1[j][i].classid, boxes1[j][i].score, boxes1[j][i].xmin, boxes1[j][i].ymin, boxes1[j][i].xmax, boxes1[j][i].ymax])
      
j=1
for i in range(len(boxes1[j])):
    #for j in range(0,10):
    pred_boxes.append([boxes1[j][i].classid, boxes1[j][i].score, boxes1[j][i].xmin, boxes1[j][i].ymin, boxes1[j][i].xmax, boxes1[j][i].ymax])
      
j=2
for i in range(len(boxes1[j])):
    #for j in range(0,10):
    pred_boxes.append([boxes1[j][i].classid, boxes1[j][i].score, boxes1[j][i].xmin, boxes1[j][i].ymin, boxes1[j][i].xmax, boxes1[j][i].ymax])
j=3
for i in range(len(boxes1[j])):
    #for j in range(0,10):
    pred_boxes.append([boxes1[j][i].classid, boxes1[j][i].score, boxes1[j][i].xmin, boxes1[j][i].ymin, boxes1[j][i].xmax, boxes1[j][i].ymax])
#j=4
#for i in range(len(boxes1[j])):
    #for j in range(0,10):
 #   pred_boxes.append([boxes1[j][i].classid, boxes1[j][i].score, boxes1[j][i].xmin, boxes1[j][i].ymin, boxes1[j][i].xmax, boxes1[j][i].ymax])

print((len(pred_boxes)))
#(pred_boxes)


# %%
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



# %%
from collections import Counter
import torch
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.25, box_format="corners", num_classes=1):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(0,num_classes+1):
        detections = []
        ground_truths = []
        #print(c)

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[0] == c:
                detections.append(detection)
        #print(detections)       

        for true_box in true_boxes:
            if true_box[0] == c:
                ground_truths.append(true_box)
        #print(detections)    

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
       

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[1], reverse=True)
        
        TP = torch.zeros((len(detections)))
       
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
       
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            #print(ground_truth_img )
            num_gts = len(ground_truth_img)
           # print(num_gts)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[2:]),
                    torch.tensor(gt[2:]),
                    box_format=box_format,
                )
               # print(iou)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        #print(recalls)
        precisions = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        one_tensor = torch.tensor([1.0], dtype=torch.float32)  # Float tensor
        zero_tensor = torch.tensor([0.0], dtype=torch.float32)  # Float tensor

        print(precisions)
        precisions = torch.cat((one_tensor, precisions))
        print(precisions)
        recalls = torch.cat((zero_tensor, recalls))
        #torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
        print( average_precisions)
        
    return sum(average_precisions) / len(average_precisions),recalls,precisions
    


# %%
mAP,R,P =mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.25, box_format="corners", num_classes=1)