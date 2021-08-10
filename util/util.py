import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
import glob
from openslide import open_slide, __library_version__ as openslide_version
import os
from PIL import Image
from skimage.color import rgb2gray
from scipy.ndimage import morphology
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

DATASET_DIR = '../gdrive/MyDrive/projectdata/'
DATASET_DIR2 = '../gdrive/MyDrive/projectdata/dataset2/'



def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im



def find_tissue_pixels(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return np.copy(list(zip(indices[0], indices[1])))

def find_tumor_pixels(image):
    indices = np.where(image != 0)
    return np.copy(list(zip(indices[0], indices[1])))


def find_combined_pixels(A,B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))
    D = np.setdiff1d(A.view(dtype),C.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)
    D = D.view(A.dtype).reshape(-1, ncols)
    return C, D


def apply_mask(im, mask, color=(255,0,0)):
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked




def generate_level_image(level, tumor_mask, slide, window_size, base_level = 0,random_num=1001):
    slide_image = read_slide(slide,
                         x=0,
                         y=0,
                         level=level,
                         width=slide.level_dimensions[level][0],
                         height=slide.level_dimensions[level][1])
    mask_image = read_slide(tumor_mask,
                        x=0,
                        y=0,
                        level=level,
                        width=slide.level_dimensions[level][0],
                        height=slide.level_dimensions[level][1])[:,:,0]
    tumor_pixels =  find_tumor_pixels(mask_image)
    tissue_pixels = find_tissue_pixels(slide_image)
    combined_pixels,D = find_combined_pixels(tissue_pixels,tumor_pixels)
    id_y,id_x = combined_pixels[random_num]

    zoom = 2**level
    margin = window_size//2
    x = id_x*zoom - margin
    y = id_y*zoom -margin
    image = read_slide(slide,x,y,base_level,window_size,window_size)
    return image


def make_directory_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def data_allocation(slide_path, save_dir, id_x, id_y, level, base_level,window_size):
    slide = open_slide(slide_path)
    zoom = 2**level
    margin = window_size//2
    x = id_x*zoom - margin
    y = id_y*zoom -margin
    im = slide.read_region((x, y), base_level, (window_size, window_size)).convert('RGB')
    save_dir = save_dir + '_{0}_{1}.png'.format(x, y)
    im.save(save_dir, compress_level=0)
    return


def prepare_normal_tissue_data(level=0, sample_size=50, window_size=256,range_size=22,base_levels=[0]):
    for no in range(1,range_size):
      image_no = '{:03}'.format(no)
      slide_path = DATASET_DIR+'tumor_'+image_no+".tif"
      tumor_path = DATASET_DIR+'tumor_'+image_no+"_mask"+".tif"
      slide = open_slide(slide_path)
      tumor_mask_slide = open_slide(tumor_path)


      slide_image = read_slide(slide,
                              x=0,
                              y=0,
                              level=level,
                              width=slide.level_dimensions[level][0],
                              height=slide.level_dimensions[level][1])
      tumor_mask_image = read_slide(tumor_mask_slide,
                              x=0,
                              y=0,
                              level=level,
                              width=tumor_mask_slide.level_dimensions[level][0],
                              height=tumor_mask_slide.level_dimensions[level][1])
      tissue_pixels = find_tissue_pixels(slide_image)
      tumor_pixels = find_tumor_pixels(tumor_mask_image)
      combined_pixels,difference_pixels = find_combined_pixels(tissue_pixels,tumor_pixels)
      selected_idx = difference_pixels[np.random.choice(difference_pixels.shape[0],sample_size,replace=True)]


      for id_y, id_x in selected_idx:
          for base_level in base_levels:
            NORMAL_PATCH_DIR = DATASET_DIR2 + 'normal_level{0}/'.format(base_level)
            save_dir = NORMAL_PATCH_DIR  + '{0}_{1}'.format(0, image_no)
            data_allocation(slide_path, save_dir, id_x, id_y, level, base_level, window_size)





def prepare_tumor_tissue_data(level=0, sample_size=50, window_size=256,range_size=22,base_levels=[0]):
    for no in range(1,range_size):
      image_no = '{:03}'.format(no)
      slide_path = DATASET_DIR+'tumor_'+image_no+".tif"
      tumor_path = DATASET_DIR+'tumor_'+image_no+"_mask"+".tif"
      slide = open_slide(slide_path)
      tumor_mask_slide = open_slide(tumor_path)
      tumor_mask_image = read_slide(tumor_mask_slide,
                              x=0,
                              y=0,
                              level=level,
                              width=tumor_mask_slide.level_dimensions[level][0],
                              height=tumor_mask_slide.level_dimensions[level][1])
      tumor_pixels = find_tumor_pixels(tumor_mask_image)
      selected_idx = tumor_pixels[np.random.choice(tumor_pixels.shape[0],sample_size,replace=True)]

      for id_y, id_x in selected_idx:
        for base_level in base_levels:
          TUMOR_PATCH_DIR = DATASET_DIR2 + 'tumor_level{0}/'.format(base_level)
          save_dir = TUMOR_PATCH_DIR + '{0}_{1}'.format(1, image_no)
          data_allocation(slide_path, save_dir, id_x, id_y, level,base_level, window_size)


def build_tf_data(normal_list,tumor_list):
    balance_factor = int(len(normal_list)/len(tumor_list))
    if balance_factor == 0:
      balance_factor = 1
    list_size = len(tumor_list) * balance_factor
    path_list = normal_list[:list_size] + tumor_list * balance_factor
    labels = [0] * list_size + [1] * len(tumor_list) * balance_factor
    combined_path_list = list(zip(path_list, labels))
    random.seed(51)
    random.shuffle(combined_path_list)
    path_list, labels = zip(*combined_path_list)
    return path_list, labels



def vis_sample(y_ds_train0,X_ds_train0):
    plt.figure(figsize=(10, 20), dpi=150)
    for n, (label, image) in enumerate(zip(y_ds_train0.take(8), X_ds_train0.take(8))):
        plt.subplot(8, 4, n+1)
        plt.imshow(image.numpy()*0.5+0.5)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def vis_patch(ds_train):
    for x in ds_train.take(1):
      images, labels = x
      plt.figure(dpi=150, figsize=(5, 2))
      for k in range(4):
          plt.subplot(1, 4, k+1)
          plt.imshow(images[0][k].numpy()*0.5+0.5)
          plt.title(['normal', 'tumor'][labels[k].numpy()])
      plt.tight_layout()



def display_right_cc(ds_test, model):
    image_batch,label_batch = next(iter(ds_test))
    y_pred_show = model.predict(image_batch, batch_size=256, verbose=1).ravel()
    image_batch0, image_batch1,image_batch2,image_batch3 = image_batch
    indices_right_prediction = np.where((y_pred_show.ravel() > 0.5) == label_batch.numpy())[0]
    y_pred_classes = y_pred_show.ravel() > 0.5
    plot_prediction(image_batch1.numpy(), label_batch.numpy(),y_pred_classes, indices_right_prediction[:3])



def display_wrong_cc(ds_test, model):
    image_batch,label_batch = next(iter(ds_test))
    y_pred_show = model.predict(image_batch, batch_size=256, verbose=1).ravel()
    image_batch0, image_batch1,image_batch2,image_batch3 = image_batch
    indices_wrong_prediction = np.where((y_pred_show.ravel() > 0.5) != label_batch.numpy())[0]
    y_pred_classes = y_pred_show.ravel() > 0.5
    plot_prediction(image_batch1.numpy(), label_batch.numpy(),y_pred_classes, indices_wrong_prediction[:3])



#https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
def _parse_function(path):
    image = tf.io.read_file(path)
    image_decoded = tf.image.decode_png(image, channels=3)
    image = tf.cast(image_decoded, tf.uint8)
    image = tf.reshape(image, [256, 256, 3])
    return image


def read_dataset(x):
    image = tf.io.parse_tensor(x, out_type=tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [256, 256, 3])
    image = image/127.5-1
    return image



def augment(x):
    image = tf.io.parse_tensor(x, out_type=tf.uint8)
    image = tf.reshape(image, [256, 256, 3])
    image = tf.image.rot90(image, random.randint(0, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.92, 1.08)
    image = tf.image.random_contrast(image, 0.92, 1.08)
    image = tf.image.random_brightness(image, 0.08)
    image = tf.cast(image, tf.float32)
    image = image/127.5-1
    return image





def plot_train(history):
    """
    visualize the training history
    """
    plt.figure(dpi=300, figsize=(5, 4))
    ax1 = plt.subplot(1, 1, 1)
    color = 'darkblue'
    ax1.plot(history['loss'], c=color, label='training loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('training loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'darkred'
    ax2.plot(history['acc'], c=color, label='training accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('training accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Training loss and accuracy')
    ax1.grid(True)
    plt.show()



def plot_confusion_matrix(y1,y2, classes,
                          normalize=False,
                          title_prefix="",
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y1, y2)
    print("confusion_raw:\n",cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = title_prefix + "confusion matrix (Normalized)"
    else:
        title = title_prefix + 'Confusion matrix (raw)'
    plt.figure(figsize=(3, 2), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=0)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout()
    plt.gca()
    plt.show()
    return



def plot_prediction(images, labels, predicted_classes, indices):
    num_plots = len(indices)
    n_cols = 3
    n_rows = int(np.ceil(num_plots/n_cols))
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=300)
    ax = ax.ravel()
    k = 0
    for idx, image in zip(indices, images[indices]):
        ax[k].imshow(image*0.5+0.5)
        label = labels[idx]
        pred = predicted_classes[idx]
        color = 'r' if label != pred else 'k'
        ax[k].set_title("label: {0}, pred = {1}".format(
            label, pred), color=color)
        ax[k].grid(False)
        k += 1
    # remove extra axes
    for j in range(k, n_cols*n_rows):
        fig.delaxes(ax[j])
    plt.tight_layout()






def prepare_test(slide_path, tissue_pixels, level, shift_x,shift_y):
    slide = open_slide(slide_path)
    window_size = 256
    base_levels = [0,1,2,3]
    i = 0
    x_test0 = np.empty((tissue_pixels.shape[0],256,256,3), np.float32)
    for id_y, id_x in tissue_pixels:
        zoom = 2**level
        margin = window_size//2
        x = (shift_x + id_x)*zoom - margin
        y = (shift_y + id_y)*zoom -margin
        im = read_slide(slide,x,y,0,window_size,window_size)/127.5-1
        x_test0[i] = im
        i += 1
    i = 0
    x_test1 = np.empty((tissue_pixels.shape[0],256,256,3), np.float32)
    for id_y, id_x in tissue_pixels:
        zoom = 2**level
        margin = window_size//2
        x = (shift_x + id_x)*zoom - margin
        y = (shift_y + id_y)*zoom -margin
        im = read_slide(slide,x,y,1,window_size,window_size)/127.5-1
        x_test1[i] = im
        i += 1
    i = 0
    x_test2 = np.empty((tissue_pixels.shape[0],256,256,3), np.float32)
    for id_y, id_x in tissue_pixels:
        zoom = 2**level
        margin = window_size//2
        x = (shift_x + id_x)*zoom - margin
        y = (shift_y + id_y)*zoom -margin
        im = read_slide(slide,x,y,2,window_size,window_size)/127.5-1
        x_test2[i] = im
        i += 1
    i = 0
    x_test3 = np.empty((tissue_pixels.shape[0],256,256,3), np.float32)
    for id_y, id_x in tissue_pixels:
        zoom = 2**level
        margin = window_size//2
        x = (shift_x + id_x)*zoom - margin
        y = (shift_y + id_y)*zoom -margin
        im = read_slide(slide,x,y,3,window_size,window_size)/127.5-1
        x_test3[i] = im
        i += 1
    return x_test0, x_test1, x_test2, x_test3




def tissue_mask_bool(tissue_pixels,image):
    bool_arr = np.zeros(image.shape,dtype=bool)
    for x,y in tissue_pixels:
        bool_arr[x][y] = True
    return bool_arr



def plot_heat_map(y_pred_test,tissue_pixels,tumor_mask_image,slide_image):
  cut_off_probability = 0.5
  bool_arr = tissue_mask_bool(tissue_pixels,tumor_mask_image)
  tumor_mask_pred_prob = np.zeros(tumor_mask_image.shape, dtype=np.float32)
  tumor_mask_pred_prob[bool_arr] = y_pred_test.ravel()
  tumor_mask_pred = tumor_mask_pred_prob > cut_off_probability
  plt.figure(dpi=300, figsize=(10, 8))
  plt.imshow(slide_image)
  plt.imshow(tumor_mask_pred_prob, cmap='gist_heat',
            vmin=cut_off_probability, vmax=1, alpha=0.9)

  plt.colorbar(aspect=40,orientation='vertical')
  plt.tight_layout()
  plt.show()
  return tumor_mask_pred




def show_performance(tumor_mask_pred, tumor_mask_image, tissue_mask, slide_image):
    figsize = (5, 5)
    dpi = 300
    mask_visualization = np.zeros(
        (tissue_mask.shape[0], tissue_mask.shape[1], 4), dtype=float)
    mask_visualization[:, :, 1] = np.logical_and(
        tumor_mask_pred, tumor_mask_image)

    mask_visualization[:, :, 2] = np.logical_and(
        tumor_mask_pred, tumor_mask_image == False)

    mask_visualization[:, :, 0] = np.logical_and(
        tumor_mask_pred == False, tumor_mask_image)
    mask_visualization[:, :, 3] = True

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(np.zeros_like(tissue_mask), cmap='inferno', vmin=0, vmax=1)
    plt.imshow(slide_image)
    plt.imshow(mask_visualization, alpha=0.8, vmin=0, vmax=1)
    red_patch = mpatches.Patch(color='red', label='False negative')
    blue_patch = mpatches.Patch(color='green', label='True positive')
    yellow_patch = mpatches.Patch(color='blue', label='False positive')
    plt.legend(handles=[red_patch, blue_patch, yellow_patch], loc=1)
    plt.grid(False)
    plt.show()




def find_tissue_pixels2(image,intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    tissue_mask = im_gray <= intensity
    indices = np.where(im_gray <= intensity)
    return np.copy(list(zip(indices[0], indices[1]))),tissue_mask



def find_y_true(tissue_pixels,tumor_pixels):
  y_true = []
  for x,y in tissue_pixels:
    if np.array([x,y]) in tumor_pixels:
        y_true.append(1)
    else:
        y_true.append(0)
  return y_true
