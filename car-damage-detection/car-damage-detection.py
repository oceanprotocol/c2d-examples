import io
import os
import json
import zipfile
import shutil
from datetime import datetime
from urllib.parse import urlparse, unquote

import cv2
import h5py
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import Input, Model
from keras.optimizers import *
from keras.metrics import *
from keras.models import load_model, model_from_json, save_model
from keras.callbacks import ModelCheckpoint, CSVLogger
# from keras.src.applications.resnet import ResNet50
from keras_unet_collection.models import unet_2d as unet_2d
import requests

datasets_urls = [
    "*" # Define URLs here
]

models_urls = [
    "*" # Define URLs here
]

tf.test.is_built_with_cuda()
# SAVE_RESULTS_DIR = "/data/outputs/"
SAVE_RESULTS_DIR = "./results"
SAVE_PREDICTIONS_DIR = "/data/outputs/" # here will be /data/outputs
# BACKBONE = 'DenseNet201'
BACKBONE = "ResNet50V2"
LEARNING_RATE = 1e-3
INPUT_SHAPE = (512, 512, 3) # RGB image
NUM_CLASSES = 2 # background and damage

def download_datasets():
    for url in datasets_urls:
        print(url)
        response = requests.get(url)
        print(response.status_code)
        if response.status_code == 200:
            zip_filename = os.path.basename(urlparse(url).path)
            decoded_name = unquote(zip_filename)
            folder_name = os.path.splitext(decoded_name)[0]
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                names = zip_ref.namelist()

                if len(names) == 1 or str(decoded_name.strip('.zip') + '/') == names[0]:
                    # ZIP already has a clean top-level folder — extract to current dir
                    zip_ref.extractall(os.getcwd())
                    print(f"Extracted to: {os.path.join(os.getcwd(), list(names)[0])}")
                else:
                    # ZIP doesn't have a single top-level folder — create one
                    target_dir = os.path.join(os.getcwd(), folder_name)
                    os.makedirs(target_dir, exist_ok=True)
                    zip_ref.extractall(target_dir)
                    print(f"Extracted into new folder: {target_dir}")

def download_models():
    response = requests.get(models_urls[0], stream=True) # for the moment test c2d with one model
    if response.status_code == 200:
        with open("kaggle_model.h5", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded kaggle_model.h5 from '{models_urls[0]}'.")


def generator_v2(*dataset_dirs, batch_size, augmentation=None, **kwargs):
    assert batch_size > 0, "The batch size must be greater than 0"

    for dataset_dir in dataset_dirs:
        assert os.path.exists(dataset_dir), f"The dataset directory {dataset_dir} does not exist"

    # get std and mean for each channel from kwargs
    std = kwargs.get('std', None)
    mean = kwargs.get('mean', None)
    resize = kwargs.get('resize', (512, 512))

    # get the list of all images
    images_paths = []
    print(f"Dataset dirs: {dataset_dirs}")
    for dataset_dir in dataset_dirs:
        print(f"Dataset dir: {dataset_dir}")
        current_images_dir = os.path.join(dataset_dir, 'images')
        print(f"Current image dir: {current_images_dir}")
        current_dir_images_filenames = os.listdir(current_images_dir)
        print(f"Current image dir filenames: {current_dir_images_filenames}")
        current_dir_images_paths = [os.path.join(current_images_dir, f) for f in current_dir_images_filenames]
        current_dir_images_paths = list(filter(lambda x: x.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')), current_dir_images_paths))
        print(f"Current image dir paths: {current_dir_images_paths}")
        images_paths += current_dir_images_paths

    def to_categorical(mask, num_classes=2):
        identity_matrix = np.eye(num_classes, dtype=np.uint8)
        assert np.unique(mask).all() < num_classes, f"mask values must be less than {num_classes}"
        return identity_matrix[mask]

    images = []
    masks = []
    print(f"Generating {len(images_paths)} images...")
    print(f"Images paths: {images_paths}")
    # while True:
    for image_path in images_paths:
        assert os.path.exists(image_path), f"The image path: {image_path} does not exist"

        mask_path = image_path.replace('images', 'masks').rsplit('.', 1)[0] + '.png'
        print(f"Generating gmask paths {mask_path}...")

        assert os.path.exists(mask_path), f"The mask path: {mask_path} does not exist"

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if augmentation is not None:
            sample = augmentation(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask[mask > 0] = 1

        mask = to_categorical(mask)
        mask = np.asarray(mask, dtype=np.float32)

        # resize the image and the mask
        image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)

        images.append(image)
        masks.append(mask)

        if len(images) == batch_size:

            if std is not None and mean is not None:
                images = np.asarray(images, dtype=np.float32)
                images = (images - mean) / std

            images = np.asarray(images, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.float32)

            yield images, masks

            images = []
            masks = []

def custom_focal_tversky(num_classes, alpha=0.5, gamma=4/3, const=K.epsilon()):
    def custom_tversky_coef(y_true, y_pred):
        # Flattening the tensors to ensure the calculations are performed for each pixel
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # True Positives, False Positives & False Negatives
        tp = tf.reduce_sum(y_true_flat * y_pred_flat)
        fp = tf.reduce_sum((1-y_true_flat) * y_pred_flat)
        fn = tf.reduce_sum(y_true_flat * (1-y_pred_flat))

        tversky = (tp + const) / (tp + alpha*fp + (1-alpha)*fn + const)

        return tversky

    def multi_class_focal_tversky_loss(y_true, y_pred):
        # tf tensor casting
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        loss = 0
        for c in range(num_classes):
            y_true_c = y_true[..., c]
            y_pred_c = y_pred[..., c]

            # squeeze-out length-1 dimensions.
            y_pred_c = tf.squeeze(y_pred_c)
            y_true_c = tf.squeeze(y_true_c)

            # (Tversky loss)**(1/gamma)
            loss += tf.math.pow((1 - custom_tversky_coef(y_true_c, y_pred_c)), 1/gamma)

        return loss / num_classes

    return multi_class_focal_tversky_loss

def get_checkpoints(save_name, save_format):
    models_folder = os.path.join(SAVE_RESULTS_DIR, 'models')
    if not os.path.exists(models_folder):
      os.makedirs(models_folder)

    scores_folder = os.path.join(SAVE_RESULTS_DIR, 'scores')
    if not os.path.exists(scores_folder):
      os.makedirs(scores_folder)

    saved_model_path = os.path.join(models_folder, f"{save_name}.{save_format}")
    saved_scores_path = os.path.join(scores_folder, f"{save_name}.csv")
    print(f"Saving model to {saved_model_path}...")
    print(f"Saving scores to {saved_scores_path}...")

    save_last_checkpoint = ModelCheckpoint(saved_model_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto', save_freq="epoch")
    print(f"Saving last checkpoint to {save_last_checkpoint}...")
    csv_logger = CSVLogger(saved_scores_path, append=True)
    print(f"Saving csv_logger to {csv_logger}...")

    return [save_last_checkpoint, csv_logger]

def init_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, backbone=BACKBONE):
    # define the model
    model = unet_2d(input_shape, [64, 128, 256, 512, 1024], n_labels=num_classes,
                          stack_num_down=2, stack_num_up=2,
                          activation='ReLU', output_activation='Softmax',
                          batch_norm=True, pool='max', unpool='bilinear', backbone=backbone, weights='imagenet', name='unet_large')
    # optimizer and loss
    optimizer = Adam(learning_rate=learning_rate)
    loss = custom_focal_tversky(num_classes=num_classes)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', OneHotMeanIoU(num_classes=num_classes)])
    return model

def zip_all(parent_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            # Create ZIP file in temporary location (same as parent_dir)
            zip_filename = f"{item}.zip"
            temp_zip_path = os.path.join(parent_dir, zip_filename)

            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, item_path)
                        zipf.write(abs_path, os.path.join(item, rel_path))

            # Move the zip file to the destination directory
            final_path = os.path.join(destination_dir, zip_filename)
            shutil.move(temp_zip_path, final_path)
            print(f"Moved {zip_filename} → {final_path}")


def test(model_path, test_dir, num_classes, num_images=None, max_num_images_per_line=4, std=None, mean=None):
    model = load_model(model_path,
        custom_objects={'multi_class_focal_tversky_loss': custom_focal_tversky(num_classes=num_classes)
        }, compile=False)
    print(f"Finished loading model: {model}")
    images_paths = [os.path.join(test_dir, image_filename) for image_filename in os.listdir(test_dir)]
    num_images = num_images if num_images is not None and num_images < len(images_paths) else len(images_paths)
    images_paths = images_paths[:num_images]

    resize = (model.input_shape[1], model.input_shape[2])

    # Ensure output dir exists
    output_dir = SAVE_RESULTS_DIR + '/pred'
    os.makedirs(output_dir, exist_ok=True)

    # Define a color map (can be expanded for more classes)
    colormap = np.array([
        [0, 0, 0],        # Class 0 - black
        [0, 255, 0],      # Class 1 - green
        [0, 0, 255],      # Class 2 - blue
        [255, 0, 0],      # Class 3 - red
        [255, 255, 0],    # Class 4 - yellow
        [255, 0, 255],    # Class 5 - magenta
        [0, 255, 255]     # Class 6 - cyan
    ], dtype=np.uint8)

    for idx, image_path in enumerate(images_paths):
        if not image_path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue

        assert os.path.exists(image_path), f"The image path: {image_path} does not exist"

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_input = np.asarray(image_rgb, dtype=np.float32) / 255.0
        image_input = cv2.resize(image_input, resize, interpolation=cv2.INTER_NEAREST)

        if std is not None and mean is not None:
            image_input = (image_input - mean) / std

        image_input = np.expand_dims(image_input, 0)
        pred = model.predict_on_batch(image_input)[0]

        # Get the predicted mask (argmax across classes)
        # pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
        # pred_mask = (pred_mask * 255).astype(np.uint8)  # scale for saving

        # # Save predicted mask image
        # base_filename = os.path.basename(image_path).split('.')[0]
        # output_path = os.path.join(output_dir, f"{base_filename}_pred.png")
        # cv2.imwrite(output_path, pred_mask)
        # print(f"Saved prediction to: {output_path}")
        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
        pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create color mask using colormap
        color_mask = colormap[pred_mask_resized % len(colormap)]

        # Overlay the mask on the original image (BGR for OpenCV)
        overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

        # Save outputs
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(output_dir + '/masks', f"{base_filename}_mask.png")
        overlay_path = os.path.join(output_dir, f"{base_filename}_overlay.png")

        cv2.imwrite(mask_path, color_mask)
        cv2.imwrite(overlay_path, overlay)

        print(f"Saved mask to: {mask_path}")
        print(f"Saved overlay to: {overlay_path}")

    # Zip all and save to /data/outputs -> scores, predictions and model
    zip_all(
        parent_dir=SAVE_RESULTS_DIR,
        destination_dir=SAVE_PREDICTIONS_DIR
    )


def evaluate(model_path, test_dir, num_classes, batch_size=5, std=None, mean=None):
    print(f"Evaluate model... {model_path}")
    focal_tversky_loss = custom_focal_tversky(num_classes=num_classes)
    model = load_model(model_path,
        custom_objects={'multi_class_focal_tversky_loss': focal_tversky_loss
        }, compile=False)

    generator = generator_v2(test_dir, batch_size=batch_size, std=std, mean=mean)

    eval_steps = len(os.listdir(os.path.join(test_dir, 'images'))) // batch_size
    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss = custom_focal_tversky(num_classes=num_classes)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', OneHotMeanIoU(num_classes=num_classes)])

    scores = model.evaluate(generator, steps=eval_steps, verbose=1)

    # combine names and scores
    scores_dict = {}
    for i in range(len(model.metrics_names)):
        scores_dict[model.metrics_names[i]] = scores[i]

start_time = datetime.now()
print("Start time:", start_time)

#Downloads prerequisites
download_datasets()
download_models()

# Actual training for the first model
kaggle_model = init_model()
kaggle_model.summary()


kaggle_path = './Kaggle Dataset'
BATCH_SIZE = 5
# init generator
print('Initiate generator v2...')
generator = generator_v2(kaggle_path, batch_size=BATCH_SIZE, std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])

EPOCHS = 1
STEPS_PER_EPOCH = len(os.listdir(os.path.join(kaggle_path, 'images'))) // BATCH_SIZE

save_format = 'h5'
save_name = 'kaggle_model'

print(f'Start training model {save_name} with format {save_format}...')
print(f'Number of epochs: {EPOCHS} and steps per epoch {STEPS_PER_EPOCH}')

history = kaggle_model.fit(
        generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        batch_size=BATCH_SIZE,
        verbose=1,
        initial_epoch=0,
        epochs=EPOCHS,
        callbacks=get_checkpoints(save_name, save_format))

test_dir = 'InsureValDataset/images'
# To use your trained model, uncomment the first line starting with 'model_path' and comment the second
# model_path = os.path.join(os.path.join(SAVE_RESULTS_DIR, 'models'), 'kaggle_model.h5')


model_path = os.path.join('.', 'kaggle_model.h5')
print(f"model_path: {model_path}")
assert os.path.exists(model_path), "Model not found at path!"


print(f"Testing model: {model_path}")

evaluate(model_path, 'InsureValDataset', num_classes=2, batch_size=5, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
test(model_path, test_dir, num_classes=2, num_images=16, max_num_images_per_line=4, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))

end_time = datetime.now()
print("End time:", end_time)

# Delta
delta = end_time - start_time
print("Duration:", delta)