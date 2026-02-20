import os
import json
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import zipfile

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def overlay_mask_on_image(image, mask, alpha=0.5):
    # convert mask to RGB
    mask_cv = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # create mask
    overlay = mask_cv.copy()
    output = image.copy()

    overlay = np.asarray(overlay, dtype=np.uint8)

    # make overlay green
    for i in range(0, overlay.shape[0]):
        for j in range(0, overlay.shape[1]):
            if overlay[i,j,0] >= 1:
                overlay[i,j,0] = 0
                overlay[i,j,1] = 255
                overlay[i,j,2] = 0
    # apply alpha
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output

def save_images_with_mask_overlay(images, output_path, masks, std=None, mean=None):
    images = np.asarray(images)
    masks = np.asarray(masks)

    if std is not None and mean is not None:
        for image in images:
            for i in range(3):
                image[:, :, i] = image[:, :, i] * std[i] + mean[i]
                image[:, :, i] = image[:, :, i] * 255.0

        images = np.asarray(images, dtype=np.uint8)

    if len(masks.shape) > 3:
        masks = np.argmax(masks, axis=-1)
        masks = np.asarray(masks, dtype=np.uint8)

    for i, image in enumerate(images):
        image_with_overlay = overlay_mask_on_image(image, masks[i])

        image_path = output_path if output_path is not None else 'test' + str(i) + ".png"

        # Save the image using OpenCV
        cv2.imwrite(image_path, image_with_overlay)

def run_prediction(model_path, input_images_path, output_path, num_images=None, max_num_images_per_line=1, std=None, mean=None):
    print("Loading the model from path: " + model_path)
    model = load_model(model_path, compile=False)
    output_directory = output_path if output_path is not None else '/data/outputs/results'
    print("Results directory: " + output_directory)

    images_paths = [os.path.join(input_images_path, image_filename) for image_filename in os.listdir(input_images_path)]
    num_images = num_images if num_images is not None and num_images < len(images_paths) else len(images_paths)
    images_paths = images_paths[:num_images]
    print("Number of input images:" + str(num_images))

    resize = (model.input_shape[1], model.input_shape[2])

    images = []
    masks = []
    for image_path in images_paths:
        print("Running prediction on: " + image_path)
        if not image_path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue

        assert os.path.exists(image_path), f"The image path: {image_path} does not exist"

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        image = np.asarray(image, dtype=np.float32)

        if std is not None and mean is not None:
            image = (image - mean) / std

        image = np.expand_dims(image, 0)
        pred = model.predict_on_batch(image)[0]

        images.append(image[0])
        masks.append(pred)
        
        file_name = os.path.basename(image_path)
        file_path = os.path.join(output_directory, file_name)
        if len(images) == max_num_images_per_line:
            images = np.asarray(images, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.float32)
            save_images_with_mask_overlay(images, file_path, masks, std=std, mean=mean)

            images = []
            masks = []

    if len(images) > 0:
        images = np.asarray(images, dtype=np.float32)
        masks = np.asarray(masks, dtype=np.float32)
        save_images_with_mask_overlay(images, file_path, masks, std=std, mean=mean)


def getInputFiles():
    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)
    files = []
    for did in dids:
        inputFile = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {inputFile}.")

        files.append(inputFile)

    return files

def printFilesAndDirs(directory):
    print('directory = ' + directory)

    print('directory (absolute) = ' + os.path.abspath(directory))

    for root, subdirs, files in os.walk(directory):
        print('--\nroot = ' + root)

        for filename in files:
            file_path = os.path.join(root, filename)

            print('\t- file %s (full path: %s)' % (filename, file_path))
        
        for subdir in subdirs: 
            print('\t- subdirectory ' + subdir)
            printFilesAndDirs(subdir)

def printEnvVariables():
    for name, value in os.environ.items():
        print("{0}: {1}".format(name, value))

def createDirectory(directory_path):
    print("Creating a new directory at path: " + directory_path)
    try:
        os.mkdir(directory_path)
    except OSError as error:
        print("Error creating directory") 
    
def main():
    print("----======Start======----") 
    print("\n Please pass: \n1. The model DID first(.h5 model) \n 2. The images zip files\n")

    print(f"\n\n-----======ENV VAR=======--------")    
    printEnvVariables()
    
    print(f"\n\n-----======DATA DIR=======--------")    
    printFilesAndDirs('data')

    input_files = getInputFiles()
    if (input_files is None or len(input_files) < 2):
        print("Error - input files not available")
        return

    model_path = input_files[0] 
    print("Reading the model from path: " + model_path)
    assert os.path.exists(model_path), "Model not found at path!"

    input_images_path = input_files[1]
    print("Reading images from path: " + input_images_path)
    assert os.path.exists(model_path), "Images not found at path!"

    unzipped_path = '/data/outputs/images'
    createDirectory(unzipped_path)
    
    results_path = '/data/outputs'
    # createDirectory(results_path)

    print("Unzipping the input files. Input path: " + input_images_path + " output path: " + unzipped_path)
    with zipfile.ZipFile(input_images_path, 'r') as zip_ref:
        zip_ref.extractall(unzipped_path)

    # The zip file contains a folder called images 
    unzipped_images_path = os.path.join(unzipped_path, "images")
    
    print("Unzipped images are at folder: " + unzipped_images_path) 

    print("----======Running prediction======----") 
    run_prediction(model_path=model_path, 
                input_images_path=unzipped_images_path, 
                output_path=results_path,
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225]))
    
    print("----======End======----") 

if __name__ == "__main__":
    main()
