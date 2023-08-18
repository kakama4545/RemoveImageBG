import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import uuid
import tempfile
from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image
from flask import Flask, request, send_file
import atexit
import shutil
import threading
import sched
import time

app = Flask(__name__)

# Get The Current Directory
currentDir = os.path.dirname(__file__)

# ------- Load Trained Model --------
print("---Loading Model---")
model_name = 'u2net'
model_dir = os.path.join(currentDir, 'saved_models', model_name, model_name + '.pth')
net = U2NET(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
# ------- Load Trained Model --------

# Functions:
# Save Results
def save_output(image_name, output_name, pred, d_dir, type):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        imo = Image.fromarray(imo, 'RGBA')

    imo.save(d_dir+output_name)

# Remove Background From Image (Generate Mask, and Final Results)
def removeBg(imageFile):
    # Create a temporary directory to store the images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save image to the temporary directory
        unique_filename = str(uuid.uuid4())
        image_path = os.path.join(temp_dir, unique_filename + '.jpg')
        imageFile.save(image_path)

        # processing
        image = io.imread(image_path)
        image = transform.resize(image, (320, 320), mode='constant')

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

        tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
        tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
        tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpImg = np.expand_dims(tmpImg, 0)
        image = torch.from_numpy(tmpImg)

        image = image.type(torch.FloatTensor)
        image = Variable(image)

        d1, d2, d3, d4, d5, d6, d7 = net(image)
        pred = d1[:, 0, :, :]
        ma = torch.max(pred)
        mi = torch.min(pred)
        dn = (pred-mi)/(ma-mi)
        pred = dn

        save_output(image_path, unique_filename + '.png', pred, os.path.join(currentDir, 'static/results/'), 'image')
        #save_output(image_path, unique_filename + '.png', pred, os.path.join(currentDir, 'static/masks/'), 'mask')

        # Return the URL of the processed image
        output_image_url = os.path.join(currentDir, 'static/results', unique_filename + '.png')
        return {'output_image_url': output_image_url}

# Helper function to handle errors during file deletion
def on_rm_error(func, path, exc_info):
    # Handle any error that occurs while deleting files
    print(f"Error deleting file: {path} - {exc_info}")

# Cleanup function to delete old temporary files
def cleanup_temp_files():
    temp_dir = os.path.join(currentDir, 'static/results')
    for file_name in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file_name)
        try:
            # Check if the file is older than 5 minutes
            if (time.time() - os.path.getctime(file_path)) // 60 >= 1:
                os.remove(file_path)  # Delete the file
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")


# Schedule cleanup function to run every 5 minutes
cleanup_scheduler = sched.scheduler(time.time, time.sleep)

def schedule_cleanup():
    cleanup_temp_files()
    cleanup_scheduler.enter(300, 1, schedule_cleanup, ())

cleanup_scheduler.enter(300, 1, schedule_cleanup, ())
cleanup_thread = threading.Thread(target=cleanup_scheduler.run)
cleanup_thread.daemon = True
cleanup_thread.start()

# Register the cleanup function to run at program exit
atexit.register(cleanup_temp_files)

# API route to remove background from an image
@app.route('/remove_bg', methods=['POST'])
def remove_bg_api():
    if 'image' not in request.files:
        return 'No image data provided.', 400

    # Get the image data from the POST request
    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected image.', 400

    # Call removeBg function to process the image
    result = removeBg(image_file)
    if 'output_image_url' in result:
        # Return the processed image directly
        return send_file(result['output_image_url'], mimetype='image/png')
    else:
        return 'Unknown error occurred.', 500

if __name__ == "__main__":
    app.run(debug=True)
