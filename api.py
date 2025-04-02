from flask import Flask, request, jsonify, send_file
import os
import torch
from pathlib import Path
from generate import load_model, prepare_inputs, SyncDDIMSampler, SyncMultiviewDiffusion
from skimage.io import imsave
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Load your model here (same as generate.py)
MODEL_CFG = 'configs/syncdreamer.yaml'
MODEL_CKPT = 'ckpt/syncdreamer-pretrain.ckpt'

def load_model_for_api(cfg, ckpt):
  model = load_model(cfg, ckpt, strict=True)
  return model

model = load_model_for_api(MODEL_CFG, MODEL_CKPT)

# Define the API endpoint to process images
@app.route('/generate', methods=['POST'])
def generate_image():
  if 'file' not in request.files:
    return jsonify({"error": "No file part"}), 400
  
  file = request.files['file']
  
  if file.filename == '':
    return jsonify({"error": "No selected file"}), 400
  
  if file:
    # Save the image temporarily
    input_image_path = 'input_image.png'
    file.save(input_image_path)
    
    # Optional: Get parameters from the form (you can add more)
    cfg_scale = float(request.form.get('cfg_scale', 2.0))
    elevation = float(request.form.get('elevation', 30))
    sample_num = int(request.form.get('sample_num', 4))
    crop_size = int(request.form.get('crop_size', -1))
    output_dir = request.form.get('output_dir', 'output/aircraft')  # Allow output directory to be passed in

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare the inputs (similar to generate.py)
    data = prepare_inputs(input_image_path, elevation, crop_size)
    
    # Repeat data as required
    for k, v in data.items():
      data[k] = v.unsqueeze(0).cuda()
      data[k] = torch.repeat_interleave(data[k], sample_num, dim=0)

    # Set up the sampler (DDIM in this case)
    sampler = SyncDDIMSampler(model, sample_steps=50)
    x_sample = model.sample(sampler, data, cfg_scale, batch_view_num=8)

    # Convert the tensor to a numpy array and save it as an image
    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    # Save the output to the output directory
    output_filename = os.path.join(output_dir, 'generated_image.png')
    imsave(output_filename, np.concatenate([x_sample[0, ni] for ni in range(N)], 1))  # Using the first generated sample

    # Create a BytesIO object to send the image back in the response
    output_image = BytesIO()
    imsave(output_image, np.concatenate([x_sample[0, ni] for ni in range(N)], 1))  # Using the first generated sample
    output_image.seek(0)

    return send_file(output_image, mimetype='image/png')

@app.route('/health', methods=['GET'])
def health_check():
  return jsonify({"status": "API is running"}), 200

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
