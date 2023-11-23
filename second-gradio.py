import gradio as gr
import numpy as np
import cv2

import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation

def segment(image): 
    with mp_selfie.SelfieSegmentation(model_selection=0) as model: 
        res = model.process(image)
        mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5 
        return np.where(mask, image, cv2.blur(image, (40,40)))
def flip(im):
    return np.flipud(im)

with gr.Blocks() as demo:
  with gr.Row():
      with gr.Column():
        camera = gr.Image(sources="webcam")
      with gr.Column():
        image = gr.Image()
  camera.stream(fn=segment, inputs=[camera], outputs=[camera])

demo.launch()