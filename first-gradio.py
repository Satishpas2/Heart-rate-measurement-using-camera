import gradio as gr

def predict(video_in, image_in_video, image_in_img):
    if video_in == None and image_in_video == None and image_in_img == None:
        raise gr.Error("Please upload a video or image.")
    if image_in_video or image_in_img:
        print("image", image_in_video, image_in_img)
        image = image_in_video or image_in_img
        return image

    return video_in


def toggle(choice):
    if choice == "webcam":
        return gr.update(visible=True, value=None), gr.update(visible=False, value=None)
    else:
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None)


with gr.Blocks() as blocks:
    gr.Markdown("### Video or Image? WebCam or Upload?""")
    with gr.Tab("Video") as tab:
        with gr.Row():
            with gr.Column():
                video_or_file_opt = gr.Radio(["webcam", "upload"], value="webcam",
                                             label="How would you like to upload your video?")
                video_in = gr.Video(sources="webcam", include_audio=False)
                video_or_file_opt.change(fn=lambda s: gr.update(source=s, value=None), inputs=video_or_file_opt,
                                         outputs=video_in, queue=False, show_progress=False)
            with gr.Column():
                video_out = gr.Video()
        run_btn = gr.Button("Run")
        run_btn.click(fn=predict, inputs=[video_in], outputs=[video_out])
        gr.Examples(fn=predict, examples=[], inputs=[
                    video_in], outputs=[video_out])

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image_or_file_opt = gr.Radio(["webcam", "file"], value="webcam",
                                             label="How would you like to upload your image?")
                image_in_video = gr.Image(sources="webcam", type="filepath")
                image_in_img = gr.Image(
                    sources="upload", visible=False, type="filepath")

                image_or_file_opt.change(fn=toggle, inputs=[image_or_file_opt],
                                         outputs=[image_in_video, image_in_img], queue=False, show_progress=False)
            with gr.Column():
                image_out = gr.Image()
        run_btn = gr.Button("Run")
        run_btn.click(fn=predict, inputs=[
                      image_in_img, image_in_video], outputs=[image_out])
        gr.Examples(fn=predict, examples=[],  inputs=[
                    image_in_img, image_in_video], outputs=[image_out])

blocks.queue()
blocks.launch()
