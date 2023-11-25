import gradio as gr

from model_loader import load
canvas_size = 600

def process_image(pr, steps, cfg, seed):
    return infer_no_image(
        prompt=pr,
        num_inference_steps=steps,
        guidance_scale=cfg,
        seed=int(seed)
    )

def update_model(model_name):
    global infer_no_image
    infer_no_image = load(model_name)


with gr.Blocks() as demo:
    infer_no_image = load()
    with gr.Row():
        prompt = gr.Text(label="Enter Prompt", value="Mighty eagle, 4K, realistic, dazzling", interactive=True)
        s = gr.Slider(label="Choose number of steps", minimum=4, maximum=8, step=1, value=4, interactive=True)
        c = gr.Slider(label="Guidance scale", minimum=0.0, maximum=3, step=0.1, value=0, interactive=True)
        se = gr.Number(label="Seed", value=1337, interactive=True)
    with gr.Column():
        btn = gr.Button(value="Run Model")
    with gr.Row():
        o = gr.Image(label="Upload a picture", height=canvas_size, width=canvas_size )
        btn.click(process_image, inputs=[prompt, s, c, se], outputs=o)

if __name__ == "__main__":
    demo.launch(show_api=False)




