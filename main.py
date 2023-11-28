import gradio as gr
import argparse

from model_loader import load
canvas_size = 600

def update_model(model_name):
    global infer_no_image
    infer_no_image = load(model_name)


def gradio_start(args):
    with gr.Blocks() as demo:
        infer_no_image = load(model_id=args.model)
        def process_image(pr, neg_pr, steps, cfg, seed):
            return infer_no_image(
                prompt=pr,
                negative_prompt=neg_pr,
                num_inference_steps=steps,
                guidance_scale=cfg,
                seed=int(seed)
            )

        with gr.Row():
            with gr.Column():
                prompt = gr.Text(label="Prompt", value="Mighty eagle, 4K, realistic, dazzling", interactive=True)
                negative_prompt = gr.Text(label="Negative Prompt", value="nsfw", interactive=True)
            s = gr.Slider(label="Choose number of steps", minimum=4, maximum=8, step=1, value=4, interactive=True)
            c = gr.Slider(label="Guidance scale", minimum=0.0, maximum=3, step=0.1, value=0, interactive=True)
            se = gr.Number(label="Seed", value=1337, interactive=True)
        with gr.Column():
            btn = gr.Button(value="Run Model")
        with gr.Row():
            o = gr.Image(label="Upload a picture", height=canvas_size, width=canvas_size )
            btn.click(process_image, inputs=[prompt, negative_prompt, s, c, se], outputs=o)

    demo.launch(show_api=False, ssl_verify=False, server_name=args.bind)




if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('--model', dest='model', type=str, default="CompVis/stable-diffusion-v1-4", help='model name')
    argparser.add_argument('--bind', dest='bind', type=str, default="127.0.0.1", help='Gradio bind address')
    args = argparser.parse_args()
    gradio_start(args)
    




