import logging
import random

import gradio as gr
import numpy as np
import torch
import yaml
from model_utils import get_model_from_config
from torchvision.utils import make_grid

logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024
NUM_INFERENCE_STEPS = 4

# Store the loaded model in a global variable
model_global = None
model_config_path_global = None
ckpt_name_global = None


def load_model(model_config_path, ckpt_name, phase: str = "nano-512"):
    global model_global, model_config_path_global, ckpt_name_global
    gr.Info("Loading model...", duration=40)
    with open(model_config_path, "r") as file:
        model_config_yaml = yaml.safe_load(file)

    for phase_yaml in model_config_yaml["phases"]:
        if phase_yaml["name"] == phase:
            model_config_yaml = phase_yaml
            break
    else:
        raise ValueError(f"Phase {phase} not found in model config")

    ckpt_path = model_config_yaml["training"]["save_ckpt_path"]

    model, global_step = get_model_from_config(
        model_config_yaml["model"], ckpt_name, ckpt_path
    )
    model.eval()
    model.to(device, torch.bfloat16)
    model.conditioner.to(device, torch.bfloat16)
    model.vae.to(device, torch.bfloat16)
    gr.Info(
        f"Model loaded from ckpt at step {global_step} for phase {phase}. Ready to generate images. Put model on device {device}"
    )
    model_global = model
    model_config_path_global = model_config_path
    ckpt_name_global = ckpt_name
    return None  # No need to return model, we use the global variable


def infer(
    prompt,
    seed,
    randomize_seed,
    num_steps,
    guidance_scale,
    shift_value,
    resolution,
    num_samples,
    model_config_path,
    phase,
    ckpt_name,
):

    global model_global, model_config_path_global, ckpt_name_global

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # fix seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Always ensure the correct model is loaded
    if (
        model_global is None
        or model_config_path_global != model_config_path
        or ckpt_name_global != ckpt_name
    ):
        load_model(model_config_path, ckpt_name, phase)

    conditioner_inputs = {
        "text": [prompt] * num_samples,
    }
    unconditional_conditioner_inputs = {
        "text": [""] * num_samples,
    }

    z = torch.randn(
        num_samples,
        model_global.vae.latent_channels,
        resolution // model_global.vae.downsampling_factor,
        resolution // model_global.vae.downsampling_factor,
    ).to(device, torch.bfloat16)

    samples = model_global.sample(
        z,
        num_steps=num_steps,
        uncond_conditioner_inputs=unconditional_conditioner_inputs,
        conditioner_inputs=conditioner_inputs,
        guidance_scale=guidance_scale,
        shift_value=shift_value,
        max_samples=num_samples,
        do_guidance=guidance_scale > 1.0,
    )

    samples = (samples.float().cpu() + 1.0) / 2.0
    grid = make_grid(samples, nrow=4)
    grid = grid.permute(1, 2, 0)
    grid = grid.mul(255).clamp(0, 255).to(torch.uint8)

    return grid.numpy()


examples_prompts = [
    "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
    "A moody black and white film portrait of an older artisan. The lighting is extremely dramatic high-key chiaroscuro, illuminating only one half of the face in sharp relief, casting the rest into deep, absolute shadow. The texture of their wrinkles and beard is hyper-detailed. Grainy analog film aesthetic.",
    "A whimsical purple and pink dragon made of clay, showing tiny fingerprint marks and handcrafted texture, soft studio lighting, stop-motion aesthetic.",
    "An ultra-detailed nocturnal landscape of a hidden tropical lagoon. The water glows with intense neon blue bioluminescence where it laps against jet-black volcanic sand. Towering ancient banyan trees with glowing hanging vines frame the scene. In the background, a massive silver moon hangs low over a calm ocean, casting a shimmering path on the waves. Fireflies create bokeh light clusters in the dark jungle shadows. Intricate textures of wet sand and leaf veins. Surreal atmospheric lighting, high contrast, 16k masterwork, Unreal Engine 5 render style.",
]


css = """
#col-container {
    margin: 0 auto;
    max-width: 512px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        f"""
    # ⚡ Bloom Sandbox
    """
    )
    with gr.Column(elem_id="col-container"):

        with gr.Row():
            model_config_path = gr.Text(
                label="config model path",
                value="examples/trainings/configs/nano.yaml",
            )
            phase = gr.Text(
                label="phase",
                value="nano-512",
            )
        with gr.Row():
            ckpt_name = gr.Text(
                label="ckpt name",
                value="last.ckpt",
            )
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0, variant="primary")

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Column():
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                num_samples = gr.Slider(
                    label="Number of samples",
                    minimum=1,
                    maximum=16,
                    step=1,
                    value=4,
                )
                resolution = gr.Slider(
                    label="Resolution",
                    minimum=256,
                    maximum=1024,
                    step=256,
                    value=512,
                )
                num_steps = gr.Slider(
                    label="Number of steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=50,
                )
            with gr.Column():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0,
                    maximum=20,
                    step=0.5,
                    value=3.5,
                )
                shift_value = gr.Slider(
                    label="Shift value",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    value=None,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

        examples = gr.Examples(
            examples_prompts, inputs=prompt, label="Examples", cache_examples=False
        )
    gr.on(
        [
            run_button.click,
        ],
        fn=infer,
        inputs=[
            prompt,
            seed,
            randomize_seed,
            num_steps,
            guidance_scale,
            shift_value,
            resolution,
            num_samples,
            model_config_path,
            phase,
            ckpt_name,
        ],
        outputs=[result],
        show_progress="minimal",
        trigger_mode="always_last",
    )

demo.queue().launch(server_name="127.0.0.1", server_port=7860)
