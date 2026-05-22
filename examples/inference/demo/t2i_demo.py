import argparse
import base64
import logging
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import yaml
from model_utils import get_model_from_config

logging.basicConfig(level=logging.INFO)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024
DEFAULT_NUM_STEPS = 50
DEFAULT_SHIFT_VALUE = 3.0

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGO_PATH = REPO_ROOT / "assets" / "logo.png"


def _logo_data_uri() -> Optional[str]:
    if not LOGO_PATH.is_file():
        return None
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cpu":
    logging.warning(
        "No CUDA device available — running on CPU will be extremely slow and "
        "may OOM when loading the text encoder/VAE."
    )


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelKey:
    config_path: str
    ckpt_name: str
    phase: str


class ModelCache:
    """Thread-safe holder for the currently-loaded diffusion model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: Optional[ModelKey] = None
        self._model = None
        self._global_step: Optional[int] = None

    def get(self, key: ModelKey, progress: Optional[gr.Progress] = None):
        with self._lock:
            if self._key == key and self._model is not None:
                return self._model, self._global_step

            def _progress(frac: float, desc: str) -> None:
                if progress is not None:
                    progress(frac, desc=f"[{key.phase}] {desc}")

            _progress(0.0, "Reading config…")
            gr.Info(f"Loading model (phase={key.phase})…", duration=10)

            with open(key.config_path, "r") as f:
                cfg_yaml = yaml.safe_load(f)

            phase_cfg = next(
                (p for p in cfg_yaml["phases"] if p["name"] == key.phase), None
            )
            if phase_cfg is None:
                raise gr.Error(f"Phase '{key.phase}' not found in {key.config_path}")

            ckpt_path = phase_cfg["training"]["save_ckpt_path"]
            model, global_step = get_model_from_config(
                phase_cfg["model"],
                key.ckpt_name,
                ckpt_path,
                progress_cb=_progress,
            )

            _progress(0.95, f"Moving model to {DEVICE}…")
            model.eval()
            model.to(DEVICE, DTYPE)
            model.conditioner.to(DEVICE, DTYPE)
            model.vae.to(DEVICE, DTYPE)

            self._key = key
            self._model = model
            self._global_step = global_step
            _progress(1.0, "Model ready")
            gr.Info(
                f"Model ready (phase={key.phase}, step={global_step}, device={DEVICE})."
            )
            return self._model, self._global_step


CACHE = ModelCache()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.inference_mode()
def infer(
    prompt,
    seed,
    randomize_seed,
    num_steps,
    guidance_scale,
    use_default_shift,
    shift_value,
    resolution,
    num_samples,
    model_config_path,
    phase,
    ckpt_name,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is empty.")

    seed = int(random.randint(0, MAX_SEED)) if randomize_seed else int(seed)

    try:
        model, _ = CACHE.get(
            ModelKey(
                config_path=model_config_path,
                ckpt_name=ckpt_name,
                phase=phase,
            ),
            progress=progress,
        )
    except gr.Error:
        raise
    except Exception as e:
        logging.exception("Failed to load model")
        raise gr.Error(f"Failed to load model: {e}") from e

    num_samples = int(num_samples)
    resolution = int(resolution)
    num_steps = int(num_steps)
    effective_shift = None if use_default_shift else float(shift_value)

    progress(0.1, desc="Encoding prompt & sampling…")

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randn(
        num_samples,
        model.vae.latent_channels,
        resolution // model.vae.downsampling_factor,
        resolution // model.vae.downsampling_factor,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )

    try:
        samples = model.sample(
            z,
            num_steps=num_steps,
            uncond_conditioner_inputs={"text": [""] * num_samples},
            conditioner_inputs={"text": [prompt] * num_samples},
            guidance_scale=guidance_scale,
            shift_value=effective_shift,
            max_samples=num_samples,
            do_guidance=guidance_scale > 1.0,
        )
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise gr.Error(
            f"CUDA out of memory — try fewer samples or a lower resolution. ({e})"
        ) from e

    samples = samples.float().add(1.0).div(2.0).clamp(0.0, 1.0)
    samples = samples.mul(255.0).to(torch.uint8).cpu()
    images = [s.permute(1, 2, 0).numpy() for s in samples]

    progress(1.0, desc="Done")
    return images, seed


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLES_PROMPTS = [
    "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
    "A moody black and white film portrait of an older artisan. The lighting is extremely dramatic high-key chiaroscuro, illuminating only one half of the face in sharp relief, casting the rest into deep, absolute shadow. The texture of their wrinkles and beard is hyper-detailed. Grainy analog film aesthetic.",
    "A whimsical purple and pink dragon made of clay, showing tiny fingerprint marks and handcrafted texture, soft studio lighting, stop-motion aesthetic.",
    "An ultra-detailed nocturnal landscape of a hidden tropical lagoon. The water glows with intense neon blue bioluminescence where it laps against jet-black volcanic sand. Towering ancient banyan trees with glowing hanging vines frame the scene. In the background, a massive silver moon hangs low over a calm ocean, casting a shimmering path on the waves. Fireflies create bokeh light clusters in the dark jungle shadows. Intricate textures of wet sand and leaf veins. Surreal atmospheric lighting, high contrast, 16k masterwork, Unreal Engine 5 render style.",
]


CSS = """
#col-container { margin: 0 auto; max-width: 960px; }
#logo-header { margin: 0 auto 8px auto; max-width: 720px; }
#logo-header img { width: 100%; height: auto; display: block; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Nano-T2I Demo") as demo:
        logo_uri = _logo_data_uri()
        if logo_uri is not None:
            gr.HTML(
                f'<div id="logo-header"><img src="{logo_uri}" alt="Nano-T2I"/></div>'
            )
        else:
            gr.Markdown("# Nano-T2I Demo")

        with gr.Column(elem_id="col-container"):
            with gr.Row():
                model_config_path = gr.Text(
                    label="Model config path",
                    value="examples/trainings/configs/nano.yaml",
                )
                phase = gr.Text(label="Phase", value="nano-512")
                ckpt_name = gr.Text(label="Checkpoint name", value="last.ckpt")

            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="Enter your prompt",
                    container=False,
                    scale=8,
                )
                run_button = gr.Button("Run", scale=1, variant="primary")
                stop_button = gr.Button("Stop", scale=1, variant="stop")

            result = gr.Gallery(
                label="Result",
                show_label=False,
                columns=4,
                object_fit="contain",
                preview=True,
            )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
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
                        maximum=MAX_IMAGE_SIZE,
                        step=256,
                        value=512,
                    )
                    num_steps = gr.Slider(
                        label="Number of steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=DEFAULT_NUM_STEPS,
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0,
                        maximum=20,
                        step=0.5,
                        value=3.5,
                    )
                    use_default_shift = gr.Checkbox(
                        label="Use model default shift", value=True
                    )
                    shift_value = gr.Slider(
                        label="Shift value",
                        minimum=1,
                        maximum=10,
                        step=0.1,
                        value=DEFAULT_SHIFT_VALUE,
                        interactive=False,
                    )

                use_default_shift.change(
                    fn=lambda use_default: gr.update(interactive=not use_default),
                    inputs=use_default_shift,
                    outputs=shift_value,
                )

            gr.Examples(
                EXAMPLES_PROMPTS,
                inputs=prompt,
                label="Examples",
                cache_examples=False,
            )

        run_event = gr.on(
            [run_button.click, prompt.submit],
            fn=infer,
            inputs=[
                prompt,
                seed,
                randomize_seed,
                num_steps,
                guidance_scale,
                use_default_shift,
                shift_value,
                resolution,
                num_samples,
                model_config_path,
                phase,
                ckpt_name,
            ],
            outputs=[result, seed],
            show_progress="full",
            trigger_mode="always_last",
        )
        stop_button.click(fn=None, cancels=[run_event])

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nano-T2I Gradio demo")
    p.add_argument("--host", default="127.0.0.1", help="Server bind address")
    p.add_argument("--port", type=int, default=7860, help="Server port")
    p.add_argument(
        "--share",
        action="store_true",
        help="Expose a public Gradio share link (off by default)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_demo().queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=CSS,
    )
