import gradio as gr
import json
import os
import random
import shutil
import argparse
from core.schemas import Config
import scripts.generate as generate
from core.utils.helpers import global_seed

def apply_magic_wand(base_prompts):
    QUALITY = ["trending on artstation", "featured on artstation", "artstationHQ", "rendered in Unreal Engine", "Octane Render", "V-Ray", "Raytraced", "CryEngine", "8k resolution", "4k UHD", "highly detailed", "intricate details", "sharp focus", "trending on Pixiv", "CGSociety", "Behance HD", "Flickr", "500px HDR"]
    ARTIST = ["by Greg Rutkowski", "by Simon Stålenhag", "by Loish", "by Salvador Dali", "by Zdzisław Beksiński", "by Claude Monet", "by Henri Matisse", "by James Gurney", "by Moebius", "by Yusuke Kozaki"]
    LIGHTING = ["volumetric lighting", "cinematic lighting", "rim lighting", "global illumination", "golden hour", "blue hour", "afternoon sun", "dappled sunlight", "chiaroscuro", "moody lighting", "backlit", "neon lighting"]
    MEDIUM = ["oil on canvas", "watercolor painting", "charcoal drawing", "pencil sketch", "matte painting", "concept art", "psychedelic surrealism", "low poly", "voxel art", "unreal engine 5", "cyberpunk", "steampunk", "uproar", "synthwave", "fauvism"]
    MAGIC = ["the cosmic void", "infinite detail", "control the soul", "cel-shaded", "a sense of awe"]
    
    results = []
    for bp in base_prompts.split("|"):
        bp = bp.strip()
        if not bp:
            continue
        modifiers = [
            random.choice(ARTIST),
            random.choice(MEDIUM),
            random.choice(QUALITY),
            random.choice(LIGHTING),
            random.choice(MAGIC)
        ]
        results.append(f"{bp}, {', '.join(modifiers)}")
    return "|".join(results)

def save_uploaded_file(temp_file):
    if not temp_file:
        return ""
    input_dir = "/mars/input/VQGAN-CLIP"
    os.makedirs(input_dir, exist_ok=True)
    filename = os.path.basename(temp_file)
    new_path = os.path.join(input_dir, filename)
    shutil.copy(temp_file, new_path)
    return new_path

def append_uploaded_file(temp_file, current):
    new_path = save_uploaded_file(temp_file)
    if not new_path:
        return current
    if current:
        return f"{current}|{new_path}"
    return new_path

def load_defaults():
    try:
        with open("./configs/local.json", "r") as f:
            return json.load(f)
    except:
        return {}

defaults = load_defaults()

def run_generate(
    base_prompt, prompts, image_prompts, max_iterations, save_freq, width, height,
    init_image, init_noise, init_weight, mse_decay_rate, output_dir,
    models_dir, clip_model, vqgan_checkpoint, vqgan_config, step_size,
    cutn, cut_pow, seed, optimizer, nwarm_restarts, augments, batch_count
):
    prompts_list = [p.strip() for p in prompts.split("|")] if prompts else []
    if not prompts_list and base_prompt and base_prompt.strip():
        prompts_list = [base_prompt.strip()]
        
    image_prompts_list = [p.strip() for p in image_prompts.split("|")] if image_prompts else []
    
    if not prompts_list and not image_prompts_list:
        raise gr.Error("Please enter at least one text prompt or image prompt!")
    
    params_dict = {
        "prompts": prompts_list,
        "image_prompts": image_prompts_list,
        "max_iterations": int(max_iterations),
        "save_freq": int(save_freq),
        "size": [int(width), int(height)],
        "init_image": init_image,
        "init_noise": init_noise,
        "init_weight": float(init_weight),
        "mse_decay_rate": int(mse_decay_rate),
        "output_dir": output_dir,
        "models_dir": models_dir,
        "clip_model": clip_model,
        "vqgan_checkpoint": vqgan_checkpoint,
        "vqgan_config": vqgan_config,
        "noise_prompt_seeds": [],
        "noise_prompt_weights": [], 
        "step_size": float(step_size),
        "cutn": int(cutn),
        "cut_pow": float(cut_pow),
        "seed": int(seed),
        "optimizer": optimizer,
        "nwarm_restarts": int(nwarm_restarts),
        "augments": augments
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    current_seed = int(seed)
    last_path = None
    
    for i in range(int(batch_count)):
        params_dict["seed"] = current_seed
        
        generate.PARAMS = Config(**params_dict)
        generate.PARAMS.seed = global_seed(generate.PARAMS.seed)
        generate.main()
        
        filename = "output"
        if len(generate.PARAMS.prompts):
            filename = '_'.join(generate.PARAMS.prompts).replace(' ', '_')
        filename = f"{filename}_{generate.PARAMS.seed}"
        path = f"{generate.PARAMS.output_dir}/{filename}.png"
        
        if os.path.exists(path):
            last_path = path
            
        if current_seed != -1:
            current_seed += 1
            
        yield last_path, last_path

with gr.Blocks(title="VQGAN-CLIP WebUI") as demo:
    gr.Markdown("# VQGAN-CLIP WebUI Generator")
    gr.Markdown("Queue up generation jobs and edit all standard local.json options from here.")
    
    with gr.Row():
        with gr.Column(scale=1):
            base_prompt = gr.Textbox(label="Base Prompt (Keep simple!)", value="")
            magic_btn = gr.Button("✨ Magic Wand Modifier", variant="secondary")
            prompts = gr.Textbox(label="Full Prompts (separate multiple with |)", value="|".join(defaults.get("prompts", [])))
            
            magic_btn.click(fn=apply_magic_wand, inputs=[base_prompt], outputs=[prompts])
            
            with gr.Row():
                image_prompts = gr.Textbox(label="Image Prompts (paths, separate with |)", value="|".join(defaults.get("image_prompts", [])), scale=3)
                upload_image_prompt = gr.Image(label="Upload Image Prompt", type="filepath", scale=1)
                upload_image_prompt.upload(
                    fn=append_uploaded_file,
                    inputs=[upload_image_prompt, image_prompts],
                    outputs=[image_prompts]
                )
            
            with gr.Row():
                width = gr.Number(label="Width", value=defaults.get("size", [960, 512])[0], precision=0)
                height = gr.Number(label="Height", value=defaults.get("size", [960, 512])[1], precision=0)
            
            with gr.Row():
                max_iterations = gr.Number(label="Max Iterations", value=defaults.get("max_iterations", 250), precision=0)
                save_freq = gr.Number(label="Save Frequency", value=defaults.get("save_freq", 50), precision=0)
                batch_count = gr.Slider(minimum=1, maximum=100, step=1, label="Batch Count", value=1)
            
            with gr.Row():
                seed = gr.Number(label="Seed (-1 for random)", value=defaults.get("seed", -1), precision=0)
                optimizer = gr.Dropdown(label="Optimizer", choices=["Adam","AdamW","Adagrad","Adamax","DiffGrad","AdamP","RAdam"], value=defaults.get("optimizer", "Adam"))
                
            with gr.Row():
                step_size = gr.Number(label="Step Size (LR)", value=defaults.get("step_size", 0.1))
                cutn = gr.Number(label="Cutouts (cutn)", value=defaults.get("cutn", 32), precision=0)
                cut_pow = gr.Number(label="Cut Power", value=defaults.get("cut_pow", 1.0))

            with gr.Accordion("Advanced Options", open=False):
                with gr.Row():
                    init_image = gr.Textbox(label="Init Image Path", value=defaults.get("init_image", ""), scale=3)
                    upload_init_image = gr.Image(label="Upload Init Image", type="filepath", scale=1)
                    upload_init_image.upload(fn=save_uploaded_file, inputs=[upload_init_image], outputs=[init_image])
                
                init_noise = gr.Dropdown(label="Init Noise", choices=["", "gradient", "pixels", "fractal"], value=defaults.get("init_noise", ""))
                init_weight = gr.Number(label="Init Weight", value=defaults.get("init_weight", 0.0))
                mse_decay_rate = gr.Number(label="MSE Decay Rate", value=defaults.get("mse_decay_rate", 0), precision=0)
                nwarm_restarts = gr.Number(label="nwarm_restarts", value=defaults.get("nwarm_restarts", -1), precision=0)
                
                output_dir = gr.Textbox(label="Output Dir", value=defaults.get("output_dir", "/mars/output/VQGAN-CLIP"))
                models_dir = gr.Textbox(label="Models Dir", value=defaults.get("models_dir", "./models"))
                clip_model = gr.Textbox(label="CLIP Model", value=defaults.get("clip_model", "ViT-B/16"))
                vqgan_checkpoint = gr.Textbox(label="VQGAN Checkpoint", value=defaults.get("vqgan_checkpoint", "/models/vqgan_imagenet_f16_16384.ckpt"))
                vqgan_config = gr.Textbox(label="VQGAN Config", value=defaults.get("vqgan_config", "/configs/models/vqgan_imagenet_f16_16384.json"))
                augments = gr.CheckboxGroup(label="Augments", choices=["Ji","Sh","Gn","Pe","Ro","Af","Et","Ts","Cr","Er","Re","Hf"], value=defaults.get("augments", ["Af", "Pe", "Ji", "Er"]))

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Output Image")
            output_file = gr.File(label="Downloadable Image")

    generate_btn.click(
        fn=run_generate,
        inputs=[
            base_prompt, prompts, image_prompts, max_iterations, save_freq, width, height,
            init_image, init_noise, init_weight, mse_decay_rate, output_dir,
            models_dir, clip_model, vqgan_checkpoint, vqgan_config, step_size,
            cutn, cut_pow, seed, optimizer, nwarm_restarts, augments, batch_count
        ],
        outputs=[output_image, output_file]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true", help="Launch on 0.0.0.0")
    args = parser.parse_args()

    # Enable queueing to batch queue requests sequentially without crash
    demo.queue()
    demo.launch(server_name="0.0.0.0" if args.listen else "127.0.0.1", server_port=2345, share=False, allowed_paths=["/mars/output/VQGAN-CLIP", "/mars/input/VQGAN-CLIP"])
