# import libraries

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO

# -----------------------------
# Device configuration
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# -----------------------------
# Style dictionary
# -----------------------------
style_dict = {
    "none": "",
    "anime": "cartoon, animated, Studio Ghibli style, cute, Japanese animation",
    "photo": "photograph, film, 35 mm camera",
    "video game": "rendered in unreal engine, hyper-realistic, volumetric lighting",
    "watercolor": "painting, watercolors, pastel, soft composition",
}

# -----------------------------
# Load mode
# -----------------------------
@st.cache_resource

def load_model():
    """
    Loads a CPU-friendly Stable Diffusion model and applies
    optimizations for inference without a GPU.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    # CPU optimizations
    pipe.enable_attention_slicing()

    return pipe


# -----------------------------
# Image generation function
# -----------------------------

def generate_image(prompt, negative_prompt, pipeline, guidance, steps, style):
    full_prompt = prompt + " " + style_dict.get(style, "")

    result = pipeline(
        full_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=384,
        width=384,
    )

    return result.images[0]


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Stable Diffusion")
    st.title("Stable Diffusion")

    st.sidebar.header("Generation Settings")

    prompt = st.sidebar.text_area("Text-to-Image Prompt")
    negative_prompt = st.sidebar.text_area(
        "Negative Prompt",
        help="What you want to avoid in the image",
        placeholder="blurry, low quality, distorted"
    )

    guidance = st.sidebar.slider(
        "Guidance Scale",
        min_value=2.0,
        max_value=7.0,
        value=5.0,
        help="Lower values are faster and more CPU-friendly",
    )

    steps = st.sidebar.slider(
        "Inference Steps",
        min_value=5,
        max_value=30,
        value=15,
        help="More steps = better quality but much slower on CPU",
    )

    batch_size = st.sidebar.slider(
        "Batch Size (Queued)",
        min_value=1,
        max_value=5,
        value=1,
        help="Images are generated sequentially to avoid CPU overload",
    )

    style = st.sidebar.selectbox("Style", options=list(style_dict.keys()))

    generate = st.sidebar.button("Generate Images")

    if generate:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        pipeline = load_model()

        progress_bar = st.progress(0)
        status_text = st.empty()

        images = []

        for i in range(batch_size):
            status_text.text(f"Generating image {i + 1} of {batch_size}...")
            image = generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                pipeline=pipeline,
                guidance=guidance,
                steps=steps,
                style=style,
            )
            images.append(image)
            progress_bar.progress((i + 1) / batch_size)

        status_text.text("Generation complete")

        for idx, img in enumerate(images):
            st.image(img, caption=f"Generated Image {idx + 1}", use_column_width=True)

            # Download button
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                label="Download image",
                data=buf.getvalue(),
                file_name=f"generated_{idx + 1}.png",
                mime="image/png",
            )


if __name__ == "__main__":
    main()
