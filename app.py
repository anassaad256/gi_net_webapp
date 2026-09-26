"""
GI-NET Ki-67 Grade Prediction Web Application.

Hugging Face Spaces deployment for predicting Ki-67 proliferation grade
(G1 vs G2+G3) from a set of H&E tiles from one case. For research and
testing use only; not for clinical use.
"""

import os
import zipfile
from collections.abc import Iterator

import gradio as gr
from PIL import Image

from predictor import (
    MAX_TILES,
    MIN_RECOMMENDED_TILES,
    RESEARCH_USE_NOTICE,
    GINETPredictor,
)

# Configuration - weights will be downloaded from HF Hub automatically
MODEL_REPO = os.environ.get("MODEL_REPO", "")  # e.g., "username/gi-net-weights"
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/ABMIL_binary")
DEVICE = os.environ.get("DEVICE", "cuda")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
TOP_TILES_SHOWN = 12

# Global predictor instance (lazy loaded)
predictor = None


def load_model() -> GINETPredictor:
    """Load the predictor model (lazy initialization)."""
    global predictor
    if predictor is None:
        predictor = GINETPredictor(
            model_dir=MODEL_DIR,
            model_repo=MODEL_REPO if MODEL_REPO else None,
            device=DEVICE,
        )
    return predictor


def _is_image_name(name: str) -> bool:
    base = os.path.basename(name)
    return base.lower().endswith(IMAGE_EXTENSIONS) and not base.startswith(".")


def iter_uploaded_images(paths: list[str]) -> Iterator[tuple[str, Image.Image]]:
    """Yield (name, image) for every image file and every image inside a zip."""
    for path in paths:
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                for member in sorted(zf.namelist()):
                    if member.startswith("__MACOSX/") or not _is_image_name(member):
                        continue
                    with zf.open(member) as f:
                        image = Image.open(f)
                        image.load()
                    yield os.path.basename(member), image
        elif _is_image_name(path):
            image = Image.open(path)
            image.load()
            yield os.path.basename(path), image


def predict_case(files) -> tuple[str, str, list, list, str]:
    """
    Gradio prediction function.

    Args:
        files: Uploaded file paths (images and/or zip archives) from one case

    Returns:
        Tuple of (prediction_text, probability_text, top_tiles_gallery,
        attention_table, interpretation_text)
    """
    empty = ("", "", [], [], "")
    if not files:
        return ("Please upload tiles from one case.",) + empty[1:]

    paths = [f if isinstance(f, str) else f.name for f in files]

    try:
        pred = load_model().predict_case(iter_uploaded_images(paths))
    except FileNotFoundError as e:
        return (f"**Model Error:** {str(e)}",) + empty[1:]
    except Exception as e:
        return (f"**Error:** {str(e)}",) + empty[1:]

    stats = pred["tile_stats"]
    result_text = (
        f"{RESEARCH_USE_NOTICE}\n\n"
        f"## {pred['prediction']}\n\n"
        f"**Confidence:** {pred['confidence']*100:.1f}%"
    )
    if pred["n_tiles"] < MIN_RECOMMENDED_TILES:
        result_text += (
            f"\n\n**⚠️ Only {pred['n_tiles']} tile(s) analyzed.** The model was validated "
            f"on whole cases; results from fewer than {MIN_RECOMMENDED_TILES} tiles are "
            "much less reliable."
        )

    prob_text = f"""
### Class Probabilities

| Grade | Probability |
|-------|-------------|
| G1 (Ki-67 <3%) | {pred['prob_g1']*100:.1f}% |
| G2+G3 (Ki-67 ≥3%) | {pred['prob_g2g3']*100:.1f}% |

**Tiles:** {stats['n_used']} analyzed out of {stats['n_candidates']} found in the upload
({stats['n_rejected_qc']} rejected by tissue QC, {stats['n_sampled_out']} left out by the
{MAX_TILES}-tile cap)
"""

    # Attention sums to 1 over the case; show it relative to a uniform weight
    # so values are comparable across cases with different tile counts.
    uniform = 1.0 / pred["n_tiles"]
    tiles = pred["tiles"]
    gallery = [
        (t["image"], f"#{i + 1} {t['name']} · {t['attention'] / uniform:.1f}× avg")
        for i, t in enumerate(tiles[:TOP_TILES_SHOWN])
    ]
    table = [
        [i + 1, t["name"], round(t["attention"] * 100, 2), round(t["attention"] / uniform, 2)]
        for i, t in enumerate(tiles)
    ]

    return result_text, prob_text, gallery, table, pred["interpretation"]


# Create the Gradio interface
with gr.Blocks(
    title="GI-NET Ki-67 Grade Prediction (Research Use Only)",
    theme=gr.themes.Soft(),
    css="""
    .main-title {
        text-align: center;
        margin-bottom: 1rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #5c4400;
    }
    """,
) as demo:
    gr.Markdown(
        """
        # 🔬 GI-NET Ki-67 Grade Prediction from H&E

        <div class="disclaimer">

        **⚠️ FOR RESEARCH AND TESTING USE ONLY. NOT FOR CLINICAL USE.**
        This is a research prototype. It has not been validated or approved for diagnosis,
        grading, or any patient-care decision. Do not upload identifiable patient information.

        </div>

        Upload H&E tiles from **one case**. The model scores all tiles together with
        attention-based multiple instance learning (ABMIL), the same case-level method
        used in the published study, and predicts:

        - **G1**: Ki-67 <3% (low proliferation)
        - **G2+G3**: Ki-67 ≥3% (intermediate/high proliferation)

        **Published performance (held-out test set, n=44 cases, single institution, case level):**
        94.9% balanced accuracy | 0.90 Cohen's kappa | 97% G1 sensitivity | 93% G2+G3 sensitivity
        """,
        elem_classes=["main-title"],
    )

    with gr.Accordion("How to prepare tiles", open=True):
        gr.Markdown(
            f"""
            - **One case per run.** All uploaded tiles are pooled into a single prediction.
            - **Tile format:** 1024×1024 px at **40×** magnification, as in training.
              Larger images are cut into 1024×1024 tiles automatically.
            - **How many:** ideally 100 to {MAX_TILES} tiles sampled across the whole tumor,
              not a single hotspot. Fewer than {MIN_RECOMMENDED_TILES} tiles triggers a warning;
              above {MAX_TILES}, a random {MAX_TILES} are used, as in training.
            - **What to include:** tumor-containing tissue. Tiles with less than 30% tissue
              are rejected automatically.
            - **Upload:** select multiple PNG/JPG/TIFF files, or one .zip of tiles.
            - A single tile is not a meaningful input: tile-level accuracy in the study was
              about 69%, versus 94.9% when tiles are aggregated per case.
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            files_input = gr.File(
                label="Upload tiles from one case (images or .zip)",
                file_count="multiple",
                file_types=[".zip", *IMAGE_EXTENSIONS],
                type="filepath",
            )
            predict_btn = gr.Button("🔍 Predict Case Grade", variant="primary", size="lg")

        with gr.Column(scale=1):
            prediction_output = gr.Markdown(label="Prediction")
            probability_output = gr.Markdown(label="Probabilities")

    gr.Markdown(
        f"""
        ### Most-attended tiles
        The {TOP_TILES_SHOWN} tiles the model weighted most heavily. "× avg" is each
        tile's attention relative to an equal share across all tiles. Attention shows
        where the model looked, not a validated map of proliferation.
        """
    )
    gallery_output = gr.Gallery(label="Top tiles by attention", columns=6, height="auto")

    with gr.Accordion("Attention for all tiles", open=False):
        attention_table = gr.Dataframe(
            headers=["Rank", "Tile", "Attention (%)", "× avg"],
            datatype=["number", "str", "number", "number"],
            interactive=False,
        )

    interpretation_output = gr.Markdown(label="Interpretation")

    predict_btn.click(
        fn=predict_case,
        inputs=[files_input],
        outputs=[
            prediction_output,
            probability_output,
            gallery_output,
            attention_table,
            interpretation_output,
        ],
    )

    gr.Markdown(
        """
        ---

        <div class="disclaimer">

        **⚠️ Disclaimer: for research and testing use only. Not for clinical use.**
        This tool is not a medical device and has not been validated or approved for
        diagnosis, grading, or treatment decisions. It was developed on a
        single-institution dataset. Ki-67 grading must be performed by a qualified
        pathologist using standard methods, including Ki-67 immunohistochemistry.

        </div>

        ---

        **About:** This application uses an Attention-Based Multiple Instance Learning (ABMIL)
        model with H-optimus-0 feature extraction to predict Ki-67 grade from H&E tiles.
        The model was trained on GI-NET histopathology data using 5-fold cross-validation,
        and the five fold models are ensembled at inference.
        """
    )

# For Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()
