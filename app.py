"""
GI-NET Ki-67 Grade Prediction Web Application.

Hugging Face Spaces deployment for predicting Ki-67 proliferation grade
(G1 vs G2+G3) from H&E histopathology images. For research and testing use
only; not for clinical use.
"""

import os

import gradio as gr

from predictor import GINETPredictor

# Configuration - weights will be downloaded from HF Hub automatically
MODEL_REPO = os.environ.get("MODEL_REPO", "")  # e.g., "username/gi-net-weights"
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/ABMIL_binary")
DEVICE = os.environ.get("DEVICE", "cuda")

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


def predict_grade(image) -> tuple[str, str, str]:
    """
    Gradio prediction function.

    Args:
        image: Uploaded image (PIL Image from Gradio)

    Returns:
        Tuple of (prediction_text, probability_text, interpretation_text)
    """
    if image is None:
        return "Please upload an image.", "", ""

    try:
        pred = load_model().predict(image)

        # Format results for display
        result_text = f"**⚠️ For research and testing use only. Not for clinical use.**\n\n## {pred['prediction']}\n\n**Confidence:** {pred['confidence']*100:.1f}%"

        prob_text = f"""
### Class Probabilities

| Grade | Probability |
|-------|-------------|
| G1 (Ki-67 <3%) | {pred['prob_g1']*100:.1f}% |
| G2+G3 (Ki-67 ≥3%) | {pred['prob_g2g3']*100:.1f}% |

*{pred['n_tiles']} tissue region(s) analyzed*
"""

        return result_text, prob_text, pred["interpretation"]

    except FileNotFoundError as e:
        return f"**Model Error:** {str(e)}", "", ""
    except Exception as e:
        return f"**Error:** {str(e)}", "", ""


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
        This tool has not been validated or approved for diagnosis, grading, or any
        patient-care decision.

        </div>

        Upload an H&E histopathology image to predict Ki-67 proliferation grade using AI.

        **Supported grades:**
        - **G1**: Ki-67 <3% (low proliferation)
        - **G2+G3**: Ki-67 ≥3% (intermediate/high proliferation)

        **Reported performance (held-out test set of 44 cases, single institution):**
        94.9% balanced accuracy | κ = 0.90 | 97% G1 sensitivity | 93% G2+G3 sensitivity
        """,
        elem_classes=["main-title"],
    )

    with gr.Accordion("How the model was tested", open=True):
        gr.Markdown(
            """
            - **Data:** H&E whole slide images from 218 GI-NET cases (146 G1, 52 G2, 20 G3)
              from a single institution.
            - **Processing:** each slide was tiled at 40× magnification into 1024×1024-pixel
              tiles (833,237 tiles in total). Features were extracted with H-optimus-0, and an
              attention-based multiple instance learning (ABMIL) model combined the tiles of
              each case into one case-level prediction.
            - **Evaluation:** cases were split into training/validation (174; 80%) and a held-out
              test set (44; 20%). The performance above is case-level, on those 44 test cases.
            - **Limitations:** no external validation, limited numbers of higher-grade tumors,
              and a single-institution dataset; staining and scanning differences can limit
              generalizability.

            **Images uploaded here were not part of that evaluation.** This app processes an
            upload based on its size. An image of 1024×1024 pixels or smaller is analyzed as a
            single tile. A larger image is cut into 1024×1024 tiles, tiles with less than 30%
            tissue are skipped, and up to 500 tiles (chosen at random when there are more) are
            combined into one prediction. Uploading images of different sizes, magnifications, or
            regions from the same case may lead to different results.
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload H&E Image",
                sources=["upload", "clipboard"],
                height=400,
            )
            predict_btn = gr.Button("🔍 Predict Grade", variant="primary", size="lg")

        with gr.Column(scale=1):
            prediction_output = gr.Markdown(label="Prediction")
            probability_output = gr.Markdown(label="Probabilities")

    interpretation_output = gr.Markdown(label="Interpretation")

    # Connect button to prediction function
    predict_btn.click(
        fn=predict_grade,
        inputs=[image_input],
        outputs=[prediction_output, probability_output, interpretation_output],
    )

    # Also trigger prediction on image upload
    image_input.upload(
        fn=predict_grade,
        inputs=[image_input],
        outputs=[prediction_output, probability_output, interpretation_output],
    )

    gr.Markdown(
        """
        ---

        <div class="disclaimer">

        **⚠️ Disclaimer: for research and testing use only. Not for clinical use.**
        This tool is not a medical device and has not been validated or approved for
        diagnosis, grading, or treatment decisions. Ki-67 grading must be performed by a
        qualified pathologist using standard methods, including Ki-67 immunohistochemistry.

        </div>

        ---

        **About:** This application uses an Attention-Based Multiple Instance Learning (ABMIL)
        model with H-optimus-0 feature extraction to predict Ki-67 grade from H&E images.
        The model was trained on GI-NET histopathology data using 5-fold cross-validation.
        """
    )

# For Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()
