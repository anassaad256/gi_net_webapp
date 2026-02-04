"""
GI-NET Ki-67 Grade Prediction Web Application.

This Gradio-based web application allows users to upload H&E histopathology images
and receive AI-powered predictions of Ki-67 proliferation grade (G1 vs G2+G3).
"""

import os

import gradio as gr

from predictor import GINETPredictor

# Configuration
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/ABMIL_binary")
DEVICE = os.environ.get("DEVICE", "cuda")

# Global predictor instance (lazy loaded)
predictor = None


def load_model() -> GINETPredictor:
    """Load the predictor model (lazy initialization)."""
    global predictor
    if predictor is None:
        predictor = GINETPredictor(MODEL_DIR, device=DEVICE)
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
        result_text = f"**{pred['prediction']}** (Confidence: {pred['confidence']*100:.1f}%)"

        prob_text = f"""
| Grade | Probability |
|-------|-------------|
| G1 (Ki-67 <3%) | {pred['prob_g1']*100:.1f}% |
| G2+G3 (Ki-67 ≥3%) | {pred['prob_g2g3']*100:.1f}% |

*{pred['n_tiles']} tissue region(s) analyzed*
"""

        return result_text, prob_text, pred["interpretation"]

    except FileNotFoundError as e:
        return f"Model Error: {str(e)}", "", ""
    except Exception as e:
        return f"Error: {str(e)}", "", ""


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""
    with gr.Blocks(
        title="GI-NET Ki-67 Grade Prediction",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
        # GI-NET Ki-67 Grade Prediction from H&E

        Upload an H&E histopathology image to predict Ki-67 proliferation grade.

        **Supported grades:**
        - **G1**: Ki-67 <3% (low proliferation)
        - **G2+G3**: Ki-67 ≥3% (intermediate/high proliferation)

        **Model Performance:** 94.9% accuracy, 0.90 kappa
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload H&E Image",
                    sources=["upload", "clipboard"],
                )
                predict_btn = gr.Button("Predict Grade", variant="primary")

            with gr.Column(scale=1):
                prediction_output = gr.Markdown(label="Prediction")
                probability_output = gr.Markdown(label="Probabilities")

        interpretation_output = gr.Markdown(label="Clinical Interpretation")

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
        **Disclaimer:** This tool is for research and clinical decision support only.
        Not intended for primary diagnosis. Always correlate with clinical findings and
        consider Ki-67 IHC when clinically indicated.
        """
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=True,  # Creates a public link for sharing
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
    )
