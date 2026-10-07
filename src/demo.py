"""
demo.py — Interactive desktop GUI for DermaVision.
NOTE: This module requires `customtkinter` and cannot be run in headless
environments like Kaggle Notebooks. It is strictly for local deployment.
"""

import os

from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from predict import load_model, predict_image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
PREDICTION_THRESHOLD = 0.5

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# --------------------------------------------------
# Application
# --------------------------------------------------

class SkinLesionApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("DermaVision — Skin Lesion Screening")
        self.geometry("1100x720")
        self.minsize(950, 650)

        # Load model
        try:
            self.model, self.device = load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror(
                "Model Error",
                f"Could not load the trained model:\n\n{e}"
            )
            self.destroy()
            return

        self.current_image = None
        self.selected_filename = "No image selected"
        self.threshold_used = PREDICTION_THRESHOLD

        self.build_ui()

    # --------------------------------------------------
    # UI
    # --------------------------------------------------

    def build_ui(self):

        # Main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---------------- HEADER ----------------

        header = ctk.CTkFrame(
            self,
            height=80,
            corner_radius=0,
            fg_color="#111827"
        )
        header.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        header.grid_columnconfigure(1, weight=1)

        logo = ctk.CTkLabel(
            header,
            text="◉  DERMAVISION",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#38BDF8"
        )
        logo.grid(
            row=0,
            column=0,
            padx=30,
            pady=20
        )

        subtitle = ctk.CTkLabel(
            header,
            text="AI-assisted skin lesion screening",
            font=ctk.CTkFont(size=14),
            text_color="#9CA3AF"
        )
        subtitle.grid(
            row=0,
            column=1,
            sticky="w",
            padx=10
        )

        model_status = ctk.CTkLabel(
            header,
            text="● MODEL READY",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#34D399"
        )
        model_status.grid(
            row=0,
            column=2,
            padx=30
        )

        # ---------------- CONTENT ----------------

        content = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )
        content.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=25,
            pady=25
        )

        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # ---------------- LEFT PANEL ----------------

        image_card = ctk.CTkFrame(
            content,
            corner_radius=18,
            fg_color="#1F2937"
        )
        image_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 12)
        )

        image_card.grid_columnconfigure(0, weight=1)

        image_title = ctk.CTkLabel(
            image_card,
            text="LESION IMAGE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#9CA3AF"
        )
        image_title.pack(
            anchor="w",
            padx=25,
            pady=(22, 10)
        )

        self.image_display = ctk.CTkLabel(
            image_card,
            text="No image selected\n\nUpload an ISIC lesion image",
            font=ctk.CTkFont(size=16),
            text_color="#6B7280",
            fg_color="#111827",
            corner_radius=15
        )
        self.image_display.pack(
            expand=True,
            fill="both",
            padx=25,
            pady=10
        )

        image_info = ctk.CTkFrame(
            image_card,
            fg_color="transparent"
        )
        image_info.pack(
            fill="x",
            padx=25,
            pady=(2, 0)
        )

        self.filename_label = ctk.CTkLabel(
            image_info,
            text=self.selected_filename,
            anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#E5E7EB"
        )
        self.filename_label.pack(
            anchor="w"
        )

        self.analysis_status_label = ctk.CTkLabel(
            image_info,
            text="Select an image to begin analysis",
            anchor="w",
            font=ctk.CTkFont(size=12),
            text_color="#6B7280"
        )
        self.analysis_status_label.pack(
            anchor="w",
            pady=(3, 0)
        )

        self.select_button = ctk.CTkButton(
            image_card,
            text="＋  Select Lesion Image",
            height=48,
            corner_radius=12,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.select_image
        )
        self.select_button.pack(
            fill="x",
            padx=25,
            pady=(15, 25)
        )

        # ---------------- RIGHT PANEL ----------------

        result_card = ctk.CTkFrame(
            content,
            corner_radius=18,
            fg_color="#1F2937"
        )
        result_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(12, 0)
        )

        result_card.grid_columnconfigure(0, weight=1)

        result_title = ctk.CTkLabel(
            result_card,
            text="ANALYSIS RESULT",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#9CA3AF"
        )
        result_title.pack(
            anchor="w",
            padx=30,
            pady=(25, 20)
        )

        # Prediction box
        self.prediction_box = ctk.CTkFrame(
            result_card,
            corner_radius=15,
            fg_color="#111827"
        )
        self.prediction_box.pack(
            fill="x",
            padx=30
        )

        self.prediction_label = ctk.CTkLabel(
            self.prediction_box,
            text="—",
            font=ctk.CTkFont(size=34, weight="bold"),
            text_color="#E5E7EB"
        )
        self.prediction_label.pack(pady=(25, 5))

        self.prediction_subtitle = ctk.CTkLabel(
            self.prediction_box,
            text="Select an image to begin analysis",
            font=ctk.CTkFont(size=13),
            text_color="#6B7280"
        )
        self.prediction_subtitle.pack(
            pady=(0, 25)
        )

        # Confidence
        confidence_title = ctk.CTkLabel(
            result_card,
            text="MODEL CONFIDENCE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#9CA3AF"
        )
        confidence_title.pack(
            anchor="w",
            padx=30,
            pady=(30, 5)
        )

        self.confidence_label = ctk.CTkLabel(
            result_card,
            text="—",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.confidence_label.pack(
            anchor="w",
            padx=30
        )

        self.confidence_bar = ctk.CTkProgressBar(
            result_card,
            height=12,
            corner_radius=6
        )
        self.confidence_bar.pack(
            fill="x",
            padx=30,
            pady=(8, 20)
        )
        self.confidence_bar.set(0)

        # Malignant probability
        malignant_title = ctk.CTkLabel(
            result_card,
            text="MALIGNANT PROBABILITY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#9CA3AF"
        )
        malignant_title.pack(
            anchor="w",
            padx=30,
            pady=(5, 5)
        )

        self.malignant_label = ctk.CTkLabel(
            result_card,
            text="—",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.malignant_label.pack(
            anchor="w",
            padx=30
        )

        self.malignant_bar = ctk.CTkProgressBar(
            result_card,
            height=12,
            corner_radius=6
        )
        self.malignant_bar.pack(
            fill="x",
            padx=30,
            pady=(8, 20)
        )
        self.malignant_bar.set(0)

        # ---------------- SCREENING SUMMARY ----------------

        summary_frame = ctk.CTkFrame(
            result_card,
            corner_radius=12,
            fg_color="#172235"
        )
        summary_frame.pack(
            fill="x",
            padx=30,
            pady=(2, 12)
        )

        summary_title = ctk.CTkLabel(
            summary_frame,
            text="SCREENING SUMMARY",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38BDF8"
        )
        summary_title.pack(
            anchor="w",
            padx=16,
            pady=(14, 5)
        )

        self.summary_label = ctk.CTkLabel(
            summary_frame,
            text="Select an image to view the screening summary",
            anchor="w",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#E5E7EB"
        )
        self.summary_label.pack(
            anchor="w",
            padx=16
        )

        self.recommendation_label = ctk.CTkLabel(
            summary_frame,
            text="Further clinical evaluation is recommended.",
            anchor="w",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF"
        )
        self.recommendation_label.pack(
            anchor="w",
            padx=16,
            pady=(4, 14)
        )

        # ---------------- MODEL INFORMATION ----------------

        model_info_title = ctk.CTkLabel(
            result_card,
            text="MODEL INFORMATION",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38BDF8"
        )
        model_info_title.pack(
            anchor="w",
            padx=30,
            pady=(4, 6)
        )

        self.model_info_label = ctk.CTkLabel(
            result_card,
            text=(
                "Architecture: EfficientNet-B0\n"
                "Input: 224 × 224 RGB\n"
                "Classes: Benign / Malignant\n"
                "Preprocessing: ImageNet normalization\n"
                f"Decision threshold: {self.threshold_used * 100:.0f}%"
            ),
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF"
        )
        self.model_info_label.pack(
            anchor="w",
            padx=30,
            pady=(0, 18)
        )

        # ---------------- FOOTER ----------------

        footer = ctk.CTkLabel(
            self,
            text=(
                "AI-assisted screening only • "
                "Not a medical diagnosis"
            ),
            font=ctk.CTkFont(size=12),
            text_color="#6B7280"
        )
        footer.grid(
            row=2,
            column=0,
            pady=(0, 15)
        )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def select_image(self):

        path = filedialog.askopenfilename(
            title="Select Skin Lesion Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        try:

            self.filename_label.configure(
                text=os.path.basename(path)
            )
            self.analysis_status_label.configure(
                text="Analyzing...",
                text_color="#38BDF8"
            )
            self.update_idletasks()

            # ---------------- DISPLAY IMAGE ----------------

            image = Image.open(path)

            # Keep aspect ratio
            image.thumbnail((450, 450))

            self.current_image = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=image.size
            )

            self.image_display.configure(
                image=self.current_image,
                text=""
            )

            # ---------------- MODEL ----------------

            result = predict_image(
                path,
                self.model,
                self.device,
                threshold=PREDICTION_THRESHOLD
            )

            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            malignant_probability = result["malignant_probability"]
            self.threshold_used = result["threshold_used"]

            # ---------------- RESULT ----------------

            if predicted_class == "Malignant":

                self.prediction_label.configure(
                    text="MALIGNANT",
                    text_color="#F87171"
                )

                self.prediction_subtitle.configure(
                    text="Elevated malignant probability",
                    text_color="#F87171"
                )

            else:

                self.prediction_label.configure(
                    text="BENIGN",
                    text_color="#34D399"
                )

                self.prediction_subtitle.configure(
                    text="No malignant classification detected",
                    text_color="#34D399"
                )

            # Confidence

            self.confidence_label.configure(
                text=f"{confidence * 100:.1f}%"
            )

            self.confidence_bar.set(confidence)

            # Malignant probability

            self.malignant_label.configure(
                text=f"{malignant_probability * 100:.1f}%"
            )

            self.malignant_bar.set(malignant_probability)

            if malignant_probability < 0.30:
                summary = "Lower malignant probability"
            elif malignant_probability < 0.50:
                summary = "Moderate malignant probability"
            elif malignant_probability < 0.70:
                summary = "Elevated malignant probability"
            else:
                summary = "High malignant probability"

            self.summary_label.configure(text=summary)
            self.model_info_label.configure(
                text=(
                    "Architecture: EfficientNet-B0\n"
                    "Input: 224 × 224 RGB\n"
                    "Classes: Benign / Malignant\n"
                    "Preprocessing: ImageNet normalization\n"
                    f"Decision threshold: {self.threshold_used * 100:.0f}%"
                )
            )
            self.analysis_status_label.configure(
                text="✓ Analysis complete",
                text_color="#34D399"
            )

        except Exception as e:

            self.analysis_status_label.configure(
                text="Analysis could not be completed",
                text_color="#F87171"
            )

            messagebox.showerror(
                "Prediction Error",
                f"Could not process the image:\n\n{e}"
            )


# --------------------------------------------------
# Start application
# --------------------------------------------------

if __name__ == "__main__":

    app = SkinLesionApp()
    app.mainloop()