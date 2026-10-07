"""
demo.py

DermaVision desktop demonstration application.

Features:
- EfficientNet-B0 inference
- Temperature-calibrated probabilities
- Image quality assessment
- Grad-CAM explanation
- CustomTkinter desktop UI

This is an AI-assisted screening/research prototype.
It is NOT a medical diagnosis.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from predict import (
    load_model,
    predict_image,
    load_calibration,
    preprocess_image,
)

from gradcam import generate_gradcam


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pth",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
)

PREDICTION_THRESHOLD = 0.5

GRADCAM_TARGET_NAME = (
    "EfficientNet-B0 • blocks[-1]"
)


# ============================================================
# APPEARANCE
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# COLORS
# ============================================================

BG = "#080D16"
HEADER = "#0B1220"

SURFACE = "#101826"
SURFACE_2 = "#141F30"
SURFACE_3 = "#182437"

BORDER = "#243247"

TEXT = "#F3F7FC"
TEXT_SECONDARY = "#A7B3C4"
TEXT_MUTED = "#657286"

ACCENT = "#38BDF8"
ACCENT_DARK = "#1678B8"

SUCCESS = "#34D399"
SUCCESS_DARK = "#103B32"

DANGER = "#F87171"
DANGER_DARK = "#431F27"

WARNING = "#FBBF24"
WARNING_DARK = "#40351B"


# ============================================================
# APPLICATION
# ============================================================

class SkinLesionApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(
            "DermaVision — Skin Lesion Screening"
        )

        self.geometry("1280x850")
        self.minsize(1080, 720)

        self.configure(
            fg_color=BG
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        try:

            self.model, self.device = load_model(
                MODEL_PATH
            )

        except Exception as e:

            messagebox.showerror(
                "Model Error",
                f"Could not load the trained model:\n\n{e}",
            )

            self.destroy()
            return

        # ----------------------------------------------------
        # LOAD CALIBRATION
        # ----------------------------------------------------

        self.temperature, self.calibration_available = (
            load_calibration()
        )

        if not self.calibration_available:
            self.temperature = None

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.current_image = None
        self.current_gradcam = None

        self.selected_filename = (
            "No image selected"
        )

        self.current_prediction = None
        self.current_confidence = None
        self.current_malignant_probability = None

        self.threshold_used = (
            PREDICTION_THRESHOLD
        )

        # ----------------------------------------------------
        # BUILD UI
        # ----------------------------------------------------

        self.build_ui()


    # ========================================================
    # UI HELPERS
    # ========================================================

    def section_label(
        self,
        parent,
        text,
        color=TEXT_MUTED,
    ):

        return ctk.CTkLabel(
            parent,
            text=text.upper(),
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=color,
        )


    def create_metric(
        self,
        parent,
        title,
        value,
        row,
    ):

        frame = ctk.CTkFrame(
            parent,
            fg_color="transparent",
        )

        frame.grid(
            row=row,
            column=0,
            sticky="ew",
            pady=5,
        )

        frame.grid_columnconfigure(
            0,
            weight=1,
        )

        ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=TEXT_MUTED,
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        value_label = ctk.CTkLabel(
            frame,
            text=value,
            font=ctk.CTkFont(
                size=12,
                weight="bold",
            ),
            text_color=TEXT,
        )

        value_label.grid(
            row=0,
            column=1,
            sticky="e",
        )

        return value_label


    # ========================================================
    # BUILD UI
    # ========================================================

    def build_ui(self):

        self.grid_columnconfigure(
            0,
            weight=1,
        )

        self.grid_rowconfigure(
            1,
            weight=1,
        )

        # ====================================================
        # HEADER
        # ====================================================

        header = ctk.CTkFrame(
            self,
            height=86,
            corner_radius=0,
            fg_color=HEADER,
        )

        header.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        header.grid_columnconfigure(
            1,
            weight=1,
        )

        # Logo mark

        ctk.CTkLabel(
            header,
            text="◉",
            font=ctk.CTkFont(
                size=28,
                weight="bold",
            ),
            text_color=ACCENT,
        ).grid(
            row=0,
            column=0,
            padx=(30, 9),
            pady=22,
        )

        # Brand

        brand_frame = ctk.CTkFrame(
            header,
            fg_color="transparent",
        )

        brand_frame.grid(
            row=0,
            column=1,
            sticky="w",
            pady=16,
        )

        ctk.CTkLabel(
            brand_frame,
            text="DERMAVISION",
            font=ctk.CTkFont(
                size=24,
                weight="bold",
            ),
            text_color=TEXT,
        ).pack(
            anchor="w",
        )

        ctk.CTkLabel(
            brand_frame,
            text="AI-assisted skin lesion screening",
            font=ctk.CTkFont(
                size=11,
            ),
            text_color=TEXT_SECONDARY,
        ).pack(
            anchor="w",
        )

        # Calibration status

        calibration_text = (
            "●  CALIBRATED"
            if self.calibration_available
            else "●  UNCALIBRATED"
        )

        calibration_color = (
            SUCCESS
            if self.calibration_available
            else WARNING
        )

        calibration_bg = (
            SUCCESS_DARK
            if self.calibration_available
            else WARNING_DARK
        )

        status = ctk.CTkFrame(
            header,
            fg_color=calibration_bg,
            corner_radius=18,
        )

        status.grid(
            row=0,
            column=2,
            padx=(10, 30),
            pady=24,
        )

        ctk.CTkLabel(
            status,
            text=calibration_text,
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=calibration_color,
        ).pack(
            padx=15,
            pady=7,
        )

        # ====================================================
        # MAIN
        # ====================================================

        main = ctk.CTkFrame(
            self,
            fg_color="transparent",
        )

        main.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=28,
            pady=24,
        )

        main.grid_columnconfigure(
            0,
            weight=6,
        )

        main.grid_columnconfigure(
            1,
            weight=4,
        )

        main.grid_rowconfigure(
            0,
            weight=1,
        )

        # ====================================================
        # LEFT
        # ====================================================

        left = ctk.CTkFrame(
            main,
            fg_color="transparent",
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 18),
        )

        left.grid_columnconfigure(
            0,
            weight=1,
        )

        left.grid_rowconfigure(
            1,
            weight=1,
        )

        # Section heading

        heading = ctk.CTkFrame(
            left,
            fg_color="transparent",
        )

        heading.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 9),
        )

        heading.grid_columnconfigure(
            0,
            weight=1,
        )

        ctk.CTkLabel(
            heading,
            text="LESION ANALYSIS",
            font=ctk.CTkFont(
                size=13,
                weight="bold",
            ),
            text_color=TEXT,
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ctk.CTkLabel(
            heading,
            text="Original image + model explanation",
            font=ctk.CTkFont(
                size=10,
            ),
            text_color=TEXT_MUTED,
        ).grid(
            row=0,
            column=1,
            sticky="e",
        )

        # ====================================================
        # VISUAL AREA
        # ====================================================

        visual_area = ctk.CTkFrame(
            left,
            fg_color=SURFACE,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )

        visual_area.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        visual_area.grid_columnconfigure(
            0,
            weight=1,
        )

        visual_area.grid_rowconfigure(
            0,
            weight=3,
        )

        visual_area.grid_rowconfigure(
            1,
            weight=2,
        )

        # ----------------------------------------------------
        # ORIGINAL IMAGE
        # ----------------------------------------------------

        original_container = ctk.CTkFrame(
            visual_area,
            fg_color="#060B14",
            corner_radius=18,
        )

        original_container.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=14,
            pady=(14, 7),
        )

        self.image_display = ctk.CTkLabel(
            original_container,
            text=(
                "SELECT A LESION IMAGE\n\n"
                "JPG / JPEG / PNG"
            ),
            font=ctk.CTkFont(
                size=16,
                weight="bold",
            ),
            text_color=TEXT_MUTED,
        )

        self.image_display.pack(
            expand=True,
            fill="both",
            padx=10,
            pady=10,
        )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        cam_container = ctk.CTkFrame(
            visual_area,
            fg_color="#060B14",
            corner_radius=18,
        )

        cam_container.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=14,
            pady=(7, 14),
        )

        cam_container.grid_columnconfigure(
            0,
            weight=1,
        )

        cam_container.grid_rowconfigure(
            1,
            weight=1,
        )

        cam_header = ctk.CTkFrame(
            cam_container,
            fg_color="transparent",
        )

        cam_header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=16,
            pady=(11, 4),
        )

        cam_header.grid_columnconfigure(
            0,
            weight=1,
        )

        ctk.CTkLabel(
            cam_header,
            text="MODEL EXPLANATION  •  GRAD-CAM",
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=ACCENT,
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ctk.CTkLabel(
            cam_header,
            text=GRADCAM_TARGET_NAME,
            font=ctk.CTkFont(
                size=9,
            ),
            text_color=TEXT_MUTED,
        ).grid(
            row=0,
            column=1,
            sticky="e",
        )

        self.gradcam_display = ctk.CTkLabel(
            cam_container,
            text=(
                "Grad-CAM explanation will appear here\n"
                "after analysis"
            ),
            font=ctk.CTkFont(
                size=12,
            ),
            text_color=TEXT_MUTED,
        )

        self.gradcam_display.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=12,
            pady=(0, 12),
        )

        # ----------------------------------------------------
        # IMAGE FOOTER
        # ----------------------------------------------------

        image_footer = ctk.CTkFrame(
            left,
            fg_color="transparent",
        )

        image_footer.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

        image_footer.grid_columnconfigure(
            0,
            weight=1,
        )

        self.filename_label = ctk.CTkLabel(
            image_footer,
            text="No image selected",
            font=ctk.CTkFont(
                size=12,
                weight="bold",
            ),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )

        self.filename_label.grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.analysis_status_label = ctk.CTkLabel(
            image_footer,
            text="Ready for analysis",
            font=ctk.CTkFont(
                size=10,
            ),
            text_color=TEXT_MUTED,
            anchor="w",
        )

        self.analysis_status_label.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(3, 0),
        )

        self.select_button = ctk.CTkButton(
            image_footer,
            text="＋  Select Lesion Image",
            height=44,
            width=215,
            corner_radius=13,
            font=ctk.CTkFont(
                size=12,
                weight="bold",
            ),
            fg_color=ACCENT_DARK,
            hover_color="#2089C9",
            command=self.select_image,
        )

        self.select_button.grid(
            row=0,
            column=1,
            rowspan=2,
            padx=(20, 0),
        )

        # ====================================================
        # RIGHT
        # ====================================================

        right = ctk.CTkFrame(
            main,
            fg_color="transparent",
        )

        right.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        right.grid_columnconfigure(
            0,
            weight=1,
        )

        # ====================================================
        # RESULT HEADER
        # ====================================================

        result_header = ctk.CTkFrame(
            right,
            fg_color="transparent",
        )

        result_header.pack(
            fill="x",
            pady=(0, 9),
        )

        ctk.CTkLabel(
            result_header,
            text="SCREENING RESULT",
            font=ctk.CTkFont(
                size=13,
                weight="bold",
            ),
            text_color=TEXT,
        ).pack(
            side="left",
        )

        self.result_indicator = ctk.CTkLabel(
            result_header,
            text="WAITING",
            font=ctk.CTkFont(
                size=9,
                weight="bold",
            ),
            text_color=TEXT_MUTED,
        )

        self.result_indicator.pack(
            side="right",
        )

        # ====================================================
        # PRIMARY RESULT
        # ====================================================

        result_area = ctk.CTkFrame(
            right,
            fg_color=SURFACE,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )

        result_area.pack(
            fill="x",
        )

        self.prediction_label = ctk.CTkLabel(
            result_area,
            text="—",
            font=ctk.CTkFont(
                size=39,
                weight="bold",
            ),
            text_color=TEXT,
        )

        self.prediction_label.pack(
            pady=(25, 2),
        )

        self.prediction_subtitle = ctk.CTkLabel(
            result_area,
            text="Select an image to begin analysis",
            font=ctk.CTkFont(
                size=11,
            ),
            text_color=TEXT_MUTED,
        )

        self.prediction_subtitle.pack(
            pady=(0, 24),
        )

        # ====================================================
        # METRICS
        # ====================================================

        metrics = ctk.CTkFrame(
            right,
            fg_color="transparent",
        )

        metrics.pack(
            fill="x",
            pady=17,
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence_top = ctk.CTkFrame(
            metrics,
            fg_color="transparent",
        )

        confidence_top.pack(
            fill="x",
        )

        ctk.CTkLabel(
            confidence_top,
            text="CALIBRATED CONFIDENCE",
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=TEXT_MUTED,
        ).pack(
            side="left",
        )

        self.confidence_label = ctk.CTkLabel(
            confidence_top,
            text="—",
            font=ctk.CTkFont(
                size=18,
                weight="bold",
            ),
            text_color=TEXT,
        )

        self.confidence_label.pack(
            side="right",
        )

        self.confidence_bar = ctk.CTkProgressBar(
            metrics,
            height=7,
            corner_radius=4,
            fg_color="#202C3C",
            progress_color=ACCENT,
        )

        self.confidence_bar.pack(
            fill="x",
            pady=(7, 15),
        )

        self.confidence_bar.set(0)

        # ----------------------------------------------------
        # MALIGNANT PROBABILITY
        # ----------------------------------------------------

        malignant_top = ctk.CTkFrame(
            metrics,
            fg_color="transparent",
        )

        malignant_top.pack(
            fill="x",
        )

        ctk.CTkLabel(
            malignant_top,
            text="MALIGNANT PROBABILITY",
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=TEXT_MUTED,
        ).pack(
            side="left",
        )

        self.malignant_label = ctk.CTkLabel(
            malignant_top,
            text="—",
            font=ctk.CTkFont(
                size=18,
                weight="bold",
            ),
            text_color=TEXT,
        )

        self.malignant_label.pack(
            side="right",
        )

        self.malignant_bar = ctk.CTkProgressBar(
            metrics,
            height=7,
            corner_radius=4,
            fg_color="#202C3C",
            progress_color=DANGER,
        )

        self.malignant_bar.pack(
            fill="x",
            pady=(7, 0),
        )

        self.malignant_bar.set(0)

        # ====================================================
        # QUALITY
        # ====================================================

        quality_header = ctk.CTkFrame(
            right,
            fg_color="transparent",
        )

        quality_header.pack(
            fill="x",
            pady=(2, 7),
        )

        ctk.CTkLabel(
            quality_header,
            text="IMAGE QUALITY",
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=ACCENT,
        ).pack(
            side="left",
        )

        self.quality_status_label = ctk.CTkLabel(
            quality_header,
            text="—",
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=TEXT_MUTED,
        )

        self.quality_status_label.pack(
            side="right",
        )

        quality_frame = ctk.CTkFrame(
            right,
            fg_color=SURFACE_2,
            corner_radius=18,
        )

        quality_frame.pack(
            fill="x",
        )

        quality_frame.grid_columnconfigure(
            0,
            weight=1,
        )

        self.quality_score_label = self.create_metric(
            quality_frame,
            "QUALITY SCORE",
            "—",
            0,
        )

        self.resolution_label = self.create_metric(
            quality_frame,
            "RESOLUTION",
            "—",
            1,
        )

        self.sharpness_label = self.create_metric(
            quality_frame,
            "SHARPNESS",
            "—",
            2,
        )

        self.contrast_label = self.create_metric(
            quality_frame,
            "CONTRAST",
            "—",
            3,
        )

        self.quality_warning_label = ctk.CTkLabel(
            quality_frame,
            text="",
            font=ctk.CTkFont(
                size=10,
            ),
            text_color=WARNING,
            anchor="w",
            justify="left",
            wraplength=350,
        )

        self.quality_warning_label.grid(
            row=4,
            column=0,
            sticky="w",
            padx=5,
            pady=(8, 8),
        )

        # ====================================================
        # SCREENING SUMMARY
        # ====================================================

        ctk.CTkLabel(
            right,
            text="SCREENING SUMMARY",
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=ACCENT,
        ).pack(
            anchor="w",
            pady=(16, 7),
        )

        self.summary_frame = ctk.CTkFrame(
            right,
            fg_color=SURFACE_2,
            corner_radius=18,
        )

        self.summary_frame.pack(
            fill="x",
        )

        self.summary_label = ctk.CTkLabel(
            self.summary_frame,
            text="Awaiting image analysis",
            font=ctk.CTkFont(
                size=14,
                weight="bold",
            ),
            text_color=TEXT,
            anchor="w",
        )

        self.summary_label.pack(
            anchor="w",
            padx=17,
            pady=(15, 3),
        )

        self.recommendation_label = ctk.CTkLabel(
            self.summary_frame,
            text=(
                "Select a lesion image to generate "
                "a screening result."
            ),
            font=ctk.CTkFont(
                size=10,
            ),
            text_color=TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=360,
        )

        self.recommendation_label.pack(
            anchor="w",
            padx=17,
            pady=(0, 15),
        )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        ctk.CTkLabel(
            right,
            text="MODEL INFORMATION",
            font=ctk.CTkFont(
                size=10,
                weight="bold",
            ),
            text_color=ACCENT,
        ).pack(
            anchor="w",
            pady=(15, 6),
        )

        self.model_info_label = ctk.CTkLabel(
            right,
            text=(
                "EfficientNet-B0\n"
                "~23,000 training images\n"
                "224 × 224 RGB input\n"
                "Benign / Malignant classification\n"
                "Temperature scaling: "
                + (
                    f"T={self.temperature:.4f}\n"
                    if self.temperature is not None
                    else "unavailable\n"
                )
                + f"Grad-CAM: {GRADCAM_TARGET_NAME}\n"
                f"Decision threshold: "
                f"{PREDICTION_THRESHOLD * 100:.0f}%"
            ),
            font=ctk.CTkFont(
                size=10,
            ),
            text_color=TEXT_MUTED,
            justify="left",
            anchor="w",
        )

        self.model_info_label.pack(
            anchor="w",
        )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        ctk.CTkLabel(
            right,
            text=(
                "AI-assisted screening only  •  "
                "Not a medical diagnosis"
            ),
            font=ctk.CTkFont(
                size=9,
            ),
            text_color=TEXT_MUTED,
        ).pack(
            anchor="w",
            pady=(14, 0),
        )


    # ========================================================
    # DISPLAY ORIGINAL IMAGE
    # ========================================================

    def display_original_image(self, path):

        image = Image.open(
            path
        ).convert("RGB")

        image.thumbnail(
            (610, 390),
            Image.Resampling.LANCZOS,
        )

        self.current_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size,
        )

        self.image_display.configure(
            image=self.current_image,
            text="",
        )


    # ========================================================
    # DISPLAY GRAD-CAM
    # ========================================================

    def display_gradcam(
        self,
        visualization,
    ):

        image = Image.fromarray(
            visualization
        ).convert("RGB")

        image.thumbnail(
            (610, 225),
            Image.Resampling.LANCZOS,
        )

        self.current_gradcam = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size,
        )

        self.gradcam_display.configure(
            image=self.current_gradcam,
            text="",
        )


    # ========================================================
    # GENERATE GRAD-CAM
    # ========================================================

    def generate_explanation(
        self,
        path,
        predicted_class,
    ):

        try:

            input_tensor = preprocess_image(
                path
            ).to(self.device)

            visualization = generate_gradcam(
                model=self.model,
                input_tensor=input_tensor,
                target_layer=self.model.blocks[-1],
                target_class=predicted_class,
            )

            self.display_gradcam(
                visualization
            )

            return True

        except Exception as e:

            print(
                f"Grad-CAM generation failed: {e}"
            )

            self.gradcam_display.configure(
                image=None,
                text=(
                    "MODEL EXPLANATION\n\n"
                    "Grad-CAM could not be generated.\n"
                    "Prediction is still available."
                ),
                text_color=TEXT_MUTED,
            )

            return False


    # ========================================================
    # UPDATE QUALITY
    # ========================================================

    def update_quality_display(
        self,
        result,
    ):

        score = result[
            "quality_score"
        ]

        status = result[
            "quality_status"
        ]

        details = result[
            "quality_details"
        ]

        warnings = result[
            "quality_warnings"
        ]

        self.quality_score_label.configure(
            text=f"{score * 100:.1f}%"
        )

        self.resolution_label.configure(
            text=(
                f"{details['width']} × "
                f"{details['height']}"
            )
        )

        self.sharpness_label.configure(
            text=f"{details['sharpness']:.2f}"
        )

        self.contrast_label.configure(
            text=f"{details['contrast']:.2f}"
        )

        self.quality_status_label.configure(
            text=status
        )

        if status == "Good":

            self.quality_status_label.configure(
                text_color=SUCCESS
            )

        elif status == "Acceptable":

            self.quality_status_label.configure(
                text_color=WARNING
            )

        else:

            self.quality_status_label.configure(
                text_color=DANGER
            )

        if warnings:

            warning_text = (
                "⚠ "
                + "\n⚠ ".join(warnings)
            )

            self.quality_warning_label.configure(
                text=warning_text,
                text_color=WARNING,
            )

        else:

            self.quality_warning_label.configure(
                text="✓ No quality warnings",
                text_color=SUCCESS,
            )


    # ========================================================
    # SELECT IMAGE
    # ========================================================

    def select_image(self):

        path = filedialog.askopenfilename(
            title="Select Skin Lesion Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png",
                ),
                (
                    "All Files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        try:

            # ------------------------------------------------
            # UI STATE
            # ------------------------------------------------

            self.filename_label.configure(
                text=os.path.basename(path)
            )

            self.analysis_status_label.configure(
                text="Analyzing image...",
                text_color=ACCENT,
            )

            self.result_indicator.configure(
                text="ANALYZING",
                text_color=ACCENT,
            )

            self.select_button.configure(
                state="disabled",
                text="Analyzing...",
            )

            self.update_idletasks()

            # ------------------------------------------------
            # DISPLAY ORIGINAL
            # ------------------------------------------------

            self.display_original_image(
                path
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            result = predict_image(
                image_path=path,
                model=self.model,
                device=self.device,
                threshold=PREDICTION_THRESHOLD,
                calibration_temperature=(
                    self.temperature
                ),
            )

            predicted_class_name = result[
                "predicted_class"
            ]

            confidence = result[
                "confidence"
            ]

            malignant_probability = result[
                "malignant_probability"
            ]

            # ------------------------------------------------
            # STATE
            # ------------------------------------------------

            self.threshold_used = result[
                "threshold_used"
            ]

            self.current_prediction = (
                predicted_class_name
            )

            self.current_confidence = (
                confidence
            )

            self.current_malignant_probability = (
                malignant_probability
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if predicted_class_name == "Malignant":

                self.prediction_label.configure(
                    text="MALIGNANT",
                    text_color=DANGER,
                )

                self.prediction_subtitle.configure(
                    text=(
                        "Model classified the image "
                        "as malignant"
                    ),
                    text_color=DANGER,
                )

                self.result_indicator.configure(
                    text="ELEVATED RISK",
                    text_color=DANGER,
                )

            else:

                self.prediction_label.configure(
                    text="BENIGN",
                    text_color=SUCCESS,
                )

                self.prediction_subtitle.configure(
                    text=(
                        "Model classified the image "
                        "as benign"
                    ),
                    text_color=SUCCESS,
                )

                self.result_indicator.configure(
                    text="LOWER ESTIMATED MALIGNANCY",
                    text_color=SUCCESS,
                )

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            self.confidence_label.configure(
                text=f"{confidence * 100:.1f}%"
            )

            self.confidence_bar.set(
                confidence
            )

            # ------------------------------------------------
            # MALIGNANT PROBABILITY
            # ------------------------------------------------

            self.malignant_label.configure(
                text=(
                    f"{malignant_probability * 100:.1f}%"
                )
            )

            self.malignant_bar.set(
                malignant_probability
            )

            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            self.update_quality_display(
                result
            )

            # ------------------------------------------------
            # SCREENING SUMMARY
            # ------------------------------------------------

            if malignant_probability < 0.30:

                summary = (
                    "Lower malignant probability"
                )

                recommendation = (
                    "The model assigns a lower probability "
                    "to the malignant class. This result is "
                    "for AI-assisted screening only and does "
                    "not exclude clinical evaluation."
                )

            elif malignant_probability < 0.50:

                summary = (
                    "Below decision threshold"
                )

                recommendation = (
                    "The malignant probability is below the "
                    "current 0.50 decision threshold. The "
                    "result should not be interpreted as "
                    "clinical exclusion."
                )

            elif malignant_probability < 0.70:

                summary = (
                    "Elevated malignant probability"
                )

                recommendation = (
                    "The model assigns an elevated probability "
                    "to the malignant class. Professional "
                    "clinical evaluation is recommended."
                )

            else:

                summary = (
                    "High malignant probability"
                )

                recommendation = (
                    "The model assigns a high probability "
                    "to the malignant class. Professional "
                    "clinical evaluation is recommended."
                )

            self.summary_label.configure(
                text=summary
            )

            self.recommendation_label.configure(
                text=recommendation
            )

            # ------------------------------------------------
            # MODEL INFORMATION
            # ------------------------------------------------

            calibration_text = (
                f"Temperature scaling: "
                f"T={result['calibration_temperature']:.4f}"
                if result["calibration_applied"]
                else
                "Temperature scaling: unavailable"
            )

            self.model_info_label.configure(
                text=(
                    "EfficientNet-B0\n"
                    "~23,000 training images\n"
                    "224 × 224 RGB input\n"
                    "Benign / Malignant classification\n"
                    f"{calibration_text}\n"
                    f"Grad-CAM: {GRADCAM_TARGET_NAME}\n"
                    f"Decision threshold: "
                    f"{self.threshold_used * 100:.0f}%"
                )
            )

            # ------------------------------------------------
            # GRAD-CAM
            # ------------------------------------------------

            self.analysis_status_label.configure(
                text="Generating model explanation...",
                text_color=ACCENT,
            )

            self.update_idletasks()

            predicted_class_index = (
                1
                if predicted_class_name == "Malignant"
                else 0
            )

            gradcam_success = (
                self.generate_explanation(
                    path,
                    predicted_class_index,
                )
            )

            # ------------------------------------------------
            # FINAL STATUS
            # ------------------------------------------------

            if gradcam_success:

                self.analysis_status_label.configure(
                    text=(
                        "✓ Analysis complete • "
                        "Quality + calibration + "
                        "explanation ready"
                    ),
                    text_color=SUCCESS,
                )

            else:

                self.analysis_status_label.configure(
                    text=(
                        "✓ Analysis complete • "
                        "Explanation unavailable"
                    ),
                    text_color=WARNING,
                )

            self.select_button.configure(
                state="normal",
                text="＋  Select Lesion Image",
            )

        except Exception as e:

            print(
                f"Analysis error: {e}"
            )

            self.select_button.configure(
                state="normal",
                text="＋  Select Lesion Image",
            )

            self.result_indicator.configure(
                text="ERROR",
                text_color=DANGER,
            )

            self.analysis_status_label.configure(
                text="Analysis could not be completed",
                text_color=DANGER,
            )

            messagebox.showerror(
                "Prediction Error",
                f"Could not process the image:\n\n{e}",
            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app = SkinLesionApp()

    app.mainloop()