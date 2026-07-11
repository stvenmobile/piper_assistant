#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time

class PiperSketcher:
    def __init__(self):
        # Establish the absolute path anchor relative to this file's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.workspace_dir = os.path.join(base_dir, "assets", "sketchbook")
        
        # Ensure the entire directory tree exists safely
        os.makedirs(self.workspace_dir, exist_ok=True)
        # print(f"🎨 PiperSketcher Core Initialized. Workspace bound to: {self.workspace_dir}")

    def sketch_from_frame(self, cv_frame, description="Reality Snapshot"):
        """
        Processes an incoming OpenCV frame matrix, extracts clean physical boundaries,
        and saves a high-contrast ink-on-paper sketch asset.
        """
        if cv_frame is None or cv_frame.size == 0:
            print("ERROR: PiperSketcher received an empty or corrupt frame matrix.")
            return None

        # 1. Simplify the color matrix to pure spatial intensity (Grayscale)
        gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Smooth out high-frequency sensor noise using a subtle Gaussian filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Apply Canny Edge Detection to calculate structural boundary gradients
        edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
        
        # 4. Invert the binary matrix so edges are black ink strokes on a clean white page
        sketch_canvas = cv2.bitwise_not(edges)
        
        # ✨ 5. Lock the asset away with a readable local timestamp index (e.g., sketch_20260706_173045.jpg)
        time_string = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        filename = f"sketch_{time_string}.jpg"
        filepath = os.path.join(self.workspace_dir, filename)
        
        success = cv2.imwrite(filepath, sketch_canvas)
        if success:
            # print(f"🎨 Sketch successfully rendered and committed: {filename} ({description})")
            return filename
        else:
            print(f"ERROR: Failed to write matrix payload to disk at: {filepath}")
            return None

    def apply_harmonic_distortion(self, input_filename, frequency=120, amplitude=15):
        """
        Skill Expansion: Takes an existing sketch asset and warps its lines 
        using a periodic sine wave to simulate abstract harmonic frequency fields.
        """
        input_path = os.path.join(self.workspace_dir, input_filename)
        if not os.path.exists(input_path):
            print(f"ERROR: Target asset '{input_filename}' missing from workspace.")
            return None

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        rows, cols = img.shape
        abstract_canvas = np.zeros(img.shape, dtype=np.uint8) + 255  # Solid white background

        # Mathematical transformation loop across the line pixel vectors
        for i in range(rows):
            # Calculate horizontal pixel displacement via a clean sine function
            shift = int(amplitude * np.sin(2 * np.pi * i / frequency))
            for j in range(cols):
                if img[i, j] < 128:  # Isolate active line pixels (black strokes)
                    new_j = (j + shift) % cols
                    abstract_canvas[i, new_j] = 0  # Re-project line onto abstract canvas

        # ✨ Readable local timestamp index for abstract transformations
        time_string = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        filename = f"abstract_{time_string}.jpg"
        filepath = os.path.join(self.workspace_dir, filename)
        cv2.imwrite(filepath, abstract_canvas)
        
        print(f"🌀 Abstract transformation completed: {filename}")
        return filename