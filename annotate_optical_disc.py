#!/usr/bin/env python3
"""
Interactive Optical Disc Annotation Tool

This script allows users to annotate optical discs on retinal images by clicking
to define the center and dragging to set the radius of a circular mask.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import glob


class OpticalDiscAnnotator:
    def __init__(self, images_dir="./images", output_dir="./discs"):
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.current_image_idx = 0
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Circle parameters
        self.center = None
        self.radius = 0
        self.drawing = False
        self.current_image = None
        self.current_mask = None

        if not self.image_files:
            print(f"No PNG images found in {images_dir}")
            return

        print(f"Found {len(self.image_files)} images to annotate")
        print("Instructions:")
        print("- Click to set the center of the optical disc")
        print("- Move mouse to adjust radius")
        print("- Press 'n' for next image, 'p' for previous")
        print("- Press 's' to save current annotation")
        print("- Press 'r' to reset current annotation")
        print("- Press 'q' to quit")

    def load_image(self, idx):
        """Load image at specified index"""
        if 0 <= idx < len(self.image_files):
            self.current_image_idx = idx
            image_path = self.image_files[idx]
            self.current_image = cv2.imread(image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

            # Load existing mask if it exists
            mask_name = os.path.splitext(os.path.basename(image_path))[0] + "_disc.png"
            mask_path = os.path.join(self.output_dir, mask_name)

            if os.path.exists(mask_path):
                self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Try to extract circle parameters from existing mask
                self.extract_circle_from_mask()
            else:
                self.current_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                self.center = None
                self.radius = 0

            return True
        return False

    def extract_circle_from_mask(self):
        """Extract circle parameters from existing mask using Hough Circle Transform"""
        if self.current_mask is None:
            return

        circles = cv2.HoughCircles(
            self.current_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=200
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) > 0:
                # Take the first circle found
                self.center = (circles[0][0], circles[0][1])
                self.radius = circles[0][2]

    def update_mask(self):
        """Update the binary mask with current circle parameters"""
        if self.center is not None and self.radius > 0:
            self.current_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
            cv2.circle(self.current_mask, self.center, self.radius, 255, -1)

    def save_mask(self):
        """Save current mask to file"""
        mask_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_idx]))[0] + "_disc.png"
        mask_path = os.path.join(self.output_dir, mask_name)
        cv2.imwrite(mask_path, self.current_mask)
        print(f"Saved mask: {mask_path}")

    def on_mouse_event(self, event):
        """Handle mouse events for circle drawing"""
        if event.inaxes != self.ax:
            return

        if event.name == 'button_press_event' and event.button == 1:
            # Left click - set center
            self.center = (int(event.xdata), int(event.ydata))
            self.drawing = True

        elif event.name == 'motion_notify_event' and self.drawing:
            # Mouse motion - update radius
            if self.center is not None:
                dx = event.xdata - self.center[0]
                dy = event.ydata - self.center[1]
                self.radius = int(np.sqrt(dx**2 + dy**2))
                self.update_mask()
                self.update_display()

        elif event.name == 'button_release_event' and event.button == 1:
            # Left release - stop drawing
            self.drawing = False

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'n':
            # Next image
            if self.current_image_idx < len(self.image_files) - 1:
                self.load_image(self.current_image_idx + 1)
                self.update_display()
        elif event.key == 'p':
            # Previous image
            if self.current_image_idx > 0:
                self.load_image(self.current_image_idx - 1)
                self.update_display()
        elif event.key == 's':
            # Save mask
            self.save_mask()
        elif event.key == 'r':
            # Reset current annotation
            self.center = None
            self.radius = 0
            self.current_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
            self.update_display()
        elif event.key == 'q':
            # Quit
            plt.close('all')

    def update_display(self):
        """Update the display with current image and mask"""
        self.ax.clear()

        # Display image
        self.ax.imshow(self.current_image)

        # Overlay mask with transparency
        if self.current_mask is not None:
            mask_colored = np.zeros((*self.current_mask.shape, 4))
            mask_colored[:, :, 0] = 1  # Red channel
            mask_colored[:, :, 3] = self.current_mask / 255.0 * 0.5  # Alpha channel
            self.ax.imshow(mask_colored)

        # Draw circle outline
        if self.center is not None and self.radius > 0:
            circle = Circle(self.center, self.radius, fill=False, color='red', linewidth=2)
            self.ax.add_patch(circle)
            # Draw center point
            self.ax.plot(self.center[0], self.center[1], 'ro', markersize=5)

        # Set title
        image_name = os.path.basename(self.image_files[self.current_image_idx])
        self.ax.set_title(f"Image {self.current_image_idx + 1}/{len(self.image_files)}: {image_name}")
        self.ax.axis('off')

        self.fig.canvas.draw()

    def run(self):
        """Start the interactive annotation process"""
        if not self.image_files:
            return

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Load first image
        self.load_image(0)

        # Set up event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_event)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_event)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_event)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initial display
        self.update_display()

        plt.show()


def main():
    """Main function to run the optical disc annotation tool"""
    annotator = OpticalDiscAnnotator()
    annotator.run()


if __name__ == "__main__":
    main()