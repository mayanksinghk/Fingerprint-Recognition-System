import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import List
from skimage.morphology import skeletonize
import fingerprint_enhancer

@dataclass
class Minutia:
    x: int
    y: int
    theta: float
    type: str = "unknown"

class Fingerprint:
    GLOBAL_THRESHOLD = 0.01

    def __init__(self, image: np.ndarray, name: str = "Fingerprint"):
        self.name = name
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            self.raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.raw_image = image
            
        self.image_area = 0
        self.segmented_image = None
        self.normalized_image = None
        self.enhanced_image = None
        self.binarized_image = None
        self.thinned_image = None
        self.orientation_matrix = None
        self.minutiae: List[Minutia] = []

    def extract_features(self):
        """Executes the full pipeline to extract minutiae."""
        self._segment()
        self._normalize()
        self._enhance()
        self._binarize()
        self._thin()
        self._compute_orientation()
        self._extract_minutiae()

    def _segment(self, block_size: int = 15):
        n, m = self.raw_image.shape
        new_image = np.zeros((n, m))
        area = 0

        for row in range((n // block_size) + 1):
            for col in range((m // block_size) + 1):
                block = self.raw_image[row * block_size:(row + 1) * block_size, 
                                       col * block_size:(col + 1) * block_size]
                if block.size == 0:
                    continue
                
                mean_val = np.mean(block)
                var = np.sum(np.square(block - mean_val)) / (block_size ** 2)

                if var < self.GLOBAL_THRESHOLD:
                    new_image[row * block_size:(row + 1) * block_size, 
                              col * block_size:(col + 1) * block_size] = 0
                else:
                    new_image[row * block_size:(row + 1) * block_size, 
                              col * block_size:(col + 1) * block_size] = block
                    area += block_size * block_size

        self.image_area = area
        self.segmented_image = new_image

    def _normalize(self):
        image = self.segmented_image
        n, m = image.shape
        M_0, V_0 = 0.5, 0.5
        M = image.mean()
        V = np.sum(np.square(image - M)) / (n * m)

        temp = np.sqrt((V_0 / (V + 1e-8)) * np.square(image - M))
        N = M_0 - temp
        N = N + np.multiply(2 * (image > M), temp)
        
        # Scale to 0-255 for OpenCV processing
        self.normalized_image = cv2.normalize(N, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def _enhance(self):
        """Use fingerprint_enhancer library for superior ridge enhancement and background masking."""
        enhanced = fingerprint_enhancer.enhance_fingerprint(self.normalized_image)
        
        # Ensure the array is correctly scaled for the OpenCV binarization step
        if enhanced.max() <= 1.0:
            enhanced = (enhanced * 255).astype(np.uint8)
        else:
            enhanced = enhanced.astype(np.uint8)
            
        self.enhanced_image = enhanced

    def _binarize(self):
        _, thresh1 = cv2.threshold(self.enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.binarized_image = thresh1 // 255

    def _thin(self):
        thinned = skeletonize(self.binarized_image)
        self.thinned_image = 1 - thinned

    def _compute_orientation(self, N: int = 5):
        kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        Gx = cv2.filter2D(self.raw_image.astype(float), -1, kernel_x)
        Gy = cv2.filter2D(self.raw_image.astype(float), -1, kernel_y)

        b, l = Gx.shape
        target_l = l - N + 1
        target_b = b - N + 1
        orientation_matrix = np.zeros((target_b, target_l))

        for i in range(target_b):
            for j in range(target_l):
                mat_x = Gx[i:i + N, j:j + N]
                mat_y = Gy[i:i + N, j:j + N]
                
                num = np.sum(np.multiply(mat_x, mat_y)) * 2
                denom = np.sum(np.square(mat_x) - np.square(mat_y))
                
                if denom != 0:
                    orientation_matrix[i, j] = math.degrees(math.atan(num / denom)) / 2 + 90
                else:
                    orientation_matrix[i, j] = 90

        self.orientation_matrix = orientation_matrix

    def _extract_minutiae(self):
        image = self.thinned_image
        n, m = image.shape
        self.minutiae = []
        
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                block = image[i - 1:i + 2, j - 1:j + 2]
                if block[1][1] == 0:
                    temp = block.sum()
                    if temp == 8 or temp == 7:
                        try:
                            theta = self.orientation_matrix[i - 1][j - 1]
                            self.minutiae.append(Minutia(x=j, y=i, theta=theta))
                        except IndexError:
                            continue