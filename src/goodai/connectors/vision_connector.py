"""
Vision Connector Module

Non-invasive data capture from visual sources (images, screenshots, camera feeds).
Designed for headless operation - works in CI/CD pipelines and Docker containers.

Good AI Philosophy: Non-invasive by default.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np


@dataclass
class ExtractionResult:
    """Result of extracting a value from an image region."""

    region_name: str
    raw_text: str
    parsed_value: Any
    confidence: float
    region_bounds: Dict[str, int]


@dataclass
class RegionConfig:
    """Configuration for a region of interest."""

    x: int
    y: int
    width: int
    height: int
    value_type: str = "text"  # text, number, integer


class VisionConnector:
    """
    Non-invasive data capture from visual sources.

    HEADLESS SAFE: Uses opencv-python-headless, no cv2.imshow calls.
    Works in Docker, CI/CD, and server environments without display.

    Args:
        config: Optional configuration dictionary

    Example:
        >>> connector = VisionConnector()
        >>> regions = {
        ...     "temperature": {"x": 100, "y": 200, "w": 80, "h": 30, "type": "number"},
        ...     "pressure": {"x": 100, "y": 250, "w": 80, "h": 30, "type": "number"}
        ... }
        >>> values = connector.read_from_image("dashboard.png", regions)
        >>> print(values)  # {"temperature": 72.5, "pressure": 14.7}
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._cv2 = None
        self._pytesseract = None
        self._validate_headless()

    def _validate_headless(self):
        """
        Ensure we can run without display.

        Uses lazy import to allow module to load without dependencies installed.
        """
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            self._cv2 = None

        try:
            import pytesseract
            self._pytesseract = pytesseract
        except ImportError:
            self._pytesseract = None

    def _ensure_dependencies(self):
        """Check that required dependencies are available."""
        if self._cv2 is None:
            raise ImportError(
                "opencv-python-headless is required for VisionConnector. "
                "Install with: pip install opencv-python-headless"
            )
        if self._pytesseract is None:
            raise ImportError(
                "pytesseract is required for VisionConnector. "
                "Install with: pip install pytesseract"
            )

    def read_from_image(
        self,
        image_path: Union[str, Path],
        regions: Dict[str, dict]
    ) -> Dict[str, Any]:
        """
        Extract values from specified regions of an image.

        Args:
            image_path: Path to image file (PNG, JPG, etc.)
            regions: Dictionary of region configurations
                     {name: {x, y, w, h, type}}
                     - x, y: Top-left corner coordinates
                     - w, h: Width and height
                     - type: "text", "number", or "integer" (default: "text")

        Returns:
            Dictionary of {region_name: extracted_value}

        Raises:
            FileNotFoundError: If image file doesn't exist
            ImportError: If required dependencies are not installed
            ValueError: If region configuration is invalid

        Example:
            >>> regions = {
            ...     "meter_reading": {"x": 50, "y": 100, "w": 120, "h": 40, "type": "number"}
            ... }
            >>> values = connector.read_from_image("meter.png", regions)
        """
        self._ensure_dependencies()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image
        img = self._cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        results = {}
        for name, region_config in regions.items():
            # Validate region config
            self._validate_region_config(region_config, name)

            # Extract region
            x = region_config["x"]
            y = region_config["y"]
            w = region_config.get("w", region_config.get("width", 100))
            h = region_config.get("h", region_config.get("height", 30))

            # Crop region
            roi = img[y:y+h, x:x+w]

            if roi.size == 0:
                results[name] = None
                continue

            # Preprocess for OCR
            gray = self._cv2.cvtColor(roi, self._cv2.COLOR_BGR2GRAY)
            thresh = self._cv2.threshold(
                gray, 0, 255,
                self._cv2.THRESH_BINARY + self._cv2.THRESH_OTSU
            )[1]

            # OCR extraction
            raw_text = self._pytesseract.image_to_string(
                thresh,
                config='--psm 7 --oem 3'
            ).strip()

            # Parse based on type
            value_type = region_config.get("type", "text")
            results[name] = self._parse_value(raw_text, value_type)

        return results

    def _validate_region_config(self, config: dict, name: str):
        """Validate region configuration."""
        required = ["x", "y"]
        for field in required:
            if field not in config:
                raise ValueError(f"Region '{name}' missing required field: {field}")

        # Must have width/height in some form
        if not any(k in config for k in ["w", "h", "width", "height"]):
            raise ValueError(
                f"Region '{name}' must specify dimensions (w/h or width/height)"
            )

    def _parse_value(self, raw_text: str, value_type: str) -> Any:
        """Parse extracted text based on expected type."""
        if not raw_text:
            return None

        if value_type == "number":
            # Clean and parse as float
            cleaned = "".join(c for c in raw_text if c.isdigit() or c in ".-")
            try:
                return float(cleaned)
            except ValueError:
                return raw_text

        elif value_type == "integer":
            cleaned = "".join(c for c in raw_text if c.isdigit() or c == "-")
            try:
                return int(cleaned)
            except ValueError:
                return raw_text

        return raw_text

    def read_from_image_detailed(
        self,
        image_path: Union[str, Path],
        regions: Dict[str, dict]
    ) -> List[ExtractionResult]:
        """
        Extract values with detailed results including confidence.

        Args:
            image_path: Path to image file
            regions: Dictionary of region configurations

        Returns:
            List of ExtractionResult objects with detailed information
        """
        self._ensure_dependencies()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = self._cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        results = []
        for name, region_config in regions.items():
            self._validate_region_config(region_config, name)

            x = region_config["x"]
            y = region_config["y"]
            w = region_config.get("w", region_config.get("width", 100))
            h = region_config.get("h", region_config.get("height", 30))

            roi = img[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            gray = self._cv2.cvtColor(roi, self._cv2.COLOR_BGR2GRAY)
            thresh = self._cv2.threshold(
                gray, 0, 255,
                self._cv2.THRESH_BINARY + self._cv2.THRESH_OTSU
            )[1]

            # Get detailed OCR data
            data = self._pytesseract.image_to_data(
                thresh,
                config='--psm 7 --oem 3',
                output_type=self._pytesseract.Output.DICT
            )

            # Extract text and confidence
            texts = []
            confidences = []
            for i, conf in enumerate(data["conf"]):
                if int(conf) > 0:
                    texts.append(data["text"][i])
                    confidences.append(int(conf))

            raw_text = " ".join(texts).strip()
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            value_type = region_config.get("type", "text")
            parsed_value = self._parse_value(raw_text, value_type)

            results.append(ExtractionResult(
                region_name=name,
                raw_text=raw_text,
                parsed_value=parsed_value,
                confidence=avg_confidence / 100.0,
                region_bounds={"x": x, "y": y, "w": w, "h": h}
            ))

        return results

    def calibrate(
        self,
        image_path: Union[str, Path],
        roi_config: Optional[Dict[str, dict]] = None
    ) -> Dict[str, dict]:
        """
        Calibrate ROI (Region of Interest) regions.

        Supports three modes:
        1. If roi_config provided: Validate and return those coordinates
        2. If DISPLAY available: Interactive selection (not recommended for production)
        3. Otherwise: Raise helpful error with instructions

        Args:
            image_path: Path to calibration image
            roi_config: Optional pre-defined region coordinates

        Returns:
            Dictionary of validated region configurations

        Raises:
            ValueError: If no display available and no roi_config provided
        """
        if roi_config:
            # Validate provided config
            for name, config in roi_config.items():
                self._validate_region_config(config, name)
            return roi_config

        # Check for display availability
        if os.environ.get("DISPLAY"):
            return self._interactive_calibration(image_path)

        raise ValueError(
            "No display available for interactive calibration. "
            "Provide roi_config parameter with region coordinates. "
            "Example: {'temperature': {'x': 100, 'y': 200, 'w': 80, 'h': 30, 'type': 'number'}}"
        )

    def _interactive_calibration(self, image_path: Union[str, Path]) -> Dict[str, dict]:
        """
        Interactive ROI selection when display is available.

        Note: This method requires a display environment. For headless
        operation, provide roi_config directly to calibrate().
        """
        self._ensure_dependencies()

        raise NotImplementedError(
            "Interactive calibration requires a display environment. "
            "For CI/Docker, pre-define regions using calibrate(image_path, roi_config={...}). "
            "Tip: Use an image editor to determine pixel coordinates."
        )

    def preprocess_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        operations: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.

        Args:
            image_path: Path to input image
            output_path: Optional path to save preprocessed image
            operations: List of operations to apply
                       Options: "grayscale", "threshold", "denoise", "deskew"

        Returns:
            Preprocessed image as numpy array
        """
        self._ensure_dependencies()

        operations = operations or ["grayscale", "threshold"]

        img = self._cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = img.copy()

        for op in operations:
            if op == "grayscale":
                if len(result.shape) == 3:
                    result = self._cv2.cvtColor(result, self._cv2.COLOR_BGR2GRAY)

            elif op == "threshold":
                if len(result.shape) == 3:
                    result = self._cv2.cvtColor(result, self._cv2.COLOR_BGR2GRAY)
                result = self._cv2.threshold(
                    result, 0, 255,
                    self._cv2.THRESH_BINARY + self._cv2.THRESH_OTSU
                )[1]

            elif op == "denoise":
                result = self._cv2.fastNlMeansDenoising(result)

        if output_path:
            self._cv2.imwrite(str(output_path), result)

        return result

    def is_available(self) -> bool:
        """Check if vision connector dependencies are available."""
        return self._cv2 is not None and self._pytesseract is not None

    def get_status(self) -> dict:
        """Get status of vision connector dependencies."""
        return {
            "opencv_available": self._cv2 is not None,
            "pytesseract_available": self._pytesseract is not None,
            "display_available": bool(os.environ.get("DISPLAY")),
            "headless_mode": True,  # Always headless
        }
