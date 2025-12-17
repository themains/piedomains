#!/usr/bin/env python3
"""
Image-based domain classification using homepage screenshots.
"""

import os
from datetime import UTC
from pathlib import Path

import numpy as np

from .base import Base
from .constants import classes
from .content_processor import ContentProcessor
from .piedomains_logging import get_logger

logger = get_logger()


class ImageClassifier(Base):
    """Image-based domain content classifier using homepage screenshots."""

    MODELFN = "model/shallalist"
    model_file_name = "shallalist_v5_model.tar.gz"

    def __init__(self, cache_dir: str | None = None, archive_date: str | None = None):
        """
        Initialize image classifier.

        Args:
            cache_dir (str, optional): Directory for caching content
            archive_date (str, optional): Date for archive.org snapshots
        """
        self.cache_dir = Path(cache_dir or "cache")
        self.archive_date = archive_date
        self.processor = ContentProcessor(str(self.cache_dir), archive_date)
        self._model = None

    def load_models(self, latest: bool = False):
        """Load image classification model."""
        if self._model is not None and not latest:
            return

        try:
            # Load model data
            model_path = self.load_model_data(self.model_file_name, latest)

            # Import TensorFlow here to avoid loading unless needed
            import tensorflow as tf

            # Load image model
            image_model_path = os.path.join(
                model_path, "saved_model", "pydomains_images"
            )
            try:
                self._model = tf.keras.models.load_model(image_model_path)
            except ValueError as e:
                if "File format not supported" in str(e) and "Keras 3" in str(e):
                    logger.info(
                        "Loading legacy SavedModel with TFSMLayer for Keras 3 compatibility"
                    )
                    self._model = tf.keras.layers.TFSMLayer(
                        image_model_path, call_endpoint="serving_default"
                    )
                else:
                    raise

            logger.info("Loaded image classification model")

        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            raise

    def classify(self, domains: list[str], latest: bool = False) -> list[dict]:
        """
        Classify domains using their cached screenshot images.

        Args:
            domains: List of domain names to classify
            latest: Whether to use latest model version

        Returns:
            List of classification result dictionaries

        Example:
            >>> classifier = ImageClassifier()
            >>> results = classifier.classify(["cnn.com", "bbc.com"])
            >>> print(results[0]["category"])
            news
        """
        self.load_models(latest)

        results = []

        for domain in domains:
            result = self._classify_single_domain(domain)
            results.append(result)

        return results

    def _classify_single_domain(self, domain: str) -> dict:
        """Classify a single domain using its cached screenshot image."""
        image_path = self.cache_dir / "images" / f"{domain}.png"

        result = {
            "url": domain,
            "domain": domain,
            "text_path": str(
                (self.cache_dir / "html" / f"{domain}.html").relative_to(self.cache_dir)
            ),
            "image_path": str(image_path.relative_to(self.cache_dir)),
            "date_time_collected": None,  # Would need to read from metadata if needed
            "model_used": "image/shallalist_ml",
            "category": None,
            "confidence": None,
            "raw_predictions": None,
            "reason": None,
            "error": None,
        }

        if not image_path.exists():
            result["error"] = f"Image file not found: {image_path}"
            return result

        try:
            # Load and process image
            from PIL import Image

            # Load image and convert to tensor
            img = Image.open(image_path)
            img = img.convert("RGB")
            img = img.resize((254, 254))

            # Convert to numpy array and normalize
            img_array = np.array(img).astype("float32") / 255.0

            # Get predictions
            predictions = self._predict_image(img_array)

            # Convert to JSON format
            result["category"] = predictions.get("image_label")
            result["confidence"] = predictions.get("image_prob")
            result["raw_predictions"] = predictions.get("image_domain_probs")

        except FileNotFoundError:
            result["error"] = f"Image file not found: {image_path}"
        except Exception as e:
            result["error"] = f"Classification error: {e}"

        return result

    def classify_from_paths(
        self, data_paths: list[dict], output_file: str = None, latest: bool = False
    ) -> list[dict]:
        """
        Classify domains using screenshot files from collected data paths.

        Args:
            data_paths: List of dicts with domain data containing image_path, domain, etc.
            output_file: Optional path to save JSON results
            latest: Whether to use latest model version

        Returns:
            List of classification result dictionaries (JSON format)

        Example:
            >>> classifier = ImageClassifier()
            >>> data = [{"domain": "cnn.com", "image_path": "images/cnn.com.png", ...}]
            >>> results = classifier.classify_from_paths(data)
            >>> print(results[0]["category"])
            news
        """
        self.load_models(latest)

        results = []

        for domain_data in data_paths:
            domain = domain_data.get("domain")
            image_path = domain_data.get("image_path")

            result = {
                "url": domain_data.get("url", domain),
                "domain": domain,
                "text_path": domain_data.get("text_path"),
                "image_path": image_path,
                "date_time_collected": domain_data.get("date_time_collected"),
                "model_used": "image/shallalist_ml",
                "category": None,
                "confidence": None,
                "raw_predictions": None,
                "reason": None,
                "error": None,
            }

            if not domain or not image_path:
                result["error"] = "Missing domain or image_path"
                results.append(result)
                continue

            try:
                # Clean path resolution using pathlib
                if not Path(image_path).is_absolute():
                    img_path = self.cache_dir / image_path
                else:
                    img_path = Path(image_path)

                # Load and process image from file path
                from PIL import Image

                # Load image and convert to tensor
                img = Image.open(img_path)
                img = img.convert("RGB")
                img = img.resize((254, 254))

                # Convert to numpy array and normalize
                img_array = np.array(img).astype("float32") / 255.0

                # Get predictions
                predictions = self._predict_image(img_array)

                # Convert to JSON format
                result["category"] = predictions.get("image_label")
                result["confidence"] = predictions.get("image_prob")
                result["raw_predictions"] = predictions.get("image_domain_probs")

            except FileNotFoundError:
                result["error"] = f"Image file not found: {image_path}"
            except Exception as e:
                result["error"] = f"Classification error: {e}"

            results.append(result)

        # Save results if output file specified
        if output_file:
            import json
            from datetime import datetime

            # Create results directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Add metadata
            output_data = {
                "inference_timestamp": datetime.now(UTC).isoformat(),
                "model_used": "image/shallalist_ml",
                "total_domains": len(data_paths),
                "successful": len([r for r in results if r["category"] is not None]),
                "failed": len([r for r in results if r["error"] is not None]),
                "results": results,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Saved image classification results to {output_file}")

        return results

    def classify_from_data(
        self, collection_data: dict, output_file: str = None, latest: bool = False
    ) -> list[dict]:
        """
        Classify domains using collection metadata from DataCollector.

        Args:
            collection_data: Collection metadata dict from DataCollector.collect()
            output_file: Optional path to save JSON results
            latest: Whether to use latest model version

        Returns:
            List of classification result dictionaries (JSON format)

        Example:
            >>> from piedomains import DataCollector
            >>> collector = DataCollector()
            >>> data = collector.collect(["cnn.com"])
            >>> classifier = ImageClassifier()
            >>> results = classifier.classify_from_data(data)
        """
        # Extract domain data from collection metadata
        domains_data = collection_data.get("domains", [])

        # Filter only successful data collection with images
        valid_domains = [
            d for d in domains_data if d.get("fetch_success") and d.get("image_path")
        ]

        if not valid_domains:
            logger.warning("No valid domains with image data found in collection")
            return []

        return self.classify_from_paths(valid_domains, output_file, latest)

    def _predict_image(self, image_tensor: np.ndarray) -> dict:
        """
        Generate predictions for image tensor.

        Args:
            image_tensor (np.ndarray): Preprocessed image array

        Returns:
            Dict: Prediction results
        """
        try:
            # Prepare input for model (add batch dimension)
            image_input = np.expand_dims(image_tensor, axis=0)
            logger.info(
                f"Image tensor stats - shape: {image_tensor.shape}, min: {np.min(image_tensor):.3f}, max: {np.max(image_tensor):.3f}, mean: {np.mean(image_tensor):.3f}"
            )

            # Get model predictions
            if hasattr(self._model, "predict"):
                raw_predictions = self._model.predict(image_input, verbose=0)[0]
            else:
                # Handle TFSMLayer case - try common output keys
                model_output = self._model(image_input)
                logger.info(f"TFSMLayer output keys: {list(model_output.keys())}")
                if "output_0" in model_output:
                    logger.info("Using output_0 key")
                    raw_predictions = model_output["output_0"][0]
                elif "dense_2" in model_output:
                    logger.info("Using dense_2 key")
                    raw_predictions = model_output["dense_2"][0]
                elif "predictions" in model_output:
                    logger.info("Using predictions key")
                    raw_predictions = model_output["predictions"][0]
                else:
                    # Fallback: use the first available key
                    key = list(model_output.keys())[0]
                    logger.info(f"Using fallback key: {key}")
                    raw_predictions = model_output[key][0]

            # Convert logits to probabilities using softmax
            import tensorflow as tf

            logger.info(
                f"Raw logits stats - shape: {raw_predictions.shape}, min: {np.min(raw_predictions):.3f}, max: {np.max(raw_predictions):.3f}"
            )
            softmax_probs = tf.nn.softmax(raw_predictions).numpy()
            logger.info("Applied softmax to convert logits to probabilities")

            # Convert to class probabilities
            probs = {}
            for i, class_name in enumerate(classes):
                probs[class_name] = float(softmax_probs[i])

            logger.info(
                f"Top 3 image predictions: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}"
            )

            # Find best prediction
            best_class = max(probs, key=probs.get)
            best_prob = probs[best_class]
            logger.info(f"Image prediction: {best_class} ({best_prob:.3f})")

            return {
                "image_label": best_class,
                "image_prob": best_prob,
                "image_domain_probs": probs,
            }

        except Exception as e:
            logger.error(f"Image prediction failed: {e}")
            return {"image_label": None, "image_prob": None, "image_domain_probs": None}
