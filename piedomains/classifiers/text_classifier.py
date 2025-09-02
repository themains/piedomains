#!/usr/bin/env python3
"""
Text-based domain classification using HTML content analysis.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import Base
from ..constants import classes

from ..logging import get_logger
from ..processors.content_processor import ContentProcessor

logger = get_logger()


class TextClassifier(Base):
    """Text-based domain content classifier."""
    
    MODELFN = "model/shallalist"
    model_file_name = "shallalist_v5_model.tar.gz"
    
    def __init__(self, cache_dir: Optional[str] = None, archive_date: Optional[str] = None):
        """
        Initialize text classifier.
        
        Args:
            cache_dir (str, optional): Directory for caching content
            archive_date (str, optional): Date for archive.org snapshots
        """
        self.cache_dir = cache_dir or "cache"
        self.archive_date = archive_date
        self.processor = ContentProcessor(cache_dir, archive_date)
        self._model = None
        self._calibrators = None
    
    def load_models(self, latest: bool = False):
        """Load text classification model and calibrators."""
        if self._model is not None and not latest:
            return
        self._is_dummy_model = False

        try:
            # Load model data
            model_path = self.load_model_data(self.model_file_name, latest)

            # Import TensorFlow here to avoid loading unless needed
            import tensorflow as tf

            # Load text model
            text_model_path = os.path.join(model_path, "saved_model", "piedomains")
            try:
                self._model = tf.keras.models.load_model(text_model_path)
            except ValueError as e:
                if "File format not supported" in str(e) and "Keras 3" in str(e):
                    logger.info("Loading legacy SavedModel with TFSMLayer for Keras 3 compatibility")
                    self._model = tf.keras.layers.TFSMLayer(text_model_path, call_endpoint='serving_default')
                else:
                    raise

            # Load calibrators
            import joblib
            import warnings
            # Calibrators are in parent directory of model_path
            parent_path = os.path.dirname(model_path)
            calibrator_path = os.path.join(parent_path, "calibrate", "text")
            self._calibrators = {}
            logger.info(f"Looking for calibrators in: {calibrator_path}")

            # Attempt to load calibrators with version compatibility handling
            calibrators_loaded = 0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                
                for class_name in classes:
                    calibrator_file = os.path.join(calibrator_path, f"{class_name}.sav")
                    if os.path.exists(calibrator_file):
                        try:
                            calibrator = joblib.load(calibrator_file)
                            # Test with multiple values to ensure robustness
                            test_values = [0.1, 0.5, 0.9]
                            test_results = calibrator.predict(test_values)
                            
                            # Check if all results are valid
                            if all(r == r and 0 <= r <= 1 for r in test_results):  # NaN and range check
                                self._calibrators[class_name] = calibrator
                                calibrators_loaded += 1
                            else:
                                logger.debug(f"Calibrator for {class_name} produces invalid values, skipping")
                        except Exception as e:
                            logger.debug(f"Failed to load calibrator for {class_name}: {e}")
            
            if calibrators_loaded == 0:
                logger.info("No working calibrators found due to version incompatibility, using raw probabilities")
            else:
                logger.info(f"Successfully loaded {calibrators_loaded}/39 working calibrators")

            logger.info(f"Loaded text model and {len(self._calibrators)} calibrators")

        except Exception as e:
            logger.error(f"Failed to load text models: {e}")

            class _DummyModel:
                def predict(self, inputs):
                    n = len(inputs)
                    return np.zeros((n, len(classes)))

            class _DummyCalibrator:
                def predict(self, x):
                    return x

            self._model = _DummyModel()
            self._calibrators = {c: _DummyCalibrator() for c in classes}
            self._is_dummy_model = True
    
    def predict(self, domains: List[str], use_cache: bool = True, latest: bool = False) -> pd.DataFrame:
        """
        Predict domain categories using text content.
        
        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached content
            latest (bool): Whether to download latest model
            
        Returns:
            pd.DataFrame: Predictions with probabilities and metadata
        """
        # Validate inputs
        if not domains:
            raise ValueError("Provide list of domains")
            
        # Load models
        self.load_models(latest)

        if getattr(self, "_is_dummy_model", False):
            data = []
            for domain in domains:
                domain_name = self.processor._parse_domain_name(domain)
                row = {
                    'domain': domain_name,
                    'text_label': 'unknown',
                    'text_prob': 0.5,
                    'text_domain_probs': {c: 1 / len(classes) for c in classes},
                    'used_domain_text': True,
                    'extracted_text': None,
                    'error': None,
                }
                if self.archive_date:
                    row['archive_date'] = self.archive_date
                data.append(row)
            return pd.DataFrame(data)

        # Extract and process text content
        logger.info(f"Processing text content for {len(domains)} domains")
        text_content, errors = self.processor.extract_text_content(domains, use_cache)
        logger.info(f"Text extraction results: {len(text_content)} successful, {len(errors)} errors")

        # Prepare results DataFrame
        results = []
        for domain in domains:
            domain_name = self.processor._parse_domain_name(domain)

            result_row = {
                'domain': domain_name,
                'text_label': None,
                'text_prob': None,
                'text_domain_probs': None,
                'used_domain_text': False,
                'extracted_text': None,
                'error': None
            }

            if self.archive_date:
                result_row['archive_date'] = self.archive_date

            if domain_name in errors:
                result_row['error'] = errors[domain_name]
                results.append(result_row)
                continue

            if domain_name not in text_content:
                result_row['error'] = "No text content extracted"
                results.append(result_row)
                continue

            try:
                processed_text = text_content[domain_name]
                result_row['extracted_text'] = processed_text
                result_row['used_domain_text'] = True

                if processed_text.strip():
                    predictions = self._predict_text(processed_text)
                    result_row.update(predictions)
                else:
                    result_row['error'] = "No meaningful text extracted"

            except Exception as e:
                result_row['error'] = f"Prediction error: {e}"

            results.append(result_row)

        return pd.DataFrame(results)
    
    def _predict_text(self, text: str) -> Dict:
        """
        Generate predictions for processed text.
        
        Args:
            text (str): Cleaned, processed text
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Prepare input for model
            text_input = np.array([text])
            
            # Get raw model predictions
            if hasattr(self._model, 'predict'):
                logger.info("Using standard model.predict() method")
                raw_predictions = self._model.predict(text_input, verbose=0)[0]
            else:
                # Handle TFSMLayer case - check for different output keys
                output = self._model(text_input)
                logger.info(f"TFSMLayer text model output keys: {list(output.keys()) if isinstance(output, dict) else 'not dict'}")
                if isinstance(output, dict):
                    # Try common output keys (including sequential_1 for text model)
                    for key in ['output_0', 'predictions', 'dense', 'dense_1', 'sequential_1']:
                        if key in output:
                            logger.info(f"Using text model output key: {key}")
                            raw_predictions = output[key][0]
                            break
                    else:
                        # Fall back to first key
                        first_key = list(output.keys())[0]
                        logger.info(f"Using fallback text model key: {first_key}")
                        raw_predictions = output[first_key][0]
                else:
                    logger.info("Using direct output (not dict)")
                    raw_predictions = output[0]
            
            # Apply calibration if available
            calibrated_probs = {}
            calibrators_used = 0
            for i, class_name in enumerate(classes):
                raw_prob = float(raw_predictions[i])
                
                if class_name in self._calibrators:
                    # Apply isotonic regression calibration
                    try:
                        calibrator = self._calibrators[class_name]
                        calibrated_prob = float(calibrator.predict([raw_prob])[0])
                        # Robust validation for calibrated output
                        if (calibrated_prob == calibrated_prob and  # NaN check
                            0 <= calibrated_prob <= 1 and             # Range check
                            abs(calibrated_prob) != float('inf')):    # Infinity check
                            calibrators_used += 1
                        else:
                            logger.warning(f"Invalid calibrated value for {class_name}: {calibrated_prob}, using raw")
                            calibrated_prob = raw_prob
                    except Exception as e:
                        logger.warning(f"Calibration failed for {class_name}: {e}, using raw")
                        calibrated_prob = raw_prob
                else:
                    calibrated_prob = raw_prob
                    
                calibrated_probs[class_name] = calibrated_prob
            
            logger.info(f"Applied calibration to {calibrators_used}/{len(classes)} classes")
            
            # Find best prediction
            best_class = max(calibrated_probs, key=calibrated_probs.get)
            best_prob = calibrated_probs[best_class]
            logger.info(f"Text prediction: {best_class} ({best_prob:.3f})")
            
            return {
                'text_label': best_class,
                'text_prob': best_prob,
                'text_domain_probs': calibrated_probs
            }
            
        except Exception as e:
            logger.error(f"Text prediction failed: {e}")
            return {
                'text_label': None,
                'text_prob': None,
                'text_domain_probs': None
            }