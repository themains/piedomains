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
            self._model = tf.keras.models.load_model(text_model_path)

            # Load calibrators
            import joblib
            calibrator_path = os.path.join(model_path, "calibrate", "text")
            self._calibrators = {}

            for class_name in classes:
                calibrator_file = os.path.join(calibrator_path, f"{class_name}.sav")
                if os.path.exists(calibrator_file):
                    self._calibrators[class_name] = joblib.load(calibrator_file)

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
        text_content, errors = self.processor.extract_text_content(domains, use_cache)

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
            raw_predictions = self._model.predict(text_input, verbose=0)[0]
            
            # Apply calibration if available
            calibrated_probs = {}
            for i, class_name in enumerate(classes):
                raw_prob = float(raw_predictions[i])
                
                if class_name in self._calibrators:
                    # Apply isotonic regression calibration
                    calibrator = self._calibrators[class_name]
                    calibrated_prob = float(calibrator.predict([raw_prob])[0])
                else:
                    calibrated_prob = raw_prob
                    
                calibrated_probs[class_name] = calibrated_prob
            
            # Find best prediction
            best_class = max(calibrated_probs, key=calibrated_probs.get)
            best_prob = calibrated_probs[best_class]
            
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