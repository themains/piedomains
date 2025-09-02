#!/usr/bin/env python3
"""
Image-based domain classification using homepage screenshots.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from ..base import Base
from ..constants import classes
from ..logging import get_logger
from ..processors.content_processor import ContentProcessor

logger = get_logger()


class ImageClassifier(Base):
    """Image-based domain content classifier using homepage screenshots."""
    
    MODELFN = "model/shallalist"
    model_file_name = "shallalist_v5_model.tar.gz"
    
    def __init__(self, cache_dir: Optional[str] = None, archive_date: Optional[str] = None):
        """
        Initialize image classifier.
        
        Args:
            cache_dir (str, optional): Directory for caching content
            archive_date (str, optional): Date for archive.org snapshots
        """
        self.cache_dir = cache_dir or "cache"
        self.archive_date = archive_date
        self.processor = ContentProcessor(cache_dir, archive_date)
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
            image_model_path = os.path.join(model_path, "saved_model", "pydomains_images")
            try:
                self._model = tf.keras.models.load_model(image_model_path)
            except ValueError as e:
                if "File format not supported" in str(e) and "Keras 3" in str(e):
                    logger.info("Loading legacy SavedModel with TFSMLayer for Keras 3 compatibility")
                    self._model = tf.keras.layers.TFSMLayer(image_model_path, call_endpoint='serving_default')
                else:
                    raise
            
            logger.info("Loaded image classification model")
            
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            raise
    
    def predict(self, domains: List[str], use_cache: bool = True, latest: bool = False) -> pd.DataFrame:
        """
        Predict domain categories using homepage screenshots.
        
        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached images
            latest (bool): Whether to download latest model
            
        Returns:
            pd.DataFrame: Predictions with probabilities and metadata
        """
        # Validate inputs
        if not domains:
            raise ValueError("Provide list of domains")
            
        # Load models
        self.load_models(latest)
        
        # Extract screenshot images
        image_paths, errors = self.processor.extract_image_content(domains, use_cache)
        
        # Convert images to tensors
        image_tensors = self.processor.prepare_image_tensors(image_paths)
        
        # Prepare results DataFrame
        results = []
        
        for domain in domains:
            domain_name = self.processor._parse_domain_name(domain)
            
            result_row = {
                'domain': domain_name,
                'image_label': None,
                'image_prob': None,
                'image_domain_probs': None,
                'used_domain_screenshot': False,
                'error': None
            }
            
            if self.archive_date:
                result_row['archive_date'] = self.archive_date
            
            if domain_name in errors:
                result_row['error'] = errors[domain_name]
                results.append(result_row)
                continue
                
            if domain_name not in image_tensors:
                result_row['error'] = "No image tensor available"
                results.append(result_row)
                continue
            
            try:
                result_row['used_domain_screenshot'] = True
                
                # Get model predictions
                predictions = self._predict_image(image_tensors[domain_name])
                result_row.update(predictions)
                    
            except Exception as e:
                result_row['error'] = f"Prediction error: {e}"
                
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def _predict_image(self, image_tensor: np.ndarray) -> Dict:
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
            
            # Get model predictions
            if hasattr(self._model, 'predict'):
                raw_predictions = self._model.predict(image_input, verbose=0)[0]
            else:
                # Handle TFSMLayer case  
                raw_predictions = self._model(image_input)['output_0'][0]
            
            # Convert to class probabilities
            probs = {}
            for i, class_name in enumerate(classes):
                probs[class_name] = float(raw_predictions[i])
            
            # Find best prediction
            best_class = max(probs, key=probs.get)
            best_prob = probs[best_class]
            
            return {
                'image_label': best_class,
                'image_prob': best_prob,
                'image_domain_probs': probs
            }
            
        except Exception as e:
            logger.error(f"Image prediction failed: {e}")
            return {
                'image_label': None,
                'image_prob': None,
                'image_domain_probs': None
            }