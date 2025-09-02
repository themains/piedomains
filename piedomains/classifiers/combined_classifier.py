#!/usr/bin/env python3
"""
Combined text and image-based domain classification.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from ..logging import get_logger
from .text_classifier import TextClassifier
from .image_classifier import ImageClassifier

logger = get_logger()


class CombinedClassifier:
    """Combined classifier using both text and image analysis."""
    
    def __init__(self, cache_dir: Optional[str] = None, archive_date: Optional[str] = None):
        """
        Initialize combined classifier.
        
        Args:
            cache_dir (str, optional): Directory for caching content
            archive_date (str, optional): Date for archive.org snapshots
        """
        self.cache_dir = cache_dir or "cache"
        self.archive_date = archive_date
        self.text_classifier = TextClassifier(cache_dir, archive_date)
        self.image_classifier = ImageClassifier(cache_dir, archive_date)
    
    def predict(self, domains: List[str], use_cache: bool = True, latest: bool = False) -> pd.DataFrame:
        """
        Predict domain categories using combined text and image analysis.
        
        Args:
            domains (List[str]): List of domain names or URLs
            use_cache (bool): Whether to use cached content
            latest (bool): Whether to download latest model
            
        Returns:
            pd.DataFrame: Combined predictions with ensemble results
        """
        if not domains:
            raise ValueError("Provide list of domains")
        
        logger.info(f"Starting combined classification for {len(domains)} domains")
        
        # Get text predictions
        try:
            text_results = self.text_classifier.predict(domains, use_cache, latest)
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            text_results = pd.DataFrame()
        
        # Get image predictions  
        try:
            image_results = self.image_classifier.predict(domains, use_cache, latest)
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            image_results = pd.DataFrame()
        
        # Combine results
        combined_results = self._combine_predictions(text_results, image_results, domains)
        
        return combined_results
    
    def _combine_predictions(self, text_df: pd.DataFrame, image_df: pd.DataFrame, domains: List[str]) -> pd.DataFrame:
        """
        Combine text and image predictions into ensemble results.
        
        Args:
            text_df (pd.DataFrame): Text classification results
            image_df (pd.DataFrame): Image classification results
            domains (List[str]): Original domain list
            
        Returns:
            pd.DataFrame: Combined results with ensemble predictions
        """
        results = []
        
        # Convert DataFrames to dictionaries for easier lookup
        text_dict = {}
        image_dict = {}
        
        if not text_df.empty:
            text_dict = text_df.set_index('domain').to_dict('index')
        if not image_df.empty:
            image_dict = image_df.set_index('domain').to_dict('index')
        
        for domain in domains:
            # Parse domain name for consistency
            domain_name = self.text_classifier.processor._parse_domain_name(domain)
            
            result_row = {
                'domain': domain_name,
                'pred_label': None,
                'pred_prob': None,
                'text_label': None,
                'text_prob': None,
                'image_label': None,
                'image_prob': None,
                'used_domain_text': False,
                'used_domain_screenshot': False,
                'extracted_text': None,
                'error': None
            }
            
            if self.archive_date:
                result_row['archive_date'] = self.archive_date
            
            # Get text results
            text_data = text_dict.get(domain_name, {})
            if text_data:
                result_row.update({
                    'text_label': text_data.get('text_label'),
                    'text_prob': text_data.get('text_prob'),
                    'used_domain_text': text_data.get('used_domain_text', False),
                    'extracted_text': text_data.get('extracted_text')
                })
            
            # Get image results
            image_data = image_dict.get(domain_name, {})
            if image_data:
                result_row.update({
                    'image_label': image_data.get('image_label'),
                    'image_prob': image_data.get('image_prob'),
                    'used_domain_screenshot': image_data.get('used_domain_screenshot', False)
                })
            
            # Combine predictions using ensemble logic
            ensemble_result = self._ensemble_predict(text_data, image_data)
            result_row.update(ensemble_result)
            
            # Set error if both failed
            if not text_data and not image_data:
                result_row['error'] = "Both text and image classification failed"
            elif text_data.get('error') and image_data.get('error'):
                result_row['error'] = f"Text: {text_data.get('error')}; Image: {image_data.get('error')}"
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def _ensemble_predict(self, text_data: Dict, image_data: Dict) -> Dict:
        """
        Combine text and image predictions using ensemble logic.
        
        Args:
            text_data (Dict): Text classification results
            image_data (Dict): Image classification results
            
        Returns:
            Dict: Ensemble prediction results
        """
        text_probs = text_data.get('text_domain_probs', {})
        image_probs = image_data.get('image_domain_probs', {})
        
        if not text_probs and not image_probs:
            return {'pred_label': None, 'pred_prob': None}
        
        if not text_probs:
            # Only image available
            return {
                'pred_label': image_data.get('image_label'),
                'pred_prob': image_data.get('image_prob')
            }
        
        if not image_probs:
            # Only text available
            return {
                'pred_label': text_data.get('text_label'),
                'pred_prob': text_data.get('text_prob')
            }
        
        # Both available - combine with equal weighting
        combined_probs = {}
        all_classes = set(text_probs.keys()) | set(image_probs.keys())
        
        for class_name in all_classes:
            text_prob = text_probs.get(class_name, 0.0)
            image_prob = image_probs.get(class_name, 0.0)
            combined_probs[class_name] = (text_prob + image_prob) / 2.0
        
        # Find best combined prediction
        best_class = max(combined_probs, key=combined_probs.get)
        best_prob = combined_probs[best_class]
        
        return {
            'pred_label': best_class,
            'pred_prob': best_prob
        }