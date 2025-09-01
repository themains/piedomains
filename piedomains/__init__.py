# New modern API (recommended)
from .api import DomainClassifier, classify_domains

# Legacy API (deprecated but maintained for backward compatibility)
from .domain import pred_shalla_cat
from .domain import pred_shalla_cat_with_images
from .domain import pred_shalla_cat_with_text
from .domain import pred_shalla_cat_archive
from .domain import pred_shalla_cat_with_text_archive  
from .domain import pred_shalla_cat_with_images_archive

__all__ = [
    # New API
    "DomainClassifier",
    "classify_domains",
    
    # Legacy API
    "pred_shalla_cat", 
    "pred_shalla_cat_with_text", 
    "pred_shalla_cat_with_images",
    "pred_shalla_cat_archive",
    "pred_shalla_cat_with_text_archive",
    "pred_shalla_cat_with_images_archive"
]