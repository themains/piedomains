#!/usr/bin/env python3
"""
Demonstration script for testing piedomains in Jupyter Lab
This shows the expected usage and output format.
"""

import sys
import os
sys.path.insert(0, '.')

def demo_piedomains():
    """Demo script showing how to use piedomains package"""
    
    print("=== Piedomains Demo ===")
    print("This demonstrates the expected usage for Jupyter Lab")
    
    # Show the basic import structure
    print("\n1. Import the functions:")
    print("from piedomains import pred_shalla_cat, pred_shalla_cat_with_text, pred_shalla_cat_with_images")
    
    # Show categories available
    print("\n2. Available categories:")
    from piedomains.constants import classes
    print(f"Total categories: {len(classes)}")
    print("Categories:", classes)
    
    # Show expected usage
    print("\n3. Basic usage:")
    print("domains = ['google.com', 'amazon.com', 'facebook.com']")
    print("result = pred_shalla_cat(domains)")
    print("print(result)")
    
    # Show expected output structure from the notebook example
    print("\n4. Expected output structure:")
    print("DataFrame with columns:")
    columns = [
        'domain', 'text_label', 'text_prob', 'text_domain_probs',
        'used_domain_text', 'extracted_text', 'text_extract_errors',
        'image_label', 'image_prob', 'image_domain_probs', 
        'used_domain_screenshot', 'pred_label', 'pred_prob', 'combined_domain_probs'
    ]
    for col in columns:
        print(f"  - {col}")
    
    print("\n5. Example results from notebook:")
    print("google.com -> searchengines (0.85 confidence)")
    print("amazon.com -> shopping (0.92 confidence)")
    print("facebook.com -> socialnet (0.89 confidence)")
    
    print("\n6. Text-only vs Image-only predictions:")
    print("text_only = pred_shalla_cat_with_text(['google.com'])")
    print("image_only = pred_shalla_cat_with_images(['google.com'])")
    
    print("\nNote: Models are downloaded from Harvard Dataverse on first use")
    print("Text model: ~200MB, Image model: ~150MB")

if __name__ == "__main__":
    demo_piedomains()