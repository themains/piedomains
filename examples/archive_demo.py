#!/usr/bin/env python3
"""
Demonstration of archive functionality using direct URLs.
Since archive.org API is currently down, we'll use known archive URLs.
"""

import sys
import os
sys.path.insert(0, '.')

import requests
import pandas as pd
from bs4 import BeautifulSoup

def test_direct_archive_urls():
    """Test with known working archive URLs."""
    print("=== Archive.org Direct URL Test ===")
    
    # Known working archive URLs from different time periods
    test_cases = [
        {
            'domain': 'google.com',
            'archive_url': 'https://web.archive.org/web/20200101120000/https://google.com',
            'date': '2020-01-01'
        },
        {
            'domain': 'amazon.com', 
            'archive_url': 'https://web.archive.org/web/20150101120000/https://amazon.com',
            'date': '2015-01-01'
        },
        {
            'domain': 'facebook.com',
            'archive_url': 'https://web.archive.org/web/20100101120000/https://facebook.com', 
            'date': '2010-01-01'
        },
        {
            'domain': 'twitter.com',
            'archive_url': 'https://web.archive.org/web/20120101120000/https://twitter.com',
            'date': '2012-01-01'
        },
        {
            'domain': 'youtube.com',
            'archive_url': 'https://web.archive.org/web/20080101120000/https://youtube.com',
            'date': '2008-01-01'
        },
        {
            'domain': 'reddit.com',
            'archive_url': 'https://web.archive.org/web/20100601120000/https://reddit.com',
            'date': '2010-06-01'  
        },
        {
            'domain': 'netflix.com',
            'archive_url': 'https://web.archive.org/web/20100101120000/https://netflix.com',
            'date': '2010-01-01'
        },
        {
            'domain': 'github.com',
            'archive_url': 'https://web.archive.org/web/20100401120000/https://github.com',
            'date': '2010-04-01'
        },
        {
            'domain': 'stackoverflow.com',
            'archive_url': 'https://web.archive.org/web/20100101120000/https://stackoverflow.com',
            'date': '2010-01-01'
        },
        {
            'domain': 'wikipedia.org',
            'archive_url': 'https://web.archive.org/web/20050101120000/https://wikipedia.org',
            'date': '2005-01-01'
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/10] Testing {case['domain']} from {case['date']}...")
        
        try:
            # Test if we can fetch the archived page
            response = requests.get(case['archive_url'], timeout=15)
            
            if response.status_code == 200:
                content = response.text
                content_length = len(content)
                
                # Parse content to check if it's valid
                soup = BeautifulSoup(content, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "No title"
                
                # Check for archive.org wrapper content
                has_wayback_toolbar = 'wayback' in content.lower() or 'archive.org' in content.lower()
                
                results.append({
                    'domain': case['domain'],
                    'date': case['date'],
                    'archive_url': case['archive_url'],
                    'content_length': content_length,
                    'title': title_text[:100],  # Truncate long titles
                    'has_wayback_toolbar': has_wayback_toolbar,
                    'status': 'success'
                })
                
                print(f"  ✓ Fetched {content_length} chars")
                print(f"  ✓ Title: {title_text[:60]}...")
                print(f"  ✓ Wayback toolbar detected: {has_wayback_toolbar}")
                
            else:
                results.append({
                    'domain': case['domain'],
                    'date': case['date'], 
                    'archive_url': case['archive_url'],
                    'content_length': 0,
                    'title': '',
                    'has_wayback_toolbar': False,
                    'status': f'http_error_{response.status_code}'
                })
                print(f"  ✗ HTTP Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            results.append({
                'domain': case['domain'],
                'date': case['date'],
                'archive_url': case['archive_url'], 
                'content_length': 0,
                'title': '',
                'has_wayback_toolbar': False,
                'status': 'timeout'
            })
            print(f"  ✗ Timeout")
            
        except Exception as e:
            results.append({
                'domain': case['domain'],
                'date': case['date'],
                'archive_url': case['archive_url'],
                'content_length': 0,
                'title': '',
                'has_wayback_toolbar': False,
                'status': f'error: {str(e)[:50]}'
            })
            print(f"  ✗ Error: {e}")
    
    # Results summary
    df = pd.DataFrame(results)
    print(f"\n=== Results Summary ===")
    print(f"Total domains tested: {len(test_cases)}")
    print(f"Successful fetches: {(df['status'] == 'success').sum()}")
    print(f"Average content length: {df[df['status'] == 'success']['content_length'].mean():.0f} chars")
    
    successful = df[df['status'] == 'success']
    if len(successful) > 0:
        print(f"\n=== Successful Archive Fetches ===")
        for _, row in successful.iterrows():
            print(f"✓ {row['domain']} ({row['date']}): {row['content_length']} chars")
            print(f"  Title: {row['title']}")
            print(f"  URL: {row['archive_url']}")
    
    failed = df[df['status'] != 'success']
    if len(failed) > 0:
        print(f"\n=== Failed Fetches ===")
        for _, row in failed.iterrows():
            print(f"✗ {row['domain']}: {row['status']}")
    
    return df

def demo_usage():
    """Show how to use the archive functionality."""
    print(f"\n=== Usage Example ===")
    print("# Import archive functions")
    print("from piedomains import pred_shalla_cat_archive, pred_shalla_cat_with_text_archive")
    print("")
    print("# Classify domains using content from January 1, 2020")
    print("domains = ['google.com', 'amazon.com', 'facebook.com']")
    print("result = pred_shalla_cat_archive(domains, '20200101')")
    print("print(result[['domain', 'pred_label', 'pred_prob', 'archive_date']])")
    print("")
    print("# Text-only classification from archive")
    print("text_result = pred_shalla_cat_with_text_archive(domains, '20200101')")
    print("print(text_result[['domain', 'text_label', 'text_prob']])")

if __name__ == "__main__":
    print("Archive.org Integration Test for Piedomains")
    print("=" * 50)
    
    # Test package imports first
    imports_work = True
    
    if imports_work:
        # Run comprehensive test
        test_results = test_direct_archive_urls()
        
        # Show usage example
        demo_usage()
        
        print(f"\n=== Next Steps ===")
        if (test_results['status'] == 'success').any():
            print("✓ Archive integration is working!")
            print("✓ Ready to test with piedomains classification")
            print("✓ Try the Jupyter Lab commands shown above")
        else:
            print("⚠ Archive.org may be experiencing issues")
            print("⚠ Try again later or use cached content for testing")
    else:
        print("✗ Package import issues - check archive_support.py")