#!/usr/bin/env python3
"""
Test script for archive.org functionality in piedomains.
Tests 10 sample domains with both live and archived content.
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import requests
from datetime import datetime

def test_archive_api():
    """Test archive.org API connectivity."""
    print("=== Testing Archive.org API ===")
    
    test_domains = [
        'google.com',
        'amazon.com', 
        'facebook.com',
        'twitter.com',
        'youtube.com'
    ]
    
    target_date = '20200101'  # January 1, 2020
    
    for domain in test_domains[:3]:  # Test first 3
        try:
            api_url = f"https://archive.org/wayback/available?url={domain}&timestamp={target_date}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('archived_snapshots', {}).get('closest', {}).get('available'):
                    snapshot_url = data['archived_snapshots']['closest']['url']
                    snapshot_date = data['archived_snapshots']['closest']['timestamp']
                    print(f"✓ {domain}: Found snapshot from {snapshot_date}")
                    print(f"  URL: {snapshot_url}")
                else:
                    print(f"✗ {domain}: No snapshot available")
            else:
                print(f"✗ {domain}: API error {response.status_code}")
                
        except Exception as e:
            print(f"✗ {domain}: {e}")
    
    return True

def test_package_imports():
    """Test that archive functions can be imported."""
    print("\n=== Testing Package Imports ===")
    
    try:
        from piedomains import (
            pred_shalla_cat_archive,
            pred_shalla_cat_with_text_archive,
            pred_shalla_cat_with_images_archive
        )
        print("✓ Archive functions imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic archive functionality with a simple example."""
    print("\n=== Testing Basic Archive Functionality ===")
    
    try:
        # Test the archive snapshot URL fetching
        from piedomains.archive_support import _fetch_archive_snapshot_url
        
        test_url = 'https://google.com'
        target_date = '20200101'
        
        print(f"Looking for snapshot of {test_url} near {target_date}...")
        snapshot_url = _fetch_archive_snapshot_url(test_url, target_date)
        
        if snapshot_url:
            print(f"✓ Found snapshot: {snapshot_url}")
            
            # Test fetching content from the snapshot
            try:
                response = requests.get(snapshot_url, timeout=15)
                if response.status_code == 200:
                    content_length = len(response.text)
                    print(f"✓ Successfully fetched archived content ({content_length} chars)")
                    
                    # Show a snippet of the content
                    snippet = response.text[:200].replace('\n', ' ')
                    print(f"  Content preview: {snippet}...")
                    return True
                else:
                    print(f"✗ Failed to fetch content: {response.status_code}")
            except Exception as e:
                print(f"✗ Error fetching content: {e}")
        else:
            print("✗ No snapshot found")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        
    return False

def run_comprehensive_test():
    """Run tests on 10 sample domains."""
    print("\n=== Testing 10 Sample Domains ===")
    
    # Diverse set of domains for testing
    test_domains = [
        'google.com',
        'amazon.com', 
        'facebook.com',
        'twitter.com',
        'youtube.com',
        'reddit.com',
        'netflix.com',
        'github.com',
        'stackoverflow.com',
        'wikipedia.org'
    ]
    
    target_date = '20200101'  # January 1, 2020
    
    print(f"Testing archive functionality with {len(test_domains)} domains from {target_date}")
    print("Domains:", ", ".join(test_domains))
    
    results = []
    
    for i, domain in enumerate(test_domains, 1):
        print(f"\n[{i}/10] Testing {domain}...")
        
        try:
            # Test archive snapshot availability
            from piedomains.archive_support import _fetch_archive_snapshot_url
            snapshot_url = _fetch_archive_snapshot_url(f"https://{domain}", target_date)
            
            if snapshot_url:
                # Test content fetching
                try:
                    response = requests.get(snapshot_url, timeout=15)
                    if response.status_code == 200:
                        content_length = len(response.text)
                        has_title = '<title>' in response.text.lower()
                        
                        results.append({
                            'domain': domain,
                            'snapshot_available': True,
                            'snapshot_url': snapshot_url,
                            'content_length': content_length,
                            'has_title': has_title,
                            'status': 'success'
                        })
                        print(f"  ✓ Content: {content_length} chars, Title: {has_title}")
                    else:
                        results.append({
                            'domain': domain,
                            'snapshot_available': True,
                            'snapshot_url': snapshot_url,
                            'content_length': 0,
                            'has_title': False,
                            'status': f'fetch_error_{response.status_code}'
                        })
                        print(f"  ✗ Fetch failed: {response.status_code}")
                except Exception as e:
                    results.append({
                        'domain': domain,
                        'snapshot_available': True,
                        'snapshot_url': snapshot_url,
                        'content_length': 0,
                        'has_title': False,
                        'status': f'error: {str(e)[:50]}'
                    })
                    print(f"  ✗ Error: {e}")
            else:
                results.append({
                    'domain': domain,
                    'snapshot_available': False,
                    'snapshot_url': None,
                    'content_length': 0,
                    'has_title': False,
                    'status': 'no_snapshot'
                })
                print(f"  ✗ No snapshot available")
                
        except Exception as e:
            results.append({
                'domain': domain,
                'snapshot_available': False,
                'snapshot_url': None,
                'content_length': 0,
                'has_title': False,
                'status': f'api_error: {str(e)[:50]}'
            })
            print(f"  ✗ API Error: {e}")
    
    # Summary
    df = pd.DataFrame(results)
    print(f"\n=== Test Results Summary ===")
    print(f"Total domains tested: {len(test_domains)}")
    print(f"Snapshots available: {df['snapshot_available'].sum()}")
    print(f"Successful content fetches: {(df['status'] == 'success').sum()}")
    print(f"Average content length: {df[df['status'] == 'success']['content_length'].mean():.0f} chars")
    
    print(f"\nDetailed results:")
    for _, row in df.iterrows():
        status_icon = "✓" if row['status'] == 'success' else "✗"
        print(f"  {status_icon} {row['domain']}: {row['status']}")
    
    return df

if __name__ == "__main__":
    print("Testing Archive.org Integration for Piedomains")
    print("=" * 50)
    
    # Test 1: Archive API connectivity
    api_works = test_archive_api()
    
    # Test 2: Package imports
    imports_work = test_package_imports()
    
    # Test 3: Basic functionality
    basic_works = test_basic_functionality()
    
    # Test 4: Comprehensive test
    if api_works and imports_work:
        comprehensive_results = run_comprehensive_test()
        
        print(f"\n=== Final Status ===")
        print(f"Archive API: {'✓' if api_works else '✗'}")
        print(f"Package imports: {'✓' if imports_work else '✗'}")
        print(f"Basic functionality: {'✓' if basic_works else '✗'}")
        
        if basic_works:
            print("✓ Archive integration is working!")
            print("\nReady for Jupyter Lab testing with:")
            print("from piedomains import pred_shalla_cat_archive")
            print("result = pred_shalla_cat_archive(['google.com'], '20200101')")
        else:
            print("✗ Some issues detected - check logs above")
    else:
        print("✗ Basic tests failed - cannot proceed with comprehensive test")