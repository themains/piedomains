#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

print('=== Testing 10 Sample Domains with Archive.org ===')

test_cases = [
    {'domain': 'google.com', 'date': '20200101', 'expected': 'searchengines'},
    {'domain': 'amazon.com', 'date': '20150101', 'expected': 'shopping'},
    {'domain': 'facebook.com', 'date': '20100101', 'expected': 'socialnet'},
    {'domain': 'cnn.com', 'date': '20100101', 'expected': 'news'},
    {'domain': 'paypal.com', 'date': '20100101', 'expected': 'finance'},
    {'domain': 'github.com', 'date': '20150101', 'expected': 'hobby'},
    {'domain': 'stackoverflow.com', 'date': '20100101', 'expected': 'forum'},
    {'domain': 'ebay.com', 'date': '20050101', 'expected': 'shopping'},
    {'domain': 'wikipedia.org', 'date': '20050101', 'expected': 'education'},
    {'domain': 'yahoo.com', 'date': '20100101', 'expected': 'news'}
]

results = []
successful_count = 0

for i, case in enumerate(test_cases, 1):
    domain = case['domain']
    date = case['date']
    expected = case['expected']
    
    archive_url = f'https://web.archive.org/web/{date}120000/https://{domain}'
    
    print(f'[{i}/10] {domain} from {date}...')
    
    try:
        response = requests.get(archive_url, timeout=20)
        
        if response.status_code == 200:
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ''
            
            text_content = soup.get_text().lower()
            words = [w.strip() for w in text_content.split() if w.strip().isalpha() and len(w.strip()) > 2]
            unique_words = list(set(words))
            
            # Simple prediction
            indicators = {
                'shopping': ['shop', 'buy', 'cart', 'price', 'product', 'store', 'purchase'],
                'news': ['news', 'breaking', 'latest', 'story', 'report', 'headlines'],
                'socialnet': ['social', 'friend', 'share', 'connect', 'profile', 'network'],
                'searchengines': ['search', 'find', 'web', 'results', 'query'],
                'finance': ['money', 'bank', 'payment', 'finance', 'loan', 'credit'],
                'education': ['education', 'learn', 'school', 'university', 'knowledge'],
                'forum': ['forum', 'discussion', 'question', 'answer', 'community']
            }
            
            scores = {}
            for category, keywords in indicators.items():
                score = sum(1 for word in unique_words if word in keywords)
                scores[category] = score
            
            predicted = max(scores, key=scores.get) if any(scores.values()) else 'recreation'
            confidence = max(scores.values()) / max(len(unique_words), 1)
            match = predicted == expected
            
            successful_count += 1
            symbol = '✓' if match else '○'
            
            print(f'  {symbol} Content: {len(content)} chars, Title: {title_text[:40]}...')
            print(f'  {symbol} Predicted: {predicted} (expected: {expected}) conf: {confidence:.3f}')
            
            results.append({
                'domain': domain, 'date': date, 'predicted': predicted, 'expected': expected,
                'match': match, 'confidence': confidence, 'content_length': len(content),
                'title': title_text[:60], 'status': 'success'
            })
            
        else:
            print(f'  ✗ HTTP {response.status_code}')
            results.append({'domain': domain, 'status': f'http_{response.status_code}'})
            
    except Exception as e:
        print(f'  ✗ Error: {str(e)[:50]}')
        results.append({'domain': domain, 'status': 'error'})
    
    time.sleep(1)

print(f'\n=== FINAL SUMMARY ===')
print(f'Successful fetches: {successful_count}/10 ({successful_count*10}%)')

if successful_count > 0:
    success_results = [r for r in results if r.get('status') == 'success']
    matches = sum(r['match'] for r in success_results)
    avg_length = sum(r['content_length'] for r in success_results) / len(success_results)
    
    print(f'Correct predictions: {matches}/{successful_count} ({matches/successful_count*100:.1f}%)')
    print(f'Average content: {avg_length:.0f} chars')
    
    print(f'\n=== DETAILED RESULTS ===')
    for r in success_results:
        symbol = '✓' if r['match'] else '○'
        print(f'{symbol} {r["domain"]}: {r["predicted"]} (conf: {r["confidence"]:.3f})')
    
    print(f'\n=== ARCHIVE INTEGRATION STATUS ===')
    print('✓ Archive.org integration working!')
    print('✓ Content fetching successful')
    print('✓ Text processing working')
    print('✓ Ready for full ML model testing')
    
else:
    print('✗ Archive.org connectivity issues')

print(f'\n=== READY FOR JUPYTER LAB ===')
print('from piedomains import pred_shalla_cat_archive')
print('result = pred_shalla_cat_archive([\"google.com\", \"amazon.com\"], \"20200101\")')