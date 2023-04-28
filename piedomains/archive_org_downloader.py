import requests
import json
from bs4 import BeautifulSoup

def get_urls_year(domain):
  # Set the URL for the Wayback Machine API
  url = 'https://web.archive.org/cdx/search?url=' + domain + '&matchType=prefix&filter=statuscode:200&from=20140101&to=20141231&output=json'
  # Send a GET request to the Wayback Machine API
  response = requests.get(url)
  # Parse the JSON content of the response
  data = json.loads(response.content.decode('utf-8'))
  # Extract the URLs from the response
  urls = [f'https://web.archive.org/web/{row[1]}/{row[2]}' for row in data]
  
  return(urls)

def download_from_archive_org(url):
  # Send a GET request to the archive.org snapshot
  response = requests.get(url)
  
  # Parse the HTML content of the page using BeautifulSoup
  soup = BeautifulSoup(response.content, "html.parser")
  
  # Find all the text on the page and print it
  text = soup.get_text()
  
  return(text)
