### Piedomains

We use labeled data from Shallalist and the home page HTML data.

* [Shallalist](data/shallalist.gz)
* [Scraped home pages](data/html/)
	- some missing (URLs are no longer available) --- remove
	- some html files will be 0 KB so remove them. 
	- I would remove anything less than 5 KB as generally they are error pages
	
### Strategy ~ Text

1. HTML features
	- metadata tags
	- title
	- number of images (img tags)
	- number of links (a href tags)
	- h1, h2, h3 ....

2. Page size

3. text = soup.get_text() etc. and tokenize/common bi-grams/tri-grams etc. or embeddings



