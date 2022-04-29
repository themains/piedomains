### Piedomains

We use labeled data from Shallalist and the home page HTML data.

* [Shallalist](data/shallalist.gz)
* [Scraped home pages](data/html/)
	- some missing (URLs are no longer available) --- remove
	- some html files will be 0 KB so remove them. 
	- I would remove anything less than 5 KB as generally they are error pages

* [UK Classification Data](data/classification.tsv)
	- https://data.webarchive.org.uk/opendata/ukwa.ds.1/classification/

* [Crowdflower Dataset](data/URL-categorization-DFE.csv)
	- https://data.world/crowdflower/url-categorization

* [DMOZ data]()
	- https://www.kaggle.com/datasets/shawon10/url-classification-dataset-dmoz?resource=download

* [UNB data]()
	- https://www.unb.ca/cic/datasets/url-2016.html

### Strategy ~ Text

1. HTML features
	- metadata tags
	- title
	- number of images (img tags)
	- number of links (a href tags)
	- h1, h2, h3 ....

2. Page size

3. text = soup.get_text() etc. and tokenize/common bi-grams/tri-grams etc. or embeddings



