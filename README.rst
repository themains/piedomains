==================================================
Newpydomains: Predict category based on domain and its content
==================================================

This package used `Shallalist dataset <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZXTQ7V>`__ to train the model.
Scrapped homepages of the domains mentioned in above dataset. This package predicts the category based on the domain name and its content.

Install
-------
We strongly recommend installing `newpydomains` inside a Python virtual environment
(see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install newpydomains

General API
-----------
1. domain.classify will take array of domains and predicts category.

Examples
--------
::

  from newpydomains import domain
  domains = ['youtube.com', 'netflix.com', 'yahoo.com', 'khanacademy.org', 'medium.com']
  result = domain.classify(domains)
  print(result)

Output -
::
              name  pred_label  domain_probs
0      youtube.com  recreation           3.0
1      netflix.com  recreation           3.0
2        yahoo.com       forum           3.0
3  khanacademy.org  recreation           3.0
4       medium.com  recreation           3.0

Functions
----------
We expose 1 function, which will take array of domains and predicts category.

- **domain.classify(input)**

  - What it does:

    - predicts category based on domain and its content

  - Output

    - Returns panda dataframe with label and probabilities

Authors
-------

Rajashekar Chintalapati and Gaurav Sood

Contributor Code of Conduct
---------------------------------

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the `Contributor Code of
Conduct <http://contributor-covenant.org/version/1/0/0/>`__.

License
----------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.