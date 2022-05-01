==================================================
piedomains: Predict category based on domain and its content
==================================================


.. image:: https://app.travis-ci.com/themains/piedomains.svg?branch=master
    :target: https://travis-ci.org/themains/piedomains
.. image:: https://ci.appveyor.com/api/projects/status/5wkr850yy3f6sg6a?svg=true
    :target: https://ci.appveyor.com/project/themains/piedomains
.. image:: https://img.shields.io/pypi/v/piedomains.svg
    :target: https://pypi.python.org/pypi/piedomains
.. image:: https://readthedocs.org/projects/piedomains/badge/?version=latest
    :target: http://piedomains.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://pepy.tech/badge/piedomains
    :target: https://pepy.tech/project/piedomains


This package used `Shallalist dataset <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZXTQ7V>`__ to train the model.
Scrapped homepages of the domains mentioned in above dataset. This package predicts the category based on the domain name and its content.

Install
-------
We strongly recommend installing `piedomains` inside a Python virtual environment
(see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install piedomains

General API
-----------
1. domain.classify will take array of domains and predicts category.

Examples
--------
::

  from piedomains import domain
  domains = ['youtube.com', 'netflix.com', 'yahoo.com', 'khanacademy.org', 'medium.com']
  result = domain.classify(domains)
  print(result)

Output -
::
                name  pred_label                                       domain_probs
  0      youtube.com  recreation  {'adv': 0.02274143, 'aggressive': 0.022215988,...
  1      netflix.com  recreation  {'adv': 0.022540696, 'aggressive': 0.02221061,...
  2        yahoo.com       forum  {'adv': 0.022722345, 'aggressive': 0.022219377...
  3  khanacademy.org  recreation  {'adv': 0.022936365, 'aggressive': 0.022226635...
  4       medium.com  recreation  {'adv': 0.022473775, 'aggressive': 0.022233406...


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
