===========================================================================================
piedomains: Predict the kind of content hosted by a domain based on domain name and content
===========================================================================================

.. image:: https://ci.appveyor.com/api/projects/status/k0b72xay9i4ufxff?svg=true
    :target: https://ci.appveyor.com/project/soodoku/piedomains
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
1. domain.pred_shalla_cat will take array of domains and predicts category.

Examples
--------
::

  from piedomains import domain
  domains = [
      "yahoo.com",
      "forbes.com",
      "xvideos.com",
      "last.fm",
      "facebook.com",
      "bellesa.co",
      "marketwatch.com"
  ]
  result = domain.pred_shalla_cat(domains)
  print(result)

Output -
::

                name  pred_label  label_prob  used_domain_content                                   all_domain_probs
  0        yahoo.com       forum    0.020146                 True  {'adv': 0.01915626, 'aggressive': 0.01823743, ...
  1       forbes.com        news    0.024546                 True  {'adv': 0.019166455, 'aggressive': 0.018233137...
  2      xvideos.com        porn    0.023876                 True  {'adv': 0.021233741, 'aggressive': 0.018176463...
  3          last.fm       music    0.021811                 True  {'adv': 0.018456697, 'aggressive': 0.018200265...
  4     facebook.com        news    0.020649                 True  {'adv': 0.018711654, 'aggressive': 0.01822204,...
  5       bellesa.co        porn    0.022348                 True  {'adv': 0.019042958, 'aggressive': 0.018210754...
  6  marketwatch.com     finance    0.024208                 True  {'adv': 0.018741421, 'aggressive': 0.01818002,...

Functions
----------
We expose 1 function, which will take array of domains and predicts category.

- **domain.pred_shalla_cat(input)**

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
the `Contributor Code of Conduct <http://contributor-covenant.org/version/1/0/0/>`__.

License
----------
The package is released under the `MIT License <https://opensource.org/licenses/MIT>`__.