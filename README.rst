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

                  name text_pred_label  text_label_prob img_pred_label  \
  0       forbes.com            news         0.575000     recreation   
  1      xvideos.com            porn         0.897716           porn   
  2          last.fm           music         0.229545       shopping   
  3     facebook.com      recreation         0.200815           porn   
  4       bellesa.co            porn         0.962932       shopping   
  5  marketwatch.com         finance         0.790576     recreation   

    img_label_prob  used_domain_content  used_domain_screenshot  \
  0        0.911997                 True                    True   
  1        0.755726                 True                    True   
  2        0.416521                 True                    True   
  3        0.274597                 True                    True   
  4        0.374870                 True                    True   
  5        0.366329                 True                    True   

                                    text_domain_probs  \
  0  {'adv': 0.010590500641848523, 'aggressive': 0....   
  1  {'adv': 0.002181818181818182, 'aggressive': 9....   
  2  {'adv': 0.002181818181818182, 'aggressive': 0....   
  3  {'adv': 0.006381039197812215, 'aggressive': 0....   
  4  {'adv': 0.00021545223423966907, 'aggressive': ...   
  5  {'adv': 0.0007271669575334497, 'aggressive': 9...   

                                      img_domain_probs  
  0  {'adv': 9.541013423586264e-05, 'aggressive': 1...  
  1  {'adv': 0.00041423083166591823, 'aggressive': ...  
  2  {'adv': 0.008832501247525215, 'aggressive': 0....  
  3  {'adv': 0.027437569573521614, 'aggressive': 0....  
  4  {'adv': 0.0008953566430136561, 'aggressive': 3...  
  5  {'adv': 0.007870808243751526, 'aggressive': 0....

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