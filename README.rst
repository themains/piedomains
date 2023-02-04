===========================================================================================
piedomains: predict the kind of content hosted by a domain based on domain name and content
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


The package infers the kind of content hosted by domain using the domain name, and the content, and screenshot from the homepage. 

We use domain category labels from `Shallalist  <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZXTQ7V>`__ and build our own training dataset by scraping and taking screenshots of the homepage. The final dataset used to train the model is posted on the `Harvard Dataverse <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZXTQ7V>`__.  Python notebooks used to build the models can be found `here <https://github.com/themains/piedomains/tree/55cd5ea68ccec58ab2152c5f1d6fb9e6cf5df363/piedomains/notebooks>`__ and the model files can be found `here <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YHWCDC>`__

Installation
--------------
We strongly recommend installing `piedomains` inside a Python virtual environment
(see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install piedomains

General API
-----------
1. `domain.pred_shalla_cat_with_text`

  - What it does:
    - Predicts the kind of content hosted by a domain based on domain name and HTML of the homepage. 
      The function can use locally stored HTML files or fetch fresh HTML files. If you specify a local folder, 
      the function will look for HTML files corresponding to the domain. The HTML files must be stored as 
      `domainname.html`. The function returns a pandas dataframe with label and corresponding probabilities.

 - Inputs:
    - `input`: list of domains. Either `input` or `html_path` must be specified.
    - `html_path`: path to the folder where the HTMLs are stored. 
       Either `input` or `html_path` must be specified. The function will 
       by default look for a `html' folder on the same level as model files.
    - `latest`: use the latest model. Default is `True.`

  - Output:
    - Returns a pandas dataframe with label and probabilities

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
    # with only domains
    result = domain.pred_shalla_cat_with_text(domains)
    # with html path where htmls are stored (offline mode)
    result = domain.pred_shalla_cat_with_text(html_path="path/to/htmls")
    # with domains and html path, html_path will be used to store htmls
    result = domain.pred_shalla_cat_with_text(domains, html_path="path/to/htmls")
    print(result)

2. `domain.pred_shalla_cat_with_images`
  
  - What it does:
    - Predicts the kind of content hosted by a domain based on screenshot of the homepage. 
      The function can use locally stored screenshots files or fetch fresh screenshots of the homepage. 
      If you specify a local folder, the function will look for jpegs corresponding to the domain. The screenshots
      must be stored as `domainname.jpg`. The function returns a pandas dataframe with label and corresponding probabilities.

 - Inputs:
    - `input`: list of domains. Either `input` or `image_path` must be specified.
    - `image_path`: path to the folder where the screenshots are stored. 
       Either `input` or `image_path` must be specified. The function will 
       by default look for a `images' folder on the same level as model files.
    - `latest`: use the latest model. Default is `True.`

  - Output
    - Returns panda dataframe with label and probabilities

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
    # with only domains
    result = domain.pred_shalla_cat_with_images(domains)
    # with image path where images are stored (offline mode)
    result = domain.pred_shalla_cat_with_images(image_path="path/to/images")
    # with domains and image path, image_path will be used to store images
    result = domain.pred_shalla_cat_with_images(domains, image_path="path/to/images")
    print(result)

3. `domain.pred_shalla_cat`
  
  - What it does:
    - Predicts the kind of content hosted by a domain based on screenshot of the homepage. 
      The function can use locally stored screenshots and HTMLs or fetch fresh data. 
      If you specify local folders, the function will look for jpegs corresponding to the domain. The screenshots
      must be stored as `domainname.jpg`. The function returns a pandas dataframe with label and corresponding probabilities.

 - Inputs:
    - `input`: list of domains. Either `input` or `html_path` must be specified.
    - `html_path`: path to the folder where the screenshots are stored. 
       Either `input`, `image_path`, or `html_path` must be specified. The function will 
       by default look for a `html' folder on the same level as model files.
    - `image_path`: path to the folder where the screenshots are stored. 
       Either `input`, `image_path`, or `html_path` must be specified. The function will 
       by default look for a `images' folder on the same level as model files.
   - `latest`: use the latest model. Default is `True.`

  - Output
    - Returns panda dataframe with label and probabilities


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
    # with only domains
    result = domain.pred_shalla_cat(domains)
    # with html path where htmls are stored (offline mode)
    result = domain.pred_shalla_cat(html_path="path/to/htmls")
    # with image path where images are stored (offline mode)
    result = domain.pred_shalla_cat(image_path="path/to/images")
    print(result)

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