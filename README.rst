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
1. **domain.pred_shalla_cat_with_text(input)**

  - What it does:
    - Predicts the kind of content hosted by a domain based on domain name and HTML of the homepage. The function can use locally stored HTML files or fetch fresh HTML files. If you specify a local folder, the function will look for HTML files corresponding to the domain. The HTML files must be stored as `domainname.html`. The function returns a pandas dataframe with label and corresponding probabilities.

 - Inputs:
    - `input`: list of domains. Either `input` or `html_path` must be specified.
    - `html_path`: path to the folder where the HTMLs are stored.  Either `input` or `html_path` must be specified. The function will by default look for a `html` folder on the same level as model files.
    - `latest`: use the latest model. Default is `True.`

  - Output:
    - Returns a pandas dataframe with label and probabilities

  Example - 
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

  Output -
  ::

                domain  text_label  text_prob  \
    0      xvideos.com        porn   0.918919   
    1  marketwatch.com     finance   0.627119   
    2       forbes.com        news   0.575000   
    3       bellesa.co        porn   0.962932   
    4     facebook.com  recreation   0.200815   
    5          last.fm       music   0.229545   

                                      text_domain_probs  used_domain_text  \
    0  {'adv': 0.001249639527059502, 'aggressive': 9....              True   
    1  {'adv': 0.001249639527059502, 'aggressive': 9....              True   
    2  {'adv': 0.010590500641848523, 'aggressive': 0....              True   
    3  {'adv': 0.00021545223423966907, 'aggressive': ...              True   
    4  {'adv': 0.006381039197812215, 'aggressive': 0....              True   
    5  {'adv': 0.002181818181818182, 'aggressive': 0....              True   

                                          extracted_text  
    0  xvideos furry ass history mature rough redhead...  
    1  marketwatch gold stocks video chrome economy v...  
    2  forbes featured leadership watch money breakin...  
    3  bellesa audio vixen sensual passionate orgy ki...  
    4    facebook watch messenger portal bulletin oculus  
    5  last twitter music reset company back merchand...  

2. **domain.pred_shalla_cat_with_images(input)**
  
  - What it does:
    - Predicts the kind of content hosted by a domain based on screenshot of the homepage.  The function can use locally stored screenshots files or fetch fresh screenshots of the homepage.  If you specify a local folder, the function will look for jpegs corresponding to the domain. The screenshots must be stored as `domainname.jpg`. The function returns a pandas dataframe with label and corresponding probabilities.

 - Inputs:
    - `input`: list of domains. Either `input` or `image_path` must be specified.
    - `image_path`: path to the folder where the screenshots are stored.  Either `input` or `image_path` must be specified. The function will by default look for a `images`` folder on the same level as model files.
    - `latest`: use the latest model. Default is `True.`

  - Output
    - Returns panda dataframe with label and probabilities

  Example - 
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

  Output -
  ::

                domain image_label  image_prob  \
    0       bellesa.co    shopping    0.366663   
    1     facebook.com        porn    0.284601   
    2  marketwatch.com  recreation    0.367953   
    3      xvideos.com        porn    0.916550   
    4       forbes.com  recreation    0.415165   
    5          last.fm    shopping    0.303097   

                                      image_domain_probs  used_domain_screenshot  
    0  {'adv': 0.0009261096129193902, 'aggressive': 3...                    True  
    1  {'adv': 0.030470917001366615, 'aggressive': 0....                    True  
    2  {'adv': 0.006861348636448383, 'aggressive': 0....                    True  
    3  {'adv': 0.0004964823601767421, 'aggressive': 0...                    True  
    4  {'adv': 0.0016061498317867517, 'aggressive': 8...                    True  
    5  {'adv': 0.007956285960972309, 'aggressive': 0....                    True  

3. **domain.pred_shalla_cat(input)**
  
  - What it does:
    - Predicts the kind of content hosted by a domain based on screenshot of the homepage.  The function can use locally stored screenshots and HTMLs or fetch fresh data.  If you specify local folders, the function will look for jpegs corresponding to the domain. The screenshots must be stored as `domainname.jpg`. The function returns a pandas dataframe with label and corresponding probabilities.

  - Inputs:
    - `input`: list of domains. Either `input` or `html_path` must be specified.
    - `html_path`: path to the folder where the screenshots are stored.  Either `input`, `image_path`, or `html_path` must be specified. The function will by default look for a `html` folder on the same level as model files.
    - `image_path`: path to the folder where the screenshots are stored.  Either `input`, `image_path`, or `html_path` must be specified. The function will by default look for a `images` folder on the same level as model files.
    - `latest`: use the latest model. Default is `True.`

  - Output
    - Returns panda dataframe with label and probabilities

  Example - 
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

  Output -
  ::

                  domain  text_label  text_prob  \
    0      xvideos.com        porn   0.918919   
    1  marketwatch.com     finance   0.627119   
    2       forbes.com        news   0.575000   
    3       bellesa.co        porn   0.962932   
    4     facebook.com  recreation   0.200815   
    5          last.fm       music   0.229545   

                                      text_domain_probs  used_domain_text  \
    0  {'adv': 0.001249639527059502, 'aggressive': 9....              True   
    1  {'adv': 0.001249639527059502, 'aggressive': 9....              True   
    2  {'adv': 0.010590500641848523, 'aggressive': 0....              True   
    3  {'adv': 0.00021545223423966907, 'aggressive': ...              True   
    4  {'adv': 0.006381039197812215, 'aggressive': 0....              True   
    5  {'adv': 0.002181818181818182, 'aggressive': 0....              True   

                                          extracted_text image_label  image_prob  \
    0  xvideos furry ass history mature rough redhead...        porn    0.916550   
    1  marketwatch gold stocks video chrome economy v...  recreation    0.370665   
    2  forbes featured leadership watch money breakin...  recreation    0.422517   
    3  bellesa audio vixen sensual passionate orgy ki...        porn    0.409875   
    4    facebook watch messenger portal bulletin oculus        porn    0.284601   
    5  last twitter music reset company back merchand...    shopping    0.420788   

                                      image_domain_probs  used_domain_screenshot  \
    0  {'adv': 0.0004964823601767421, 'aggressive': 0...                    True   
    1  {'adv': 0.007065971381962299, 'aggressive': 0....                    True   
    2  {'adv': 0.0016623957781121135, 'aggressive': 7...                    True   
    3  {'adv': 0.0008810096187517047, 'aggressive': 0...                    True   
    4  {'adv': 0.030470917001366615, 'aggressive': 0....                    True   
    5  {'adv': 0.01235155574977398, 'aggressive': 0.0...                    True   

          label  label_prob                              combined_domain_probs  
    0      porn    0.917735  {'adv': 0.0008730609436181221, 'aggressive': 0...  
    1   finance    0.315346  {'adv': 0.004157805454510901, 'aggressive': 0....  
    2      news    0.367533  {'adv': 0.006126448209980318, 'aggressive': 0....  
    3      porn    0.686404  {'adv': 0.0005482309264956868, 'aggressive': 0...  
    4      porn    0.223327  {'adv': 0.018425978099589416, 'aggressive': 0....  
    5  shopping    0.232422  {'adv': 0.007266686965796081, 'aggressive': 0....  


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