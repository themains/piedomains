===========================================================================================
piedomains: AI-powered domain content classification
===========================================================================================

.. image:: https://github.com/themains/piedomains/actions/workflows/python-publish.yml/badge.svg
    :target: https://github.com/themains/piedomains/actions/workflows/python-publish.yml
.. image:: https://img.shields.io/pypi/v/piedomains.svg
    :target: https://pypi.python.org/pypi/piedomains
.. image:: https://readthedocs.org/projects/piedomains/badge/?version=latest
    :target: http://piedomains.readthedocs.io/en/latest/?badge=latest

**piedomains** predicts website content categories using AI analysis of domain names, text content, and homepage screenshots. Classify domains as news, shopping, adult content, education, etc. with high accuracy.

🚀 **Quickstart**
-------------------

Install and classify domains in 3 lines:

.. code-block:: python

    pip install piedomains
    
    from piedomains import DomainClassifier
    classifier = DomainClassifier()
    
    # Classify current content
    result = classifier.classify(["cnn.com", "amazon.com", "wikipedia.org"])
    print(result[['domain', 'pred_label', 'pred_prob']])
    
    # Expected output:
    #        domain    pred_label  pred_prob
    # 0     cnn.com          news   0.876543
    # 1  amazon.com      shopping   0.923456
    # 2 wikipedia.org   education   0.891234

📊 **Key Features**
--------------------

- **High Accuracy**: Combines text analysis + visual screenshots for 90%+ accuracy
- **Historical Analysis**: Classify websites from any point in time using archive.org
- **Fast & Scalable**: Batch processing with caching for 1000s of domains
- **Easy Integration**: Modern Python API with pandas output
- **41 Categories**: From news/finance to adult/gambling content

⚡ **Usage Examples**
---------------------

**Basic Classification**

.. code-block:: python

    from piedomains import DomainClassifier
    
    classifier = DomainClassifier()
    
    # Combined analysis (most accurate)
    result = classifier.classify(["github.com", "reddit.com"])
    
    # Text-only (faster)
    result = classifier.classify_by_text(["news.google.com"])
    
    # Images-only (good for visual content)  
    result = classifier.classify_by_images(["instagram.com"])

**Historical Analysis**

.. code-block:: python

    # Analyze how Facebook looked in 2010 vs today
    old_facebook = classifier.classify(["facebook.com"], archive_date="20100101")
    new_facebook = classifier.classify(["facebook.com"])
    
    print(f"2010: {old_facebook.iloc[0]['pred_label']}")
    print(f"2024: {new_facebook.iloc[0]['pred_label']}")

**Batch Processing**

.. code-block:: python

    # Process large lists efficiently
    domains = ["site1.com", "site2.com", ...] # 1000s of domains
    results = classifier.classify_batch(
        domains, 
        method="text",           # text|images|combined
        batch_size=50,           # Process 50 at a time
        show_progress=True       # Progress bar
    )

🏷️ **Supported Categories**
------------------------------

News, Finance, Shopping, Education, Government, Adult Content, Gambling, Social Networks, Search Engines, and 32 more categories based on the Shallalist taxonomy.

📈 **Performance**
-------------------

- **Speed**: ~10-50 domains/minute (depends on method and network)
- **Accuracy**: 85-95% depending on content type and method
- **Memory**: <500MB for batch processing
- **Caching**: Automatic content caching for faster re-runs

🔧 **Installation**
--------------------

**Requirements**: Python 3.9+

.. code-block:: bash

    # Basic installation
    pip install piedomains
    
    # For development
    git clone https://github.com/themains/piedomains
    cd piedomains
    pip install -e .

🔄 **Migration from v0.2.x**
-----------------------------

**Old API** (still supported):

.. code-block:: python

    from piedomains import domain
    result = domain.pred_shalla_cat_with_text(["example.com"])

**New API** (recommended):

.. code-block:: python

    from piedomains import DomainClassifier
    classifier = DomainClassifier()
    result = classifier.classify_by_text(["example.com"])

📖 **Documentation**
---------------------

- **API Reference**: https://piedomains.readthedocs.io
- **Examples**: `/examples` directory
- **Notebooks**: `/piedomains/notebooks` (training & analysis)

🤝 **Contributing**
--------------------

.. code-block:: bash

    # Setup development environment
    git clone https://github.com/themains/piedomains
    cd piedomains
    pip install -e ".[dev]"
    
    # Run tests
    pytest piedomains/tests/ -v
    
    # Run linting
    flake8 piedomains/

📄 **License**
---------------

MIT License - see LICENSE file.

📚 **Citation**
----------------

If you use piedomains in research, please cite:

.. code-block:: bibtex

    @software{piedomains,
      title={piedomains: AI-powered domain content classification},
      author={Chintalapati, Rajashekar and Sood, Gaurav},
      year={2024},
      url={https://github.com/themains/piedomains}
    }

---

**Legacy Documentation**
========================

For legacy API documentation, see LEGACY_API.rst
