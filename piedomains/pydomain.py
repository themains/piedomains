from .logging import get_logger
from .base import Base

import tensorflow as tf
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.corpus import stopwords
import nltk
import re
import string


logger = get_logger()
nltk.download("stopwords")
nltk.download("words")
words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words("english"))


class Pydomain(Base):
    MODELFN = "model/shallalist"
    weights_loaded = False
    classes = [
        "adv",
        "aggressive",
        "alcohol",
        "automobile",
        "chat",
        "dating",
        "downloads",
        "drugs",
        "dynamic",
        "education",
        "finance",
        "fortunetelling",
        "forum",
        "gamble",
        "government",
        "hacking",
        "hobby",
        "hospitals",
        "imagehosting",
        "isp",
        "jobsearch",
        "models",
        "movies",
        "music",
        "news",
        "politics",
        "porn",
        "radiotv",
        "recreation",
        "redirector",
        "religion",
        "ringtones",
        "science",
        "searchengines",
        "sex",
        "shopping",
        "socialnet",
        "spyware",
        "tracker",
        "urlshortener",
        "warez",
        "weapons",
        "webmail",
        "webradio",
    ]

    @classmethod
    def tag_visible(cls, element):
        if element.parent.name in ["style", "script", "head", "title", "meta", "[document]"]:
            return False
        if isinstance(element, Comment):
            return False
        return True

    @classmethod
    def text_from_html(cls, text):
        soup = BeautifulSoup(text, "html.parser")
        texts = soup.findAll(text=True)
        visible_texts = filter(cls.tag_visible, texts)
        result = " ".join(t.strip().lower() for t in visible_texts if t.strip().isalpha())
        return " ".join(result.split())

    @classmethod
    def data_cleanup(cls, s):
        # remove numbers
        s = re.sub(r"\d+", "", s)
        # remove duplicates
        tokens = list(set(s.split()))
        # remove punctuation from each token
        table = str.maketrans("", "", string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove non english words
        tokens = [w.lower() for w in tokens if w.lower() in words]
        # remove non alpha
        tokens = [w.lower() for w in tokens if w.isalpha()]
        # remove non ascii
        tokens = [w.lower() for w in tokens if w.isascii()]
        # filter out stop words
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        return " ".join(w for w in tokens)

    @classmethod
    def pred_shalla_cat(cls, input, latest=False):
        """
        Predict category based on domain
        Args:
            input (str): list of domain names
        Returns:
            output (str): category
        """
        model_file_name = "shallalist_v1_model.tar.gz"
        if not cls.weights_loaded:
            cls.model_path = cls.load_model_data(model_file_name, latest)
            model = tf.keras.models.load_model(f"{cls.model_path}/saved_model/newpydomains")
            cls.weights_loaded = True

        input_content = input.copy()
        for i in range(len(input)):
            page = requests.get(f"https://{input[i]}", timeout=3, headers={"Accept-Language": "en-US"})
            text = cls.text_from_html(page.text)
            text = cls.data_cleanup(text)
            input_content[i] = input[i].rsplit(".", 1)[0] + " " + text

        results = model.predict(input_content)
        probs = tf.nn.softmax(results)
        res_args = tf.argmax(results, 1)

        labels = []
        domain_probs = []
        for i in range(len(input)):
            labels.append(cls.classes[res_args[i]])
            domain_probs.append(dict(zip(cls.classes, probs[i].numpy())))

        return pd.DataFrame(data={"name": input, "pred_label": labels, "domain_probs": domain_probs})