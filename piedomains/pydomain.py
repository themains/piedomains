from .logging import get_logger
from .base import Base

import re
import os
import string
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from selenium import webdriver
from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import joblib
from nltk.corpus import stopwords


logger = get_logger()
nltk.download("stopwords")
nltk.download("words")
words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words("english"))
most_common_words = [
    "home",
    "contact",
    "us",
    "new",
    "news",
    "site",
    "privacy",
    "search",
    "help",
    "copyright",
    "free",
    "service",
    "en",
    "get",
    "one",
    "find",
    "menu",
    "account",
    "next",
]


class Pydomain(Base):
    MODELFN = "model/shallalist"
    weights_loaded = False
    img_width = 160
    img_height = 160

    classes = [
        "adv",
        "aggressive",
        "alcohol",
        "anonvpn",
        "automobile",
        "costtraps",
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
        "homestyle",
        "hospitals",
        "imagehosting",
        "isp",
        "jobsearch",
        "library",
        "military",
        "models",
        "movies",
        "music",
        "news",
        "podcasts",
        "politics",
        "porn",
        "radiotv",
        "recreation",
        "redirector",
        "religion",
        "remotecontrol",
        "ringtones",
        "science",
        "searchengines",
        "sex",
        "shopping",
        "socialnet",
        "spyware",
        "tracker",
        "updatesites",
        "urlshortener",
        "violence",
        "warez",
        "weapons",
        "webmail",
        "webphone",
        "webradio",
        "webtv",
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
        # remove most common words
        tokens = [w for w in tokens if not w in most_common_words]
        return " ".join(w for w in tokens)

    @classmethod
    def get_image_tensor(cls, domain):
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")  # linux only
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1280,1024")
        driver = webdriver.Chrome("chromedriver", options=options)
        # driver.implicitly_wait(5)  # seconds
        # driver.set_page_load_timeout(5)

        url = f"https://{domain}"
        driver.get(url)

        png_file = url.replace("https://", "") + ".png"
        driver.save_screenshot(png_file)

        jpg_file = png_file.replace(".png", ".jpg")
        img_file = Image.open(png_file)
        img_file = img_file.convert("RGB")
        img_file.save(jpg_file)
        img = Image.open(jpg_file)
        img_tensor = tf.convert_to_tensor(np.array(img))
        img_tensor = tf.image.resize(img_tensor, [cls.img_width, cls.img_height])
        os.unlink(png_file)
        os.unlink(jpg_file)
        return img_tensor

    @classmethod
    def pred_shalla_cat(cls, input, latest=False):
        """
        Predict category based on domain
        Args:
            input (str): list of domain names
        Returns:
            output (str): category
        """
        model_file_name = "shallalist_v4_model.tar.gz"
        if not cls.weights_loaded:
            cls.model_path = cls.load_model_data(model_file_name, latest)
            cls.model = tf.keras.models.load_model(f"{cls.model_path}/saved_model/piedomains")
            cls.model_cv = tf.keras.models.load_model(f"{cls.model_path}/saved_model/pydomains_images")

            # load calibrated models
            cls.calibrated_models = {}
            for c in cls.classes:
                cls.calibrated_models[c] = joblib.load(f"{cls.model_path}/../calibrate/text/{c}.sav")

            cls.weights_loaded = True

        input_content = input.copy()
        used_domain_content = []
        # text extraction
        for i in range(len(input)):
            try:
                page = requests.get(f"https://{input[i]}", timeout=3, headers={"Accept-Language": "en-US"})
                text = cls.text_from_html(page.text)
                text = cls.data_cleanup(text)
                used_domain_content.append(True)
            except Exception as e:
                text = ""
                used_domain_content.append(False)
            input_content[i] = input[i].rsplit(".", 1)[0] + " " + text

        # text prediction
        # print(input_content)
        results = cls.model.predict(input_content)
        probs = tf.nn.softmax(results)
        probs_df = pd.DataFrame(probs.numpy(), columns=cls.classes)

        for c in cls.classes:
            probs_df[c] = cls.calibrated_models[c].transform(probs_df[c].to_numpy())

        labels = probs_df.idxmax(axis=1)
        label_probs = probs_df.max(axis=1)
        domain_probs = probs_df.to_dict(orient="records")

        img_tensors = []
        used_domain_screenshot = []
        # image extraction
        for i in range(len(input)):
            try:
                img_tensor = cls.get_image_tensor(input[i])
                img_tensor = tf.cast(img_tensor, tf.float32)
                img_tensors.append(img_tensor)
                used_domain_screenshot.append(True)
            except Exception as e:
                print(e)
                img_tensors.append(tf.zeros((cls.img_width, cls.img_height, 3)))
                used_domain_screenshot.append(False)

        # image prediction
        img_tensors = tf.stack(img_tensors)
        img_results = cls.model_cv.predict(img_tensors)
        img_probs = tf.nn.softmax(img_results)
        img_probs_df = pd.DataFrame(img_probs.numpy(), columns=cls.classes)

        img_labels = img_probs_df.idxmax(axis=1)
        img_label_probs = img_probs_df.max(axis=1)
        img_domain_probs = img_probs_df.to_dict(orient="records")

        return pd.DataFrame(
            data={
                "name": input,
                "text_pred_label": labels,
                "text_label_prob": label_probs,
                "img_pred_label": img_labels,
                "img_label_prob": img_label_probs,
                "used_domain_content": used_domain_content,
                "used_domain_screenshot": used_domain_screenshot,
                "text_domain_probs": domain_probs,
                "img_domain_probs": img_domain_probs,
            }
        )
