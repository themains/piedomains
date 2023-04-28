import re
import os
import time
import string
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import joblib
from nltk.corpus import stopwords

from .constants import classes, most_common_words
from .logging import get_logger
from .base import Base

logger = get_logger()
nltk.download("stopwords")
nltk.download("words")
words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words("english"))

"""
    Piedomain class
    This class is used to predict the category of a given url
"""


class Piedomain(Base):
    MODELFN = "model/shallalist"
    model_file_name = "shallalist_v5_model.tar.gz"
    weights_loaded = False
    img_width = 254
    img_height = 254

    """
    @classmethod
    text_from_html(text)
    @param text: html text
    @return: string
    This function is used to extract the text from the html
    """

    @classmethod
    def text_from_html(cls, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        result = " ".join(list(set([t.lower().strip() for t in text.split() if t.strip().isalpha()])))
        return result

    """
    @classmethod
    data_cleanup(s)
    @param s: string
    @return: string
    This function is used to clean the data
    """

    @classmethod
    def data_cleanup(cls, s: str) -> str:
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
        tokens = [w for w in tokens if w not in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        # remove most common words
        tokens = [w for w in tokens if w not in most_common_words]
        return " ".join(w for w in tokens)

    """
    @classmethod
    get_driver()
    @return: webdriver
    This function is used to get the webdriver
    """

    @classmethod
    def get_driver(cls):
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")  # linux only
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1280,1024")
        return webdriver.Chrome("chromedriver", options=options)

    """
    @classmethod
    save_image(domain, image_dir)
    @param domain: string
    @param image_dir: string
    @return: boolean
    This function is used to save the screenshot of the given domain
    """

    @classmethod
    def save_image(cls, domain: str, image_dir: str) -> bool:
        saved_screenshot = False
        driver = cls.get_driver()
        url = f"https://{domain}"
        try:
            driver.get(url)
            time.sleep(5)
            driver.save_screenshot(f"{image_dir}/{domain}.png")
            saved_screenshot = True
        except Exception as e:
            print(e)
        finally:
            driver.quit()
        return saved_screenshot

    """
    @classmethod
    extract_images(input, image_dir)
    @param input: list
    @param image_dir: string
    @return: list
    This function is used to extract the images from the given domains
    """

    @classmethod
    def extract_images(cls, input: list, use_cache: bool, image_dir: str) -> list:
        domains = input.copy()
        used_domain_screenshot = []
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for domain in domains:
            if use_cache and os.path.exists(f"{image_dir}/{domain}.png"):
                used_domain_screenshot.append(True)
                continue
            saved_screenshot = cls.save_image(domain, image_dir)
            if saved_screenshot:
                used_domain_screenshot.append(saved_screenshot)
        return used_domain_screenshot

    """
    @classmethod
    extract_image_tensor(image_dir)
    @param image_dir: string
    @return: dict
    This function is used to extract the image tensor from the given image directory
    """

    @classmethod
    def extract_image_tensor(cls, offline: bool, domains: list, image_dir: str) -> dict:
        images = {}
        for image in os.listdir(image_dir):
            domain_name = image.replace(".png", "")
            if (domain_name in domains) or offline:
                img_file = Image.open(f"{image_dir}/{image}")
                img_file = img_file.convert("RGB")
                img_tensor = tf.convert_to_tensor(np.array(img_file))
                img_tensor = tf.image.resize(img_tensor, [cls.img_width, cls.img_height])
                img_tensor = tf.cast(img_tensor, tf.float32)
                images[domain_name] = img_tensor
        return images

    """
    @classmethod
    extract_htmls(domains, html_path)
    @param domains: list of domains
    @param html_path: path to save html files
    @return errors: dictionary of errors
    """

    @classmethod
    def extract_htmls(cls, domains: list, use_cache: bool, html_path: string) -> dict:
        # check if html_path exists
        if not os.path.exists(html_path):
            os.mkdir(html_path)

        errors = {}
        for domain in domains:
            try:
                if use_cache:
                    if os.path.exists(f"{html_path}/{domain}.html"):
                        continue
                page = requests.get(f"https://{domain}", timeout=3, headers={"Accept-Language": "en-US"})
                f = open(f"{html_path}/{domain}.html", "w", encoding="utf-8")
                f.write(page.text)
                f.close()
            except Exception as e:
                errors[domain] = e
        return errors

    """
    @classmethod
    extract_html_text(html_path)
    @param html_path: path to html files
    @return domains: list of domains
    @return content: list of content
    """

    @classmethod
    def extract_html_text(cls, offline: bool, input: string, html_path: string):
        content = []
        domains = []
        for file in os.listdir(html_path):
            domain_name = file.replace(".html", "")
            if (domain_name in input) or offline:
                domains.append(domain_name)
                if file.endswith(".html"):
                    content.append(domain_name)
                    f = open(f"{html_path}/{file}", "r", encoding="utf-8")
                    text = cls.text_from_html(f.read())
                    text = cls.data_cleanup(text)
                    content[-1] = content[-1].rsplit(".", 1)[0] + " " + text
                    f.close()
        return domains, content

    """
    @classmethod
    load_model(model_file_name, latest=False)
    @param model_file_name: name of the model
    @param latest: if True, load the latest model
    @return: None
    This function is used to load the model
    """

    @classmethod
    def load_model(cls, model_file_name: string, latest: bool = False):
        if not cls.weights_loaded:
            cls.model_path = cls.load_model_data(model_file_name, latest)
            cls.model = tf.keras.models.load_model(f"{cls.model_path}/saved_model/piedomains")
            cls.model_cv = tf.keras.models.load_model(f"{cls.model_path}/saved_model/pydomains_images")

            # load calibrated models
            cls.calibrated_models = {}
            for c in classes:
                cls.calibrated_models[c] = joblib.load(f"{cls.model_path}/../calibrate/text/{c}.sav")

            cls.weights_loaded = True

    """
    @classmethod
    validate_input(input, path, type)
    @param input: list of domains
    @param path: path to html/image files
    @param type: type of input, html or image
    @return: bool
    This function is used to validate the input
    """

    @classmethod
    def validate_input(cls, input: list, path: string, type: string) -> bool:
        if type == "html":
            pth = "html_path"
        else:
            pth = "image_path"

        offline = False
        # if input is empty
        if len(input) == 0:
            # if path is None, raise exception
            if path is None:
                raise Exception(f"Provide list of Domains, or for offline provide {pth}")
            else:
                # if path is not None, check if it exists and is not empty
                if not os.path.exists(path):
                    raise Exception(f"{path} does not exist")
                if len(os.listdir(path)) == 0:
                    raise Exception(f"{path} is empty")
                else:
                    offline = True
        return offline

    """
    @classmethod
    pred_shalla_cat_with_text(input, html_path, latest)
    @param input: list of domains
    @param html_path: path to html files
    @param latest: if True, load the latest model
    @return dataframe with columns: domain, label, label_prob
    This function is used to predict the Shalla category of a domain
    """

    @classmethod
    def pred_shalla_cat_with_text(
        cls, input: list = [], html_path: string = None, use_cache: bool = True, latest: bool = True
    ) -> pd.DataFrame:
        offline_htmls = cls.validate_input(input, html_path, "html")
        cls.load_model(cls.model_file_name, latest)
        # if html_path is None then use the default path
        if html_path is None:
            html_path = f"{cls.model_path}/../html"
            print(f"html_path not provided using default path: {html_path}")
        domains = input.copy()

        if not offline_htmls:
            errors = cls.extract_htmls(domains, use_cache, html_path)
            if len(errors) > 0:
                for domain in errors:
                    print(f"Error: {domain} - {errors[domain]}")

        domains, content = cls.extract_html_text(offline_htmls, domains, html_path)
        results = cls.model.predict(content)
        probs = tf.nn.softmax(results)
        probs_df = pd.DataFrame(probs.numpy(), columns=classes)

        for c in classes:
            probs_df[c] = cls.calibrated_models[c].transform(probs_df[c].to_numpy())

        labels = probs_df.idxmax(axis=1).tolist()
        label_probs = probs_df.max(axis=1).tolist()
        domain_probs = probs_df.to_dict(orient="records")

        used_domain_content = [True] * len(domains)
        text_extract_errors = [""] * len(domains)

        if len(domains) != len(input):
            for domain in input:
                if domain not in domains:
                    domains.append(domain)
                    labels.append("None")
                    label_probs.append(0)
                    domain_probs.append({c: 0 for c in classes})
                    used_domain_content.append(False)
                    content.append("")
                    text_extract_errors.append(errors[domain])

        return pd.DataFrame(
            {
                "domain": domains,
                "text_label": labels,
                "text_prob": label_probs,
                "text_domain_probs": domain_probs,
                "used_domain_text": used_domain_content,
                "extracted_text": content,
                "text_extract_errors": text_extract_errors,
            }
        )

    """
    @classmethod
    pred_shalla_cat_with_images(input, image_path, latest)
    @param input: list of domains
    @param image_path: path to save images
    @param latest: if True, load the latest model
    @return dataframe with columns: domain, label, label_prob
    This function is used to predict the Shalla category of a domain
    """

    @classmethod
    def pred_shalla_cat_with_images(
        cls, input: list = [], image_path=None, use_cache: bool = True, latest: bool = True
    ) -> pd.DataFrame:
        offline_images = cls.validate_input(input, image_path, "image")
        cls.load_model(cls.model_file_name, latest)
        # if image_path is None then use the default path
        if image_path is None:
            image_path = f"{cls.model_path}/../images"
            print(f"image_path not provided using default path: {image_path}")

        domains = input.copy()
        if not offline_images:
            used_domain_screenshot = cls.extract_images(domains, use_cache, image_path)
        images = cls.extract_image_tensor(offline_images, domains, image_path)
        img_domains = list(images.keys())
        img_tensors = tf.stack(list(images.values()))
        if offline_images:
            used_domain_screenshot = [True] * len(img_domains)
        results = cls.model_cv.predict(img_tensors)
        probs = tf.nn.softmax(results)
        probs_df = pd.DataFrame(probs.numpy(), columns=classes)

        labels = probs_df.idxmax(axis=1).tolist()
        label_probs = probs_df.max(axis=1).tolist()
        domain_probs = probs_df.to_dict(orient="records")

        if len(img_domains) != len(input):
            for domain in input:
                if domain not in img_domains:
                    img_domains.append(domain)
                    labels.append("None")
                    label_probs.append(0)
                    domain_probs.append({c: 0 for c in classes})
                    used_domain_screenshot.append(False)

        return pd.DataFrame(
            {
                "domain": img_domains,
                "image_label": labels,
                "image_prob": label_probs,
                "image_domain_probs": domain_probs,
                "used_domain_screenshot": used_domain_screenshot,
            }
        )

    """
    @classmethod
    pred_shalla_cat(input, latest=False)
    @param input: list of domain names
    @param html_path: path to html files
    @param image_path: path to image files
    @param latest: if True, load latest model
    @return output: data frame with domain, category, and probability
    """

    @classmethod
    def pred_shalla_cat(
        cls, input: list = [], html_path=None, image_path=None, use_cache: bool = True, latest: bool = False
    ) -> pd.DataFrame:
        # text prediction
        pred_df = cls.pred_shalla_cat_with_text(input, html_path, use_cache, latest)
        # image prediction
        img_pred_df = cls.pred_shalla_cat_with_images(input, image_path, use_cache, latest)
        # merge predictions
        final_df = pred_df.merge(img_pred_df, on="domain", how="outer")
        # calculate final probabilities
        for c in classes:
            final_df[c] = final_df.apply(lambda x: (x["text_domain_probs"][c] + x["image_domain_probs"][c]) / 2, axis=1)
        final_df["label"] = final_df[classes].idxmax(axis=1)
        final_df["label_prob"] = final_df[classes].max(axis=1)
        final_df["combined_domain_probs"] = final_df[classes].to_dict(orient="records")
        final_df.drop(columns=classes, inplace=True)
        return final_df
