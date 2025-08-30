import re
import os
import time
import string
import requests
from urllib.parse import urlparse
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
from .config import get_config

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

    @staticmethod
    def validate_domain_name(domain: str) -> bool:
        """
        Validate if a domain name is properly formatted.
        
        Args:
            domain (str): Domain name to validate
            
        Returns:
            bool: True if domain is valid, False otherwise
        """
        if not domain or not isinstance(domain, str):
            return False
        
        # Remove protocol if present
        if domain.startswith(('http://', 'https://')):
            parsed = urlparse(domain)
            domain = parsed.netloc
        
        # Remove trailing slash and path
        domain = domain.split('/')[0]
        
        # Check for invalid characters (spaces, special chars except hyphen and dot)
        if ' ' in domain or any(c in domain for c in '!@#$%^&*()+=[]{}|\\:";\'<>?/'):
            return False
        
        # Must contain at least one dot to be a valid domain
        if '.' not in domain:
            return False
        
        # Check for consecutive dots
        if '..' in domain:
            return False
        
        # Cannot start or end with dot or hyphen
        if domain.startswith('.') or domain.endswith('.') or domain.startswith('-') or domain.endswith('-'):
            return False
        
        # Check length
        if len(domain) > 253:
            return False
        
        # Validate each part of the domain
        parts = domain.split('.')
        for part in parts:
            if not part or len(part) > 63:
                return False
            if part.startswith('-') or part.endswith('-'):
                return False
            if not re.match(r'^[a-zA-Z0-9\-]+$', part):
                return False
            
        return True

    @classmethod
    def validate_domains(cls, domains: list) -> tuple[list, list]:
        """
        Validate a list of domain names and separate valid from invalid.
        
        Args:
            domains (list): List of domain names to validate
            
        Returns:
            tuple: (valid_domains, invalid_domains)
        """
        valid_domains = []
        invalid_domains = []
        
        for domain in domains:
            if cls.validate_domain_name(domain):
                # Normalize domain (remove protocol, trailing slash, etc.)
                if domain.startswith(('http://', 'https://')):
                    parsed = urlparse(domain)
                    domain = parsed.netloc
                domain = domain.split('/')[0]
                valid_domains.append(domain)
            else:
                invalid_domains.append(domain)
                
        return valid_domains, invalid_domains

    @classmethod
    def text_from_html(cls, text: str) -> str:
        """
        Extract clean text content from HTML.
        
        Args:
            text (str): Raw HTML content
            
        Returns:
            str: Cleaned text with unique lowercase words
        """
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        result = " ".join(list(set([t.lower().strip() for t in text.split() if t.strip().isalpha()])))
        return result

    @classmethod
    def data_cleanup(cls, s: str) -> str:
        """
        Clean and normalize text data for model input.
        
        Args:
            s (str): Raw text string
            
        Returns:
            str: Cleaned text with English words only, no stopwords or common terms
        """
        if not isinstance(s, str):
            raise AttributeError("Input must be a string")
        
        # remove numbers
        s = re.sub(r"\d+", "", s)
        # remove duplicates
        tokens = list(set(s.split()))
        # remove punctuation from each token
        table = str.maketrans("", "", string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove non alpha first
        tokens = [w.lower() for w in tokens if w.isalpha()]
        # remove non ascii
        tokens = [w.lower() for w in tokens if w.isascii()]
        # remove non english words
        tokens = [w for w in tokens if w in words]
        # filter out stop words
        tokens = [w for w in tokens if w not in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        # remove most common words
        tokens = [w for w in tokens if w not in most_common_words]
        return " ".join(w for w in tokens)

    @classmethod
    def get_driver(cls):
        """
        Get configured Chrome WebDriver instance for screenshots.
        
        Returns:
            webdriver.Chrome: Headless Chrome driver with optimized settings
        """
        from webdriver_manager.chrome import ChromeDriverManager
        config = get_config()
        
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-extensions")
        options.add_argument("--no-sandbox")  # linux only
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--window-size={config.webdriver_window_size}")
        options.add_argument(f"--user-agent={config.user_agent}")
        
        return webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=options)


    @classmethod
    def save_image(cls, domain: str, image_dir: str) -> tuple[bool, str]:
        """
        Save screenshot of domain homepage.
        
        Args:
            domain (str): Domain name to screenshot
            image_dir (str): Directory to save screenshot
            
        Returns:
            tuple[bool, str]: (success, error_message)
        """
        from .context_managers import webdriver_context
        config = get_config()
        
        try:
            with webdriver_context() as driver:
                url = f"https://{domain}"
                driver.set_page_load_timeout(config.page_load_timeout)
                driver.get(url)
                time.sleep(config.screenshot_wait_time)
                driver.save_screenshot(f"{image_dir}/{domain}.png")
                return True, ""
        except Exception as e:
            error_msg = f"Failed to screenshot {domain}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


    @classmethod
    def extract_images(cls, input: list, use_cache: bool, image_dir: str) -> tuple[list, dict]:
        """
        Extract screenshots for domains.
        
        Args:
            input (list): List of domains
            use_cache (bool): Whether to use cached screenshots
            image_dir (str): Directory to save screenshots
            
        Returns:
            tuple[list, dict]: (used_domain_screenshot, screenshot_errors)
        """
        domains = input.copy()
        used_domain_screenshot = []
        screenshot_errors = {}
        
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            
        for domain in domains:
            if use_cache and os.path.exists(f"{image_dir}/{domain}.png"):
                used_domain_screenshot.append(True)
                continue
                
            success, error_msg = cls.save_image(domain, image_dir)
            used_domain_screenshot.append(success)
            if not success:
                screenshot_errors[domain] = error_msg
                
        return used_domain_screenshot, screenshot_errors

    @classmethod
    def extract_image_tensor(cls, offline: bool, domains: list, image_dir: str) -> dict:
        """
        Convert PNG images to TensorFlow tensors for model input.
        
        Args:
            offline (bool): Whether to process all images in directory
            domains (list): List of domain names to process
            image_dir (str): Directory containing PNG files
            
        Returns:
            dict: Dictionary mapping domain names to image tensors
        """
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


    @classmethod
    def extract_htmls(cls, domains: list, use_cache: bool, html_path: str) -> dict:
        """
        Extract HTML content from domain homepages.
        
        Args:
            domains (list): List of domain names
            use_cache (bool): Whether to use cached HTML files
            html_path (str): Directory to save HTML files
            
        Returns:
            dict: Dictionary of errors encountered {domain: error_message}
        """
        # check if html_path exists
        if not os.path.exists(html_path):
            os.makedirs(html_path, exist_ok=True)

        config = get_config()
        errors = {}
        
        for domain in domains:
            try:
                if use_cache and os.path.exists(f"{html_path}/{domain}.html"):
                    continue
                    
                headers = {
                    "User-Agent": config.user_agent,
                    "Accept-Language": "en-US,en;q=0.9"
                }
                
                # Retry logic with exponential backoff
                last_exception = None
                for attempt in range(config.max_retries + 1):
                    try:
                        page = requests.get(
                            f"https://{domain}", 
                            timeout=config.http_timeout, 
                            headers=headers,
                            allow_redirects=True
                        )
                        page.raise_for_status()  # Raise exception for bad status codes
                        
                        with open(f"{html_path}/{domain}.html", "w", encoding="utf-8") as f:
                            f.write(page.text)
                        break  # Success, exit retry loop
                        
                    except (requests.exceptions.RequestException, IOError) as e:
                        last_exception = e
                        if attempt < config.max_retries:
                            wait_time = config.retry_delay * (2 ** attempt)  # Exponential backoff
                            logger.debug(f"Retrying HTML fetch for {domain} in {wait_time}s (attempt {attempt + 1}/{config.max_retries + 1})")
                            time.sleep(wait_time)
                        else:
                            # All retries exhausted, raise the last exception
                            raise last_exception
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"HTTP request failed: {str(e)}"
                errors[domain] = error_msg
                logger.error(f"Failed to fetch HTML for {domain}: {error_msg}")
            except IOError as e:
                error_msg = f"File I/O error: {str(e)}"
                errors[domain] = error_msg
                logger.error(f"Failed to save HTML for {domain}: {error_msg}")
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                errors[domain] = error_msg
                logger.error(f"Unexpected error for {domain}: {error_msg}")
                
        return errors

    @classmethod
    def extract_html_text(cls, offline: bool, input: str, html_path: str):
        """
        Extract and clean text content from HTML files.
        
        Args:
            offline (bool): Whether to process all HTML files in directory
            input (str): List of domain names to process
            html_path (str): Directory containing HTML files
            
        Returns:
            tuple: (domains, content) - lists of domain names and cleaned text
        """
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

    @classmethod
    def load_model(cls, model_file_name: str, latest: bool = False):
        """
        Load TensorFlow models and calibrators from local cache or download from server.
        
        Args:
            model_file_name (str): Name of the model file to load
            latest (bool): Whether to download the latest model version
            
        Note:
            Loads both text and image models plus isotonic regression calibrators.
            Models are cached locally after first download.
        """
        if not cls.weights_loaded:
            logger.info(f"Loading models (latest={latest})")
            cls.model_path = cls.load_model_data(model_file_name, latest)
            
            logger.info("Loading text-based TensorFlow model")
            cls.model = tf.keras.models.load_model(f"{cls.model_path}/saved_model/piedomains")
            
            logger.info("Loading image-based TensorFlow model")
            cls.model_cv = tf.keras.models.load_model(f"{cls.model_path}/saved_model/pydomains_images")

            # load calibrated models
            logger.info(f"Loading {len(classes)} calibration models")
            cls.calibrated_models = {}
            for c in classes:
                cls.calibrated_models[c] = joblib.load(f"{cls.model_path}/../calibrate/text/{c}.sav")

            cls.weights_loaded = True
            logger.info("All models loaded successfully")
        else:
            logger.debug("Models already loaded, skipping")

    @classmethod
    def validate_input(cls, input: list, path: str, type: str) -> bool:
        """
        Validate input parameters for prediction functions.
        
        Args:
            input (list): List of domain names
            path (str): Path to HTML or image files
            type (str): Input type - 'html' or 'image'
            
        Returns:
            bool: True if operating in offline mode (using local files only)
            
        Raises:
            Exception: If neither domains nor valid path provided
        """
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

    @classmethod
    def _process_text_batch(cls, domains_batch: list, html_path: str, use_cache: bool) -> tuple[list, list, dict]:
        """
        Process a batch of domains for text extraction.
        
        Args:
            domains_batch (list): Batch of domains to process
            html_path (str): Path to HTML files
            use_cache (bool): Whether to use cached files
            
        Returns:
            tuple: (domains, content, errors)
        """
        if not domains_batch:
            return [], [], {}
            
        errors = cls.extract_htmls(domains_batch, use_cache, html_path)
        domains, content = cls.extract_html_text(False, domains_batch, html_path)
        
        return domains, content, errors

    @classmethod
    def pred_shalla_cat_with_text(
        cls, input: list = [], html_path: str = None, use_cache: bool = True, latest: bool = True
    ) -> pd.DataFrame:
        """
        Predict domain categories using text content from homepage HTML.
        
        Args:
            input (list): List of domain names to classify
            html_path (str): Path to directory with HTML files (optional)
            use_cache (bool): Whether to reuse existing HTML files
            latest (bool): Whether to download latest model version
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                - domain: Domain name
                - text_label: Predicted category
                - text_prob: Confidence probability
                - text_domain_probs: All category probabilities
                - used_domain_text: Whether text was successfully extracted
                - extracted_text: Cleaned text content
                - text_extract_errors: Any extraction errors
        """
        offline_htmls = cls.validate_input(input, html_path, "html")
        cls.load_model(cls.model_file_name, latest)
        
        # Validate domain names if not in offline mode
        if not offline_htmls and input:
            logger.info(f"Validating {len(input)} input domains for text prediction")
            valid_domains, invalid_domains = cls.validate_domains(input)
            if invalid_domains:
                logger.warning(f"Invalid domains found and will be skipped: {invalid_domains}")
            logger.info(f"Proceeding with {len(valid_domains)} valid domains")
            domains = valid_domains
        else:
            domains = input.copy()
            logger.info(f"Using offline mode or direct domain list: {len(domains)} domains")
            
        # if html_path is None then use the default path
        if html_path is None:
            html_path = f"{cls.model_path}/../html"
            print(f"html_path not provided using default path: {html_path}")

        config = get_config()
        all_domains = []
        all_content = []
        all_errors = {}
        
        if not offline_htmls and len(domains) > config.batch_size:
            logger.info(f"Processing {len(domains)} domains in batches of {config.batch_size}")
            
            # Process domains in batches to manage memory usage
            for i in range(0, len(domains), config.batch_size):
                batch = domains[i:i + config.batch_size]
                logger.info(f"Processing batch {i//config.batch_size + 1}/{(len(domains)-1)//config.batch_size + 1} ({len(batch)} domains)")
                
                batch_domains, batch_content, batch_errors = cls._process_text_batch(batch, html_path, use_cache)
                
                all_domains.extend(batch_domains)
                all_content.extend(batch_content)
                all_errors.update(batch_errors)
                
                if batch_errors:
                    logger.error(f"Failed to extract HTML for {len(batch_errors)} domains in batch")
            
            domains, content = all_domains, all_content
            errors = all_errors
        else:
            if not offline_htmls:
                logger.info(f"Extracting HTML content for {len(domains)} domains")
                errors = cls.extract_htmls(domains, use_cache, html_path)
                if len(errors) > 0:
                    logger.error(f"Failed to extract HTML for {len(errors)} domains")
                    for domain in errors:
                        logger.error(f"HTML extraction error for {domain}: {errors[domain]}")

            logger.info("Processing HTML text content")
            domains, content = cls.extract_html_text(offline_htmls, domains, html_path)
        
        logger.info(f"Successfully processed text for {len(domains)} domains")
        
        if not content:
            logger.warning("No text content available for prediction")
            return pd.DataFrame()
        
        logger.info("Running text-based model prediction")
        # Process predictions in batches to manage memory
        if len(content) > config.batch_size:
            logger.info(f"Running prediction in batches of {config.batch_size}")
            all_results = []
            for i in range(0, len(content), config.batch_size):
                batch_content = content[i:i + config.batch_size]
                batch_results = cls.model.predict(batch_content)
                all_results.append(batch_results)
                # Clear intermediate results to free memory
                del batch_results
            results = np.concatenate(all_results, axis=0)
            del all_results  # Free memory
        else:
            results = cls.model.predict(content)
            
        probs = tf.nn.softmax(results)
        probs_df = pd.DataFrame(probs.numpy(), columns=classes)
        
        # Clear large tensors to free memory
        del results, probs

        logger.info("Applying model calibration")
        for c in classes:
            probs_df[c] = cls.calibrated_models[c].transform(probs_df[c].to_numpy())

        labels = probs_df.idxmax(axis=1).tolist()
        label_probs = probs_df.max(axis=1).tolist()
        domain_probs = probs_df.to_dict(orient="records")
        
        logger.info(f"Text prediction completed for {len(domains)} domains")

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

    @classmethod
    def pred_shalla_cat_with_images(
        cls, input: list = [], image_path=None, use_cache: bool = True, latest: bool = True
    ) -> pd.DataFrame:
        """
        Predict domain categories using homepage screenshots.
        
        Args:
            input (list): List of domain names to classify
            image_path (str): Path to directory with screenshot images (optional)
            use_cache (bool): Whether to reuse existing screenshot files
            latest (bool): Whether to download latest model version
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                - domain: Domain name
                - image_label: Predicted category
                - image_prob: Confidence probability
                - image_domain_probs: All category probabilities
                - used_domain_screenshot: Whether screenshot was captured
        """
        offline_images = cls.validate_input(input, image_path, "image")
        cls.load_model(cls.model_file_name, latest)
        
        # Validate domain names if not in offline mode
        if not offline_images and input:
            logger.info(f"Validating {len(input)} input domains for image prediction")
            valid_domains, invalid_domains = cls.validate_domains(input)
            if invalid_domains:
                logger.warning(f"Invalid domains found and will be skipped: {invalid_domains}")
            logger.info(f"Proceeding with {len(valid_domains)} valid domains")
            domains = valid_domains
        else:
            domains = input.copy()
            logger.info(f"Using offline mode or direct domain list: {len(domains)} domains")
            
        # if image_path is None then use the default path
        if image_path is None:
            image_path = f"{cls.model_path}/../images"
            print(f"image_path not provided using default path: {image_path}")
        screenshot_errors = {}
        if not offline_images:
            logger.info(f"Capturing screenshots for {len(domains)} domains")
            used_domain_screenshot, screenshot_errors = cls.extract_images(domains, use_cache, image_path)
            if screenshot_errors:
                logger.error(f"Failed to capture screenshots for {len(screenshot_errors)} domains")
                logger.warning(f"Screenshot errors: {screenshot_errors}")
        
        logger.info("Processing image tensors")
        images = cls.extract_image_tensor(offline_images, domains, image_path)
        img_domains = list(images.keys())
        logger.info(f"Successfully processed images for {len(img_domains)} domains")
        
        if len(img_domains) == 0:
            logger.warning("No images available for prediction")
            return pd.DataFrame()
        
        config = get_config()    
        if offline_images:
            used_domain_screenshot = [True] * len(img_domains)
            
        logger.info("Running image-based model prediction")
        
        # Process image predictions in batches to manage memory
        img_tensors_list = list(images.values())
        if len(img_tensors_list) > config.batch_size:
            logger.info(f"Running image prediction in batches of {config.batch_size}")
            all_results = []
            
            for i in range(0, len(img_tensors_list), config.batch_size):
                batch_tensors = img_tensors_list[i:i + config.batch_size]
                batch_tensor_stack = tf.stack(batch_tensors)
                batch_results = cls.model_cv.predict(batch_tensor_stack)
                all_results.append(batch_results)
                
                # Clear intermediate tensors to free memory
                del batch_tensor_stack, batch_results
                
            results = np.concatenate(all_results, axis=0)
            del all_results  # Free memory
        else:
            img_tensors = tf.stack(img_tensors_list)
            results = cls.model_cv.predict(img_tensors)
            del img_tensors  # Free memory
            
        # Clear the images dict to free memory
        del images, img_tensors_list
        
        probs = tf.nn.softmax(results)
        probs_df = pd.DataFrame(probs.numpy(), columns=classes)
        
        # Clear large tensors to free memory
        del results, probs

        labels = probs_df.idxmax(axis=1).tolist()
        label_probs = probs_df.max(axis=1).tolist()
        domain_probs = probs_df.to_dict(orient="records")
        
        logger.info(f"Image prediction completed for {len(img_domains)} domains")

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

    @classmethod
    def pred_shalla_cat(
        cls, input: list = [], html_path=None, image_path=None, use_cache: bool = True, latest: bool = False
    ) -> pd.DataFrame:
        """
        Predict domain categories using combined text and image analysis.
        
        This is the main prediction function that combines both text-based
        and image-based classification for improved accuracy.
        
        Args:
            input (list): List of domain names to classify
            html_path (str): Path to directory with HTML files (optional)
            image_path (str): Path to directory with screenshot images (optional)
            use_cache (bool): Whether to reuse existing files
            latest (bool): Whether to download latest model version
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                - domain: Domain name
                - text_label, text_prob: Text-based prediction and confidence
                - image_label, image_prob: Image-based prediction and confidence
                - label, label_prob: Final ensemble prediction and confidence
                - combined_domain_probs: Averaged probabilities across all categories
                - used_domain_text, used_domain_screenshot: Success flags
                - extracted_text: Cleaned HTML text content
        """
        logger.info(f"Starting combined prediction for {len(input) if input else 'offline'} domains")
        
        # text prediction
        logger.info("Running text-based prediction")
        pred_df = cls.pred_shalla_cat_with_text(input, html_path, use_cache, latest)
        
        # image prediction  
        logger.info("Running image-based prediction")
        img_pred_df = cls.pred_shalla_cat_with_images(input, image_path, use_cache, latest)
        
        # merge predictions
        logger.info("Combining text and image predictions")
        final_df = pred_df.merge(img_pred_df, on="domain", how="outer")
        
        # calculate final probabilities
        logger.info("Computing ensemble probabilities")
        for c in classes:
            final_df[c] = final_df.apply(lambda x: (x["text_domain_probs"][c] + x["image_domain_probs"][c]) / 2, axis=1)
        final_df["label"] = final_df[classes].idxmax(axis=1)
        final_df["label_prob"] = final_df[classes].max(axis=1)
        final_df["combined_domain_probs"] = final_df[classes].to_dict(orient="records")
        final_df.drop(columns=classes, inplace=True)
        
        logger.info(f"Combined prediction completed for {len(final_df)} domains")
        return final_df
