import os
import pandas as pd
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

driver = None

df = pd.read_csv("fulldomain_min_greater_than_5_words_v3.csv.gz", usecols=["full_domain"])

for i, r in df.iterrows():
    fn = f"png/{i}.png"
    # url = 'http://www.' + r.domain
    url = "http://" + r.full_domain
    if not os.path.exists(fn):
        print(i, url)
        try:
            if driver is None:
                options = webdriver.ChromeOptions()
                options.add_argument("--disable-extensions")
                options.add_argument("--no-sandbox")  # linux only
                options.add_argument("--headless")
                options.add_argument("--window-size=1280,1024")
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver.implicitly_wait(5)  # seconds
                driver.set_page_load_timeout(5)
            driver.get(url)
            driver.save_screenshot(fn)
            # open png image
            img_png = Image.open(fn)
            # save as jpg image
            img_png.save(fn.replace(".png", ".jpg"))
            # remove png image
            os.unlink(fn)
        except Exception as e:
            print("ERROR:", i, url, e)
            if str(e).find("invalid session id"):
                try:
                    driver.quit()
                except Exception as e:
                    print(e)
                driver = None
