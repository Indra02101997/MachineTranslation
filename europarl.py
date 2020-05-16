import os
import download
data_dir = "data/europarl/"

data_url = "http://www.statmt.org/europarl/v7/"

def maybe_download_and_extract(language_code="da"):

    url = data_url + language_code + "-en.tgz"

    download.maybe_download_and_extract(url=url, download_dir=data_dir)


def load_data(english=True, language_code="da", start="", end=""):

    if english:
        filename = "europarl-v7.{0}-en.en".format(language_code)
    else:
        filename = "europarl-v7.{0}-en.{0}".format(language_code)

    path = os.path.join(data_dir, filename)

    with open(path, encoding="utf-8") as file:
        texts = [start + line.strip() + end for line in file]

    return texts

