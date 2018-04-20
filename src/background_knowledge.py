import re
import zipfile
from lxml import etree

import numpy as np


def process_ind(text, stopwords):
    """
    Apply a simple tokenizer and remove stopwords

    :return: set of content words
    """
    if text is None:
        ind = set()
    else:
        processed = set(filter(None, re.split("[, !?:.]+", text.lower())))  # should also try just leaving as string
        ind = processed - stopwords

    return ind


def read_drugbank(f="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/drugbank/drugbank_all_full_database.xml.zip"):
    z = zipfile.ZipFile(f)
    z_h = z.open(z.infolist()[0])
    root = etree.fromstring(z_h.read())
    el_pref = "{http://www.drugbank.ca}"  # element name prefix
    stopwords = set()
    with open("stopwords.txt") as in_f:
        for l in in_f:
            stopwords.add(l.strip().lower())

    drug_to_id = {}
    id_to_indication = {}
    id_to_adr = {}

    for i, drug in enumerate(root):
        name = ""
        synonyms, products = [], []
        indication, adr = set(), set()
        for el in drug:
            if el.tag == el_pref + "name":
                if el.text not in drug_to_id:
                    name = el.text
                else:
                    continue
            if el.tag == el_pref + "synonyms":
                for e in el:
                    if e.text not in drug_to_id:
                        synonyms.append(e.text)
                    else:
                        continue
            if el.tag == el_pref + "products":
                for prod in el:
                    for e in prod:
                        if e.tag == el_pref + "name":
                            if e.text not in drug_to_id:
                                products.append(e.text)
                            else:
                                continue
            if el.tag == el_pref + "indication":
                indication = process_ind(el.text, stopwords)
            if el.tag == el_pref + "snp-adverse-drug-reactions":  # use this as a separate feat?
                for reaction in el:
                    for e in reaction:
                        if e.tag == el_pref + "description":
                            adr = process_ind(e.text, stopwords)

        if not indication:
            continue
        for drug in [name] + synonyms + products:
            drug_to_id[drug.lower()] = i
        id_to_indication[i] = indication
        id_to_adr[i] = adr

    return drug_to_id, id_to_indication, id_to_adr


def drugbank_compa(prob, treat, drug_to_id, id_to_info):
    """

    :param prob: list of words
    :param treat: list of words
    :param drug_to_id:
    :param id_to_info: id_to_indication or id_to_adr
    :return:
    """
    for w_treat in treat:
        if w_treat in drug_to_id:
            if set(prob) & id_to_info[drug_to_id[w_treat]]:
                return 1.

    return -1.


def compatibility(c1, c2, c1t, c2t, rel, drug_to_id, id_to_info):
    """
    :param c1: list of words constituting the first concept
    :param c2: list of words constituting a concept
    :param c1t: c1 type
    :param c2t: c2 type
    :param rel: relation name

    :return: a compatibility score
    """
    if c1t == "problem" and c2t == "treatment":
        compa = drugbank_compa(c1, c2, drug_to_id, id_to_info)
    elif c1t == "treatment" and c2t == "problem":
        compa = drugbank_compa(c2, c1, drug_to_id, id_to_info)
    else:
        compa = -1.

    return compa


def semclass(txt, lexicon, rel, lemmatizer=None):
    """
    :param txt:
    :param lexicon:
    :param lemmatizer: WordNetLemmatizer from NLTK
    :return:
    """
    f_vec = np.array([-1.] * len(lexicon.keys()))  # trp->5
    mapping = dict(zip(lexicon.keys(), range(len(lexicon.keys()))))
    for w in txt:
        for sem_cl, syns in lexicon.items():
            if lemmatizer is not None:
                w = lemmatizer.lemmatize(w.lower(), pos="v")
            if w in syns:
                f_vec[mapping[sem_cl]] = 1.

    return f_vec


if __name__ == "__main__":
    drug_to_id, id_to_indication, id_to_adr = read_drugbank()
