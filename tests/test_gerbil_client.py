#!/usr/bin/env python3
import logging
import os
import requests
import unittest
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
import gerbil_client  # noqa: E402

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)


def data_path(name):
    return os.path.join(os.path.dirname(__file__), 'data', name)


class GerbilClientTestCase(unittest.TestCase):
    dbp_example = 'http://dbpedia.org/resource/Moscow'
    dbp_lang_examples = [
        'http://de.dbpedia.org/resource/Moskau',
        'http://es.dbpedia.org/resource/Moscú',
        'http://fr.dbpedia.org/resource/Moscou',
        'http://ru.dbpedia.org/resource/Москва',
    ]
    wd_localname = 'Q649'
    wd_example = 'http://www.wikidata.org/entity/' + wd_localname

    def test_BENG(self):
        instance = gerbil_client.BENG()
        requests.get(f'{instance.gerbil_url}running').raise_for_status()
        eids = [instance.submit(
            dataset_file=data_path('beng_dataset.txt'),
            system_file=data_path(f'beng_system_{i}.txt')
        ) for i in range(3)]
        for eid in eids:
            res = instance.result(eid)
            assert res is not None
            assert 'BLEU' in res
            assert 'BLEU_NLTK' in res
            assert 'METEOR' in res
            assert 'TER' in res
            assert 'Error_Count' in res

    def test_QA(self):
        instance = gerbil_client.QA()
        requests.get(f'{instance.gerbil_url}running').raise_for_status()
        eids = [instance.submit(
            dataset_file=data_path('qa_dataset.txt'),
            system_file=data_path(f'qa_system_{i}.txt')
        ) for i in range(3)]
        for eid in eids:
            res = instance.result(eid)
            assert res is not None
            assert 'macroF1' in res
            assert 'macroPrecision' in res
            assert 'macroRecall' in res
            assert 'microF1' in res
            assert 'microPrecision' in res
            assert 'microRecall' in res
