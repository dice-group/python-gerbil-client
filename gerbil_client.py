#!/usr/bin/env python3
from bs4 import BeautifulSoup
import json
import logging
import requests
import time

logger = logging.getLogger(__name__)


class GerbilBase():
    def upload_file(self, *, name, path, data):
        if path is not None:
            logger.debug('Uploading file: %s', path)
            with open(path, 'rb') as f:
                return self._upload_file(name=name, upload=f)
        else:
            logger.debug('Uploading data: %s', data)
            return self._upload_file(name=name, upload=data)

    def _upload_file(self, *, name, upload):
        res = requests.post(self.gerbil_url + 'file/upload', data={'name': name}, files=[('files', (name, upload, 'text/plain'))])
        res.raise_for_status()
        upload_name = res.json()['files'][0]['name']
        logger.debug('Upload name: %s', upload_name)
        return upload_name

    def result(self, experiment_id):
        'Returns the experiment results, will block until the experiment is done.'
        start_time = time.monotonic_ns()
        while True:
            logger.debug('Retrieving results for experiment: %s', experiment_id)
            r = requests.get(self.gerbil_url + 'experiment', params={'id': experiment_id}, timeout=10)
            r.raise_for_status()
            bs = BeautifulSoup(r.text, 'html.parser')
            if (warn := bs.find('span', {'class': 'gerbil-experiment-warn'})) is not None:
                logger.warning('Server message: %s', warn.text.strip())
            data_str = bs.find('script', type='application/ld+json').string
            if len(data_str.strip()) == 0:
                return None
            data = json.loads(data_str)
            # TODO: return sub-experiments as well
            observations = list(res for res in data['@graph'] if res['@type'] == 'qb:Observation' and 'subExperimentOf' not in res)
            if len(observations) != 1:
                raise ValueError(f'{len(observations)=}; {observations=}')
            observation = observations[0]
            status = int(observation['statusCode'])
            if status < -100: # various errors
                if (error := bs.select_one('#resultTable td[colspan]')) is not None:
                    logger.error('Experiment %s failed with error %d: %s', experiment_id, status, error.text.strip())
                else:
                    logger.error('Experiment %s failed with error %d', experiment_id, status)
                return None
            elif status == -2: # TASK_NOT_FOUND
                logger.error('Experiment %s not found', experiment_id)
                return None
            elif status == -1: # TASK_STARTED_BUT_NOT_FINISHED_YET
                logger.debug('Experiment %s not finished (%ds)', experiment_id, (time.monotonic_ns() - start_time) / 10**9)
                pass
            elif status == 0: # TASK_FINISHED
                if any(k not in observation for k in self.expected_observation_metrics):
                    # https://github.com/dice-group/gerbil/issues/435
                    logger.warning('Got qb:Observation with statusCode=0 but no experiment results')
                else:
                    # FIXME: proper json-ld handling
                    for met in self.expected_observation_metrics:
                        observation[met] = float(observation[met])
                    logger.debug('Experiment %s result: %s', experiment_id, observation)
                    return observation
            else:
                logger.error('Experiment %s returned unknown status: %d', experiment_id, status)
                return None
            # FIXME: configurable/exponential
            time.sleep(1)

    def submit(self, *, experiment_type=None, system_file=None, system_data=None, dataset_file=None, dataset_data=None, **kwargs):
        '''Execute an experiment.
        Specify either system_file as a path to the file or system_data as a string containing the content, the same for dataset.
        Returns the experiment ID.
        '''

        experiment_data = {
            'type': experiment_type or self.default_experiment_type,
        }
        dataset_name = 'dataset'
        dataset_upload = self.upload_file(name=dataset_name, path=dataset_file, data=dataset_data)
        system_name = 'system'
        system_upload = self.upload_file(name=system_name, path=system_file, data=system_data)
        self._prepare_experiment(
            experiment_data=experiment_data,
            dataset_name=dataset_name,
            dataset_upload=dataset_upload,
            system_name=system_name,
            system_upload=system_upload,
            **kwargs,
        )
        logger.debug('Submitting an experiment: %s', experiment_data)
        r = requests.get(self.gerbil_url + 'execute', params={'experimentData': json.dumps(experiment_data)}, timeout=10)
        logger.debug('Response (%s): %s', r.status_code, r.text)
        r.raise_for_status()
        return r.text


class QA(GerbilBase):
    'Client for https://github.com/dice-group/gerbil/tree/QuestionAnsweringQALD10'
    default_experiment_type = 'QA'
    expected_observation_metrics = ['macroF1', 'macroPrecision', 'macroRecall', 'microF1', 'microPrecision', 'microRecall']

    def __init__(self, *, gerbil_url='https://gerbil-qa.aksw.org/gerbil/'):
        'Initialize the QA client with a default or custom URL'
        self.gerbil_url = gerbil_url

    def _prepare_experiment(self, *, experiment_data, dataset_name, dataset_upload, system_name, system_upload, matching='STRONG_ENTITY_MATCH', lang=''):
        experiment_data.update({
            'matching': matching,
            'annotator': [],
            'questionLanguage': lang,
            'dataset': [f'NIFDS_{dataset_name}({dataset_upload})'],
            'answerFiles': [f'AF_{system_name}({system_upload})(undefined)(AFDS_{dataset_upload})'],
        })

    def submit(self, **kwargs):
        if 'system_answers' in kwargs:
            kwargs['system_data'] = json.dumps({'questions': [{
                'id': 0,
                'question': [{'language': 'en', 'string': '?'}],
                'answers': [{'head': {'vars': ['result']}, 'results': {'bindings': [{'result': a} for a in kwargs['system_answers']]}}]
            }]})
            del kwargs['system_answers']
        if 'dataset_answers' in kwargs:
            kwargs['dataset_data'] = json.dumps({'questions': [{
                'id': 0,
                'question': [{'language': 'en', 'string': '?'}],
                'answers': [{'head': {'vars': ['result']}, 'results': {'bindings': [{'result': a} for a in kwargs['dataset_answers']]}}]
            }]})
            del kwargs['dataset_answers']
        return super().submit(**kwargs)


class BENG(GerbilBase):
    'Client for https://github.com/dice-group/BENG'
    default_experiment_type = 'NLG'
    expected_observation_metrics = ['BLEU', 'BLEU_NLTK', 'METEOR', 'TER']

    def __init__(self, *, gerbil_url='https://beng.dice-research.org/gerbil/'):
        'Initialize the BENG client with a default or custom URL'
        self.gerbil_url = gerbil_url

    def _prepare_experiment(self, *, experiment_data, dataset_name, dataset_upload, system_name, system_upload, lang=''):
        experiment_data.update({
            'candidate': [],
            'language': lang,
            'dataset': [f'NIFDS_{dataset_name}({dataset_upload})'],
            'hypothesis': [f'HF_{system_name}({system_upload})'],
        })
