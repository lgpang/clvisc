#!/usr/bin/env python
# -*- utf8 -*-

import json
import requests
import pprint

class HepData(object):
    '''download data in json format from hepdata.net'''
    def __init__(self, url):
        self.json_data = requests.get(url).json()

    def ls(self, indent=2):
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(self.json_data)

def json_from_file(fname):
    '''read file from local directory'''
    with open(fname, 'r') as json_file:
        data = json.load(json_file)
        return data
