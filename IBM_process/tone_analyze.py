import json
import sys

import numpy as np

from watson_developer_cloud import ToneAnalyzerV3

file = sys.argv[1]
outputfile = sys.argv[2]
USERNAME = sys.argv[3]
PASSWORD = sys.argv[4]

tone_analyzer = ToneAnalyzerV3(
    username=USERNAME, password=PASSWORD, version='2017-05-24')

data = np.load(file)
outputdata = []
for datum in data:
    TEXT = datum
    out = json.dumps(tone_analyzer.tone(text=TEXT))
    out = json.loads(out)
    outputdata.append(out)

np.save(outputfile, outputdata)
