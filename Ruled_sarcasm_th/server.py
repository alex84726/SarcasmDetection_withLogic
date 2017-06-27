import os
import signal
import subprocess
import time
from subprocess import PIPE, Popen


class Server(object):
    def __init__(self):
        commands = 'cd stanford-corenlp-full-2016-10-31; \
                  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000'

        self.pro = subprocess.Popen(
            commands, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        print('Wait 1 sec to have server ready')
        time.sleep(1)

    def stop(self):
        if self.pro.poll() is None:  # check process of server is still running
            os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM)


if __name__ == '__main__':
    # tresting codes
    nlp_server = Server()
    from pycorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP('http://localhost:9000')
    res = nlp.annotate(
        "I love you. I hate him. You are nice. He is dumb",
        properties={'annotators': 'sentiment',
                    'outputFormat': 'json'})
    for s in res["sentences"]:
        print("%d: '%s': %s %s" %
              (s["index"], " ".join([t["word"] for t in s["tokens"]]),
               s["sentimentValue"], s["sentiment"]))

    nlp_server.stop()
