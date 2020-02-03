import json

from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port, timeout=30000)

# The sentence you want to parse
with open('inputfile1.txt', 'r') as file:
    data = file.read().replace('\n', '')

# Sentiment analysis
res = json.loads(nlp.annotate(data,
                              properties={
                                  'annotators': 'sentiment',
                                  'outputFormat': 'json'
                              }))

for s in res["sentences"]:
    print(s['sentimentDistribution'])
    print(s["sentiment"])

# Close Stanford Parser
nlp.close()
