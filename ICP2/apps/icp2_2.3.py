import json

from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port, timeout=30000)

# The sentence you want to parse
with open('inputfile.txt', 'r') as file:
    data = file.read().replace('\n', '')

result = json.loads(nlp.annotate(data, properties={'annotators': 'coref', 'pipelineLanguage': 'en'}))

num, mentions = result['corefs'].items()[0]
for mention in mentions:
    print(mention)

# Close Stanford Parser
nlp.close()
