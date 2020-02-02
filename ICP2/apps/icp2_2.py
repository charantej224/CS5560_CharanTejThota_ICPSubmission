from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port, timeout=30000)

# The sentence you want to parse
with open('inputfile.txt', 'r') as file:
    data = file.read().replace('\n', '')

# POS
print('POS：', nlp.pos_tag(data))

# Tokenize
print('Tokenize：', nlp.word_tokenize(data))

# NER
#print('NER：', nlp.ner(data))

# Parser
print('Parser：')
print(nlp.parse(data))
print(nlp.dependency_parse(data))

# Sentiment analysis

res = nlp.annotate(data,
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json'
                   })

print(res)

# for s in res["sentences"]:
#     print(s['sentimentDistribution'])
#     print(s["sentiment"])

# Close Stanford Parser
nlp.close()
