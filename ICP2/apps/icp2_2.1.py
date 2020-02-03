from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port, timeout=30000)

# The sentence you want to parse
with open('inputfile1.txt', 'r') as file:
    data = file.read().replace('\n', '')

# NER
print('NER：', nlp.ner(data))

# Close Stanford Parser
nlp.close()
