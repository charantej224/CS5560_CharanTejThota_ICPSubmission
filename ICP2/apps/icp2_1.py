import spacy
import neuralcoref

'''
a.The dog saw John in the park
b.The little bear saw the fine fat trout in the rocky brook
'''

nlp = spacy.load("en_core_web_sm")
doc1 = nlp("The dog saw John in the park")
doc2 = nlp("The little bear saw the fine fat trout in the rocky brook")

print("position tagging for Document1 ")
print("*********************************")
for token in doc1:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
print("position tagging for Document2 ")
print("*********************************")
for token in doc2:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

print("Named Entities for Document 1")
print("*********************************")
for ent in doc1.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

print("Named Entities for Document 2")
print("*********************************")
for ent in doc2.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

neuralcoref.add_to_pipe(nlp)
print("Co-relation for Document 1")
print("*********************************")
print(doc1._.coref_clusters)
print("Co-relation for Document 2")
print("*********************************")
print(doc2._.coref_clusters)
