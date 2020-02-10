'''
WordNet Task:Perform following lexical relations:
1.Hyponym (a more specific concept)
2.Hypernym (a more general concept)
3.Meronym (denotes a part of something)
4.Holonym (denotes a membership to something)
5.Entailment (denotes how verbs are involved)
'''


from nltk.corpus import wordnet as wn

dog = wn.synset('human.n.01')
print("1.Hyponym (a more specific concept)")
print(dog.hyponyms())
print("2.Hypernym (a more general concept)")
print(dog.hypernyms())
print(dog.root_hypernyms())
print("3.Meronym (denotes a part of something)")
print(dog.member_holonyms())
print("4.Holonym (denotes a membership to something)")
print(dog.part_meronyms())
#print(dog.substance_meronyms())
print("5.Entailment (denotes how verbs are involved)")
print(wn.synset('eat.v.01').entailments())
