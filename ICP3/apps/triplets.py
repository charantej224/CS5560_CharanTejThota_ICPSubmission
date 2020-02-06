import spacy
import textacy

text = "Brack Obama was born in Hawaii"

nlp = spacy.load('en_core_web_sm')
for sentence in text.split("."):
    val = nlp(sentence)
    tuples = textacy.extract.subject_verb_object_triples(val)
    tuples_list = []
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)
        print(tuples_list)
