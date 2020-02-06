import nltk
from nltk import word_tokenize

predicate_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'IN']
subject_tags = ['NN', 'NNS', 'NNP', 'NNPS']
object_tags = ['NN', 'NNS', 'NNP', 'NNPS']

input_text = "Brack Obama was born in Hawaii"

text = word_tokenize(input_text)
pos_tagging = nltk.pos_tag(text)

print(pos_tagging)

result_dict = {}
subject = ''
object = ''
predicate = ''

for pos_tag in pos_tagging:
    if pos_tag[1] in subject_tags and 'subject' not in result_dict.keys():
        subject = subject + pos_tag[0] + " "
    elif pos_tag[1] in predicate_tags and 'predicate' not in result_dict.keys():
        predicate = predicate + pos_tag[0] + " "
        if 'subject' not in result_dict.keys():
            result_dict['subject'] = subject
    elif pos_tag[1] in object_tags and 'object' not in result_dict.keys():
        object = object + pos_tag[0] + " "
        if 'predicate' not in result_dict.keys():
            result_dict['predicate'] = predicate
    else:
        break
result_dict['object'] = object

print(result_dict)
