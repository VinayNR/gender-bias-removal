import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from genderize import Genderize
import spacy
import en_core_web_sm
import pandas as pd
import wikipedia

stop = stopwords.words('english')
nlp = en_core_web_sm.load()

data = pd.read_csv(r"C:\Users\rramess\Desktop\Projects\IBM Movie\coref_plot_revised_v2.csv")
data['Coref Plot'] = data['Coref Plot'].astype(str)
data['Movie Name'] = data['Movie Name'].astype(str)

document = data.iloc[1413]["Coref Plot"]
doc = nlp(document)

noun_verb_pairs = {}

class Movie:
    def __init__(self, name, plot):
        self.name = name
        self.plot = plot
        
    def getName(self):
        return self.name

    def getPlot(self):
        return self.plot

class Character:
    def __init__(self, name, gender, verb=[], adj=[], adverb=[], occupation="occupation"):
        self.name = name
        self.gender = str(gender)
        self.verb = verb
        self.adj = adj
        self.adverb = adverb
        self.occupation = occupation
        
    def getName(self):
        return self.name

    def getGender(self):
        return self.gender

# Get gender of a given first name using genderize API - Takes the most time
def getGender(name):
    print(name)
    document = wikipedia.WikipediaPage('Kaho Naa Pyaar Hai').section('Cast')
    count = 0
    check_count = 0
    flag = 0
    actor = ''

    dest = 'as ' + name
    for line in document.splitlines():
        if dest in line:
            for char in line:
                if char == ' ':
                    count = count + 1
                    if count == 2:
                        flag = 1
                        break
                actor = actor + char
        if flag == 1:
            break
        check_count = check_count + 1



        if check_count == 4:
            return "male"
    
    print(actor)
    document = wikipedia.WikipediaPage(actor)
    # print(document.content)

    for line in document.content.splitlines():
        print(line)
        if "actor" in line:
            return "male"
        elif "actress":
            return "female"



def printCharacters(characters):
    for character in characters:
        print(character.getName(), character.getGender())

# Get list of all occupations from a text file
def getOccupations():
    occupations = []
    with open('occupations.txt') as f:
        occupations = f.read().split('\n')
    occupations = [word.lower() for word in occupations]
    return occupations

# Get list of characters which are objects of Class Character from a sample Plot of a movie
def getCharacters(samplePlot):
    sentences = nltk.sent_tokenize(samplePlot)
    tokens = []
    characters = []
    for sentence in sentences:
        tokens.append(nltk.word_tokenize(sentence))

    tokensWithPOS = [nltk.pos_tag(token) for token in tokens]

    names = set()
    for token in tokensWithPOS:
        for i in range(len(token) - 1):
            if token[i][1] == 'NNP' and token[i+1][1] == 'NNP':
                names.add(token[i][0] + '_' + token[i+1][0])
                i = i+1
            elif token[i][1] == 'NNP':
                names.add(token[i][0])
            elif token[i+1][1] == 'NNP':
                names.add(token[i+1][0])

    for name in names:
        gender = getGender(name.split('_')[0])
        if gender is not None:
            characters.append(Character(name, gender, [], [], [], 'occupation'))
    return characters

# Get character name given position of the occupation(role) in the Plot and positions of all characters - Determines using least distance
def getCharacterFromRole(pos, characterPos):
    closest = 999
    for (character, i) in list(characterPos):
        if i:    #List not empty
            val = abs(min(i, key=lambda x:abs(x - pos)) - pos)
            if val < closest:
                closest = val
                name = character
    return name

def main():
    sampleMovie = 'Sample Movie'
    samplePlot = "Rohit and Rohit younger brother Amit are orphans living with an elderly couple  Lily and Anthony. Rohit is an aspiring doctor who works as a salesman in a car showroom  run by Malik. One day Rohit meets Sonia Saxena  daughter of Mr Saxena  when goes to deliver a car to home as birthday present"
    
    movie = Movie(sampleMovie, samplePlot)

    tokenizer = RegexpTokenizer(r'\w+')
    movieTokens = tokenizer.tokenize(samplePlot)

    # all occupations
    occupations = getOccupations()
    
    # occupations mapped to the names
    roles = []
    
    characters = getCharacters(movie.getPlot())
    
    names = []
    for c in characters:
        names.append(c.name)

    characterPos = []
    for character in characters:
        characterPos.append((character.getName(), [pos for pos,word in enumerate(movieTokens) if word == character.getName()]))
    #print(characterPos)

    for pos, word in enumerate(movieTokens):
        for role in occupations:
            if word.lower().decode('utf-8') == role.lower()decode('utf-8'):
                roles.append((role, getCharacterFromRole(pos, characterPos)))
                for c in characters:
                    if c.name == getCharacterFromRole(pos, characterPos):
                        c.occupation = role
    print(roles)

    for c in characters:
        print(c.name, c.gender, c.occupation)


    # Noun-Verb Pairs

    document = samplePlot
    doc = nlp(document)
    for token in doc:
        if token.pos_ == 'PROPN' and token.head.pos_ == 'VERB' and token.head.text not in stop:
            if token.text in noun_verb_pairs.keys():
                noun_verb_pairs[token.text] = noun_verb_pairs[token.text] + ',' + token.head.text
            else:
                noun_verb_pairs[token.text] = token.head.text

            for c in characters:
                if token.text == c.name:
                    c.verb.append(token.head.text)
                    

    print("NOUN VERB PAIRS\n",noun_verb_pairs)

    for c in characters:
        print (c.verb)

    # Noun-Adjective Pairs
    noun_adj_pairs = []
    for i,token in enumerate(doc):
        if token.pos_ not in ('PROPN','NOUN'):
            continue
        for j in range(i+1,len(doc)):
            if doc[j].pos_ == 'ADJ':
                noun_adj_pairs.append((token,doc[j]))
                for c in characters:
                    if str(token) == c.name:
                        c.adj.append(doc[j])
                i=j
                break
            elif doc[j].pos_ == 'PROPN':
                break
            elif doc[j-1].pos_ == 'ADJ':
                noun_adj_pairs.append((token,doc[j-1]))

    print("\nNOUN ADJECTIVE PAIRS\n",noun_adj_pairs)

    # Noun-Adverb Pairs

    noun_adverb_pairs = {}
    for token in doc:
        Cast = None
        Adverb = None
        if token.pos_ == 'VERB':
            for child in token.children:
                if child.pos_ == 'PROPN':
                    Cast = child.text
                elif child.pos_ == 'ADV':
                    Adverb = child.text
            if Cast and Adverb and Adverb not in stop:
                if Cast in noun_adverb_pairs.keys():
                    noun_adverb_pairs[Cast] = noun_adverb_pairs[Cast] + ',' + Adverb
                else:
                    noun_adverb_pairs[Cast] = Adverb

                for c in characters:
                    if Cast == c.name:
                        c.adverb.append(Adverb)
                
    print("\nNOUN ADV PAIRS\n",noun_adverb_pairs)

if __name__ == "__main__":
    main()




