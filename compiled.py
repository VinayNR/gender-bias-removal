import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
import spacy
import en_core_web_sm
import pandas as pd
import wikipedia

from gensim.models import KeyedVectors

stop = stopwords.words('english')
nlp = en_core_web_sm.load()

data = pd.read_csv("coref_plot_revised_v2.csv")
data['Coref Plot'] = data['Coref Plot'].astype(str)
data['Movie Name'] = data['Movie Name'].astype(str)

document = data.iloc[1413]["Coref Plot"]
doc = nlp(document.decode('utf-8'))

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
    def __init__(self, name, gender, verb=[], adj=[], adverb=[], occupation=[]):
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
    
    def getOccupations(self):
        return self.occupation
    
    def getVerbs(self):
        return self.verb
    
    def getAdjectives(self):
        return self.adj
    
    def getAdverbs(self):
        return self.adverb

    def setOccupation(self, value):
        self.occupation.append(value)

# Get gender of a given first name using genderize API - Takes the most time
def getGender(name, movieName):
    document = wikipedia.WikipediaPage(movieName).section('Cast')
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
    
    document = wikipedia.WikipediaPage(actor)
    #print(document.content)

    for line in document.content.splitlines():
        #print(line)
        if "actor" in line:
            return "male"
        elif "actress":
            return "female"

def printCharacters(characters):
    for character in characters:
        print(character.getName(), character.getGender(), character.getOccupations(), character.getVerbs(), character.getAdjectives(), character.getAdverbs(), "\n")

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
            characters.append(Character(name, gender, [], [], [], []))
    return characters

# Get character name given position of the occupation(role) in the Plot and positions of all characters - Determines using least distance
def getCharacterFromRoleNearestPosition(pos, characterPos):
    closest = 999
    for (character, i) in list(characterPos):
        if i:    #List not empty
            val = abs(min(i, key=lambda x:abs(x - pos)) - pos)
            if val < closest:
                closest = val
                name = character
    return name.lower()

def getCharacterFinalMethod(characters, movieTokens, role):
    characterPos = []
    # occupations mapped to the names
    for character in characters:
        characterPos.append((character.getName(), [pos for pos,word in enumerate(movieTokens) if word == character.getName()]))
    #print(characterPos)
    
    for pos, word in enumerate(movieTokens):
        #print((pos,word))
        if word.lower().decode('utf-8') == role.lower().decode('utf-8'):
            return getCharacterFromRoleNearestPosition(pos, characterPos)

def deBiasOccupation(movieTokens, characters, occupations):
    # Gensim Model
    wordEmbFile = 'glove.6B.100d.txt.word2vec'
    model = KeyedVectors.load_word2vec_format(wordEmbFile, binary=False)

    # Remove Male Bias
    mOcc = []
    for character in characters:
        if character.gender == 'male' and character.occupation:
            for item in character.occupation:
                mOcc.append(item)

    for occ in mOcc:
        for token in movieTokens:
            if token.lower() not in stop and occ != token and model.similarity(occ, token.lower()) > 0.5:
                mOcc.remove(occ)
                break

    # Remove Female Bias
    fOcc = []
    for character in characters:
        if (character.gender == 'female') and (character.occupation):
            for item in character.occupation:
                fOcc.append(item)

    return mOcc, fOcc

def removeIntroductions(movie): #to remove introductions in the given movie plot
	characters = getCharacters(movie.getPlot())
	for c in characters:
		firstOccurrenceFound = 0
		movie.plot = introduction_removed_plot(c,movie,firstOccurrenceFound)
	return movie.getPlot()

def introduction_removed_plot(c,movie,firstOccurrenceFound): #repeatedly removes introductions for each character
	newPlot = []
	sentences = nltk.sent_tokenize(movie.getPlot())
	for sent in sentences:
		if sent.find(c.name.capitalize()) > -1 and not firstOccurrenceFound:
			firstOccurrenceFound = 1
			doc = nlp(sent)
			words = []
			for chunk in doc.noun_chunks:
				if chunk.root.dep_ == 'appos':
					words.append(chunk.root.head.text)
					words.append(chunk.text)
				if chunk.root.dep_ == 'attr':
					for children in chunk.root.head.children:
						words.append(children.text)
						words.append(chunk.text)
						break
				elif chunk.root.dep_ == 'pobj':
					words.append(chunk.root.head.text)
					words.append(chunk.text)
			if words:
				last = words[-1]
				firstIndex = sent.find(c.name.capitalize()) + len(c.name)
				lastIndex = sent.find(last) + len(last)
				firstPart = sent[:firstIndex]
				secondPart = sent[lastIndex:]
				sent = firstPart+secondPart
		newPlot.append(sent)
	newPlot = (' ').join(newPlot)
	return newPlot

def main(name, text):
    sampleMovie = name
    samplePlot = text
    
    movie = Movie(sampleMovie, samplePlot)

    tokenizer = RegexpTokenizer(r'\w+')
    movieTokens = tokenizer.tokenize(samplePlot)

    # all occupations
    occupations = getOccupations()
    
    characters = getCharacters(movie.getPlot())

    names = []
    for c in characters:
        names.append(c.name)
    
    # occupation-character pairs
    roles = []
    for sent in samplePlot.split('.'):
        doc = nlp(sent)
        for token in doc:
            for occupation in occupations:
                if token.text.lower().decode('utf-8') == occupation.lower().decode('utf-8'):
                    if token.dep_ == 'attr':
                        head = token.head
                        for child in head.children:
                            if child.dep_ == 'nsubj':
                                roles.append((child.text.lower(), token.text.lower()))
                    elif (token.dep_ == 'dobj' or token.dep_ == 'amod' or token.dep_ == 'appos'):
                        roles.append((token.head.text.lower(), token.text.lower()))
                    else:
                        roles.append((getCharacterFinalMethod(characters, movieTokens, occupation), token.text.lower()))

    # Add roles of characters to the list of objects
    for role in roles:
        for c in characters:
            if role[0] == c.name.lower():
                c.setOccupation(role[1])

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

    # Debias Occupations - Get occupations to be replaced
    mOcc, fOcc = deBiasOccupation(movieTokens, characters, occupations)

    for occupation in mOcc:
        movie.plot = movie.plot.replace(occupation, "man")

    with open('fOccReplace.txt', 'r') as f:
        fOccupation = f.read().split()
    fOccupation = [word.lower() for word in fOccupation]

    for occupation in fOcc:
        if occupation in fOccupation and fOccupation.index(occupation)%2 == 0:
            movie.plot = movie.plot.replace(occupation, fOccupation[fOccupation.index(occupation) + 1])

    print(movie.plot)

    return movie.plot

if __name__ == "__main__":
    main()


