# gender-bias-removal
Removing gender bias present in Bollywood movie scripts.

# Documentation
The ideation document <br/>
The presentation <br/>
The submission video - Can be found at https://youtu.be/pO6qz3UhWlc <br/>
The Key Value Proposition document <br/>


# Files in the repository
compiled.py - The modular python script that combines removal of dependent introductions and replacement of gender biased occupations and adjectives for removal of bias in plots <br/>
Dictionary.py - Holds the mappings between commonly used gender biased adjectives and their suitable neutral replacement adjectives <br/>
occupations.txt - The list of all possible occupations characters could have in the movie plot <br/>
fOccReplace - female occupations replaced(dictionary) <br/>
fOcc - female stereotypical jobs <br/>
mOcc - male stereotypical jobs <br/>
coref_plot_revised.csv - A simple modification of the original coref_plot.csv given that makes processing of plots comprehensive <br/>
FrequentMAdj.txt and FrequentFAdj.txt - The list of frequently used adjectives for male and female characters typically, indicative of bias <br/>

# How to run the code
Install dependencies - python 3, spacy, nltk, genderize, gensim, wikipedia - and run compiled.py <br/>
