# gender-bias-removal
Removing gender bias present in Bollywood movie scripts. 

# Documentation
The ideation document
The presentation
The submission video
The Key Value Proposition document


# Files in the repository
compiled.py - The modular python script that combines removal of dependent introductions and replacement of gender biased occupations and adjectives for removal of bias in plots
Dictionary.py - Holds the mappings between commonly used gender biased adjectives and their suitable neutral replacement adjectives
occupations.txt - The list of all possible occupations characters could have in the movie plot
fOccReplace - female occupations replaced(dictionary)
fOcc - female stereotypical jobs
mOcc - male stereotypical jobs
coref_plot_revised.csv - A simple modification of the original coref_plot.csv given that makes processing of plots comprehensive
FrequentMAdj.txt and FrequentFAdj.txt - The list of frequently used adjectives for male and female characters typically, indicative of bias

# How to run the code
Install dependencies - python 3, spacy, nltk, genderize, gensim, wikipedia - and run compiled.py
