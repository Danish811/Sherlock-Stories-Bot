import numpy as np
import pandas as pd
import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import pickle
import streamlit as st
#nltk.download('punkt')
story_path = "Sherlock-holmes-stories-data\sherlock\sherlock"

st.title("Sherlock Story Generator")
st.header("Create a Story!")
two_words = st.text_input(label="Give two words to start: ")
start = st.button("Start")


def read_all_stories(story_path):
  txt = []
  for _, _, files in os.walk(story_path):
    for file in files:
      story_path += "/"
      with open(story_path + file) as f:
        for line in f:
          line = line.strip()
          if line == '----------': break
          if line !='' :txt.append(line)
  stories = read_all_stories(story_path)
  print("Number of line = ", len(stories))

  return txt


def clean_txt(txt):
  cleaned_txt = []
  for line in txt:
    line = line.lower()
    line = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-\\]", "", line)
    tokens = word_tokenize(line)
    words = [word for word in tokens if word.isalpha()]
    cleaned_txt += words
  cleaned_stories = clean_txt(stories)
  print("Number of words = ", len(cleaned_stories))
  return cleaned_txt



def make_markov_model(cleaned_stories, n_gram=2):
    markov_model = {}
    for i in range(len(cleaned_stories)-n_gram-1):
        curr_state, next_state = "", ""
        for j in range(n_gram):
            curr_state += cleaned_stories[i+j] + " "
            next_state += cleaned_stories[i+j+n_gram] + " "
        curr_state = curr_state[:-1]
        next_state = next_state[:-1]
        if curr_state not in markov_model:
            markov_model[curr_state] = {}
            markov_model[curr_state][next_state] = 1
        else:
            if next_state in markov_model[curr_state]:
                markov_model[curr_state][next_state] += 1
            else:
                markov_model[curr_state][next_state] = 1
    
    # calculating transition probabilities
    for curr_state, transition in markov_model.items():
        total = sum(transition.values())
        for state, count in transition.items():
            markov_model[curr_state][state] = count/total
        
    return markov_model
  
#markov_model = make_markov_model(cleaned_stories)

filename = 'sherlock_model.sav'
#pickle.dump(markov_model, open(filename, 'wb'))

#print("No. of states = ", len(markov_model.keys()))

#print("All possible transitions from 'the game' state is: \n")
#print(markov_model['the game'])

@st.cache_data
def load_model():
    with open('sherlock_model.sav', 'rb') as file:
        model = pickle.load(file)
    return model

# This now loads the model ONCE and then uses a fast, cached version
markov_model = load_model()

def generate_story(limit,start):
   n = 1
   curr_state = start
   next_state = None
   story = ""
   story += curr_state+" "
   while n < limit:
      if curr_state in markov_model or curr_state not in markov_model:
        print(f"Error: {curr_state} not in markov_model")
        if curr_state not in markov_model:
          curr_state = random.choice(list(markov_model['this case'].keys()))
      next_state = random.choices(list(markov_model[curr_state].keys()),list(markov_model[curr_state].values()))
      curr_state = next_state[0]
      if n%7==0:
         story += '\n'
      story += curr_state + " "
      n += 1

   return story

st.divider()

# st.markdown(
#     """
#     <style>
#     .element-container pre {
#         font-size: 20px !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

#print(new_model)
# --- Streamlit UI ---

st.title("Sherlock Story Generator 🕵️")
st.header("Create a Story!")

# Load the model using the cached function
markov_model = load_model()

two_words = st.text_input(label="Give two words to start the story:")

if st.button("Generate Story"):
    if len(two_words.split()) == 2:
        with st.spinner("Writing a new mystery..."):
            new_story = generate_story(markov_model, 100, two_words)
            st.divider()
            st.subheader("Your Story:")
            st.markdown(f"> {new_story}")
    else:
        st.error("Please enter exactly two words.")
