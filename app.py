#all the imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import streamlit as st
from PIL import Image 

#loading the dataset
df_candidats = pd.read_excel('C:\\wamp64\\www\\Streamlit\\CANDIDATS.xlsx')

#cleaning and pre-processing the column of features
#downloadig the stopwords, punkt and wordnet using the NLTK downloader
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

def black_txt(token):
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2   
  
def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "",text)
    text=re.sub("(\\d|\\W)+"," ",text) 
    text = text.replace("nbsp", "")
    clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)

#columns featurs
df_candidats['features_cdt'] = df_candidats['Profil']+" "+df_candidats['Secteur_act']+" "+df_candidats['Diplome']+" "+df_candidats['Experience'].map(str)+" "+df_candidats['Professional_skills']

candidats = df_candidats[['id_candidat','Prenom', 'Nom', 'Profil', 'features_cdt']]
candidats.loc[:, 'features_cdt'] = candidats['features_cdt'].apply(clean_txt)

cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(candidats['features_cdt']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

#similarity

def recommend(cdt):
    index = candidats[candidats['Profil'] == cdt].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    recommendations = []
    for i in distances[1:4]:
        candidate_id = candidats.iloc[i[0]].id_candidat
        candidate_name = candidats.iloc[i[0]]['Nom']
        candidate_surname = candidats.iloc[i[0]]['Prenom']
        candidate_position = candidats.iloc[i[0]]['Profil']
        recommendations.append((candidate_id, candidate_name, candidate_surname, candidate_position))
    return recommendations


# Create the Streamlit app
def app():
    st.title("Job Candidate Recommender")

    # Get user input
    job_position = st.text_input("Enter a job position")

    # Show recommendations
    if job_position:
        recommendations = recommend(job_position)
        if recommendations:
            st.write("Here are the top candidates for the position of", job_position)
            for recommendation in recommendations:
                st.write("- Name:", recommendation[1], recommendation[2])
                st.write("  Position:", recommendation[3])
        else:
            st.error("No recommendations found for the position")

img = Image.open('hire.jpg')
st.image(img)




if __name__ == '__main__':
    app()
