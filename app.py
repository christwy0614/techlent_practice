import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from nltk.stem import WordNetLemmatizer
import nltk
import re
from sklearn.pipeline import Pipeline
from joblib import load
import time
import spacy
from nltk.corpus import stopwords


def main():
    class TextProcessor:

        lemma = WordNetLemmatizer()
        nlp = spacy.load("en_core_web_sm")
        swds_set = set(stopwords.words('english'))
        swds_set.update(('bofa', 'nsf', 'boa', 'synchrony', 'amerisave', 'america', 'bank', 'chime', 
        'bb', 'mr', 'mrs', 'ms', 'ocwen', 'sls', 'rushmore', 'robinhood', 'llc', 'would', 'could',
        'please', 'will', 'can'))

        def fit(self, X, y=None):
            pass

        def transform(self, X, y=None):
            results = self.clean_text(X)

            #df = pd.DataFrame(results, columns=['complaints_cleaned'])
            #df.to_csv('X_train_cleaned.csv')
            return results

        def fit_transform(self, X, y=None):
            self.fit(X)
            return self.transform(X)

        def get_ners(self, x):
            ners = {}
            doc = TextProcessor.nlp(x)
            
            #only collect name entity from ORG
            # for ent in doc.ents:
            #     if ent.label_ in ('ORG', 'PERSON'):
            #         ners[ent.text] = ners.get(ent.text, 0) + 1
            # return ners

            for ent in doc.ents:
                ners[ent.text] = ners.get(ent.text, 0) + 1
            return ners

        def clean_text(self, x):
            #remove punctutation, use space for replacement
            x = re.sub(r'[^\w\s]',' ',x)
            #remove number, use space for replacement
            x = re.sub('[0-9]*[+-:]*[0-9]+', ' ', x)
            #remove Xx, Yy, Zz at least appear twice, use space for replacement
            x = re.sub('[Xx|Yy|Zz]{2,}', ' ', x)
            #remove more than one space between words
            x = ' '.join(x.split())
            
            ners = self.get_ners(x)
            for ner in ners:
                x = re.sub(ner, ' ', x) 
                
            #remove stop words
            x = " ".join(
                [
                    word for word in x.lower().split()
                    if word not in TextProcessor.swds_set
                ]
            )

            # tokenize
            x = nltk.word_tokenize(x)
            # lemmatization
            x = [TextProcessor.lemma.lemmatize(word, "v") for word in x]
            x = " ".join(x)
            return x

    st.title('Bank Complaint Loss Detector')
    tabs = st.tabs(['Batch', 'Individual'])

    tp = TextProcessor()
    model = load('els_gridcv_best_remove_all.joblib')
    output = pd.DataFrame()

    batch = tabs[0]
    with batch:
        uploaded_file = st.file_uploader('Upload your csv file here')
        if uploaded_file:
                df = pd.read_csv(uploaded_file, header = 0, names = ['id', 'complaint'])
                st.write(df.head())

        batch_pred = st.button('Predict', key = 'batch')
        if batch_pred:
            if isinstance(df, pd.DataFrame):
                with st.spinner(text="In progress..."):
                    time.sleep(5)
                    
                    text = df.complaint.map(tp.clean_text)

                    y_pred = pd.DataFrame(model.predict(text), columns = ['Loss'])
                    y_pred_prob = pd.DataFrame(model.predict_proba(text)[:, 1], columns = ['Probability of Loss'])
                    # output = pd.concat([df, y_pred_prob, y_pred], axis = 1)
                    output = pd.concat([df, y_pred], axis = 1)
                    # st.success('Done.')
            else:
                st.write('Please upload data first.')
        
        if not output.empty:
            st.write('Results:')
            AgGrid(output)
            st.download_button('Download results', data = output.to_csv(), file_name= 'results.csv', mime = 'text/csv')

    individual = tabs[1]
    with individual:
        complaint = st.text_area('Please enter your complaint here:', '')

        ind_predict = st.button('Submit', key = 'ind')

        if complaint != '' and ind_predict:
            df = pd.DataFrame({'complaint': [complaint]})
            text = df.complaint.map(tp.clean_text)

            y_pred = pd.DataFrame(model.predict(text), columns = ['Loss'])
            # y_pred_prob = pd.DataFrame(model.predict_proba(text)[:, 1], columns = ['Probability of Loss'])
            
            # output = pd.concat([df, y_pred_prob, y_pred], axis = 1)
            output = pd.concat([df, y_pred], axis = 1)

            st.write(output)
            st.download_button('Download results', data = output.to_csv(), file_name= 'results.csv', mime = 'text/csv')

if __name__ == '__main__':
    main()