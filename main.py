
import pandas as pd
import re as regex
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
from flask import Flask,request
import jsonify


# using pandas for text preprocessing and cleaning / loading the dataset
df= pd.read_csv('articles - articles.csv')
df.head()

# data cleaning 

def clean_data(text):
    text = regex.sub(r'<,*?','',text)
    text = regex.sub(r'\d+','',text)
    text = text.strip()
    return text

df['cleaned_text'] =  df['Full_Article'].apply(clean_data)
df = df.dropna(subset=['cleaned_text'])
#print(df)

#vector

model = SentenceTransformer('all-MiniLM-l6-v2')
df['vector']=df['cleaned_text'].apply(lambda x:model.encode(x).tolist())
#print(df)

# training data
x=list(df['vector'])
y=df['Article_Type']
#print(y)
#model data training
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(x_train,y_train)
y_pred =clf.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

#save the model in disk
joblib.dump(clf,'text_classifier_model.pkl')
joblib.dump(model,'sentence_transformer_model.pkl')

#prediction

def prediction(text):
    text = text
    cleaned_text = clean_data(text)
    vector = model.encode(cleaned_text).tolist()
    prediction = clf.predict([vector])
    print("--------------------------------")
    print (prediction)
    print("--------------------------------")
    
#function calling with sample article
prediction("The helicopter that crashed in Southeast Alaska in late September, killing three people, entered a 500-foot freefall before dropping to a Glacier Bay National Park beach, according to by the National Transportation Safety Board.&nbsp;The preliminary NTSB report released Friday offers no official probable cause. That determination won&lsquo;t be made until next year at the earliest.")

# prediction API using Flask

app = Flask(__name__)

@app.route('/predict',methods = ['POST'])
def predict():
    data = request.get_json(force = True)
    text = data['text']
    cleaned_text = clean_data(text)
    vector = model.encode(cleaned_text).tolist()
    prediction = clf.predict([vector])
    #it will return a list of predictions so took a first prediction
    return jsonify({'prediction':prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
