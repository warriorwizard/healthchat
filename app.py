import nltk
# from flask_ngrok import run_with_ngrok
# from pyngrok import ngrok
nltk.download('popular')
import webbrowser
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import os

from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('chat.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
filenumber=int(os.listdir('saved_conversations') [-1] )
filenumber=filenumber+1
file= open('saved_conversations/'+str(filenumber),"w+")
file.write('Hello Mam, I can guide you for your menstural pain.Data Garbage".\n')
file.close()

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# from flask import Flask, render_template, request
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
# run = run_with_ngrok2(app)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/response', methods=['POST'])
def response():
    user_input = request.form['user_input']
    if user_input == 'Yes':
        # move forward to the next question
        return redirect(url_for('next_question'))
    else:
        # exit the chatbot
        return "Thanks for chatting!"
@app.route('/next_question')
def next_question():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if userText == 'exit':
        return "Thanks for chatting!"
    elif userText == 'OK':
        return " Thanks for chatting!"
    elif userText == 'ok':
        return " Thanks for chatting!"
    # return chatbot_response(userText)
    # userText = request.args.get('msg')
    response = chatbot_response(userText)

    appendfile=os.listdir('saved_conversations')[-1]
    appendfile= open('saved_conversations/'+str(filenumber),"a")
    appendfile.write('user : '+userText+'\n')
    appendfile.write('bot : '+response+'\n')
    appendfile.close()
    return chatbot_response(userText)


if __name__ == "__main__":
    # webbrowser.open('http://127.0.0.1:5000/')
    app.run()