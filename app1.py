import nltk
nltk.download('popular')
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import os

from keras.models import load_model
model = load_model('D:/CHAT/Healthchat/model.h5')
import json
import random 
intents = json.loads(open('D:/CHAT/Healthchat/chat.json').read())
words = pickle.load(open('D:/CHAT/Healthchat/texts.pkl','rb'))
classes = pickle.load(open('D:/CHAT/Healthchat/labels.pkl','rb'))
# filenumber=int(os.listdir('saved_conversations')[0])
# filenumber=filenumber+1
# file= open('saved_conversations/'+str(filenumber),"w+")
# file.write('Hello Mam, I can guide you for your menstural pain.Data Garbage".\n')
# file.close()

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

# def predict_class(sentence, model):
#     # filter out predictions below a threshold
#     p = bow(sentence, words,show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.35
#     results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.35
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list


# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if(i['tag']== tag):
#             result = random.choice(i['responses'])
#             break
#     return result
import random

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    intent_dict = {i['tag']: i for i in intents_json['intents']}
    intent = intent_dict.get(tag, None)
    if intent:
        return random.choice(intent['responses'])
    else:
        return "Sorry, I didn't understand that."

# without random
# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return i['responses'][0]
#     return "I'm sorry, I didn't understand what you said."
# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if(i['tag']== tag):
#             responses = i['responses']
#             break
#     return random.choice(responses)
def get_bot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    # with open('saved_conversations/conversation_history.txt', 'a') as f:
    #     f.write('User: {}\n'.format(msg))
    #     f.write('ChatBot: {}\n'.format(res))
    return res

def chatbot_response(msg):

    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    with open('D:/CHAT/Healthchat/saved_conversations/conversation_history.txt', 'a') as f:
        f.write('User: {}\n'.format(msg))
        f.write('ChatBot: {}\n'.format(res))
    return res


from flask import Flask, render_template, request,redirect,url_for

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    login_id = request.form['login_id']
    password = request.form['password']
    if login_id == '789564' and password == '3636':
        return redirect('/home')
    else:
        return render_template('login.html', error='Invalid login credentials')
@app.route('/home')
def home():
    return render_template("index.html")
# @app.route('/ContactUs')
# def ContactUs():
#     return render_template('ContactUs.html')

@app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     return chatbot_response(userText)
def get_bot_response():
    userText = request.args.get('msg')
    # if userText=='9010':
        
        # return chatbot_response(userText)
        # userText = request.args.get('msg')
    response = chatbot_response(userText)

        # appendfile=os.listdir('saved_conversations')
        # # appendfile=os.listdir('saved_conversations')[-1]
        # appendfile= open('saved_conversations/'+str(filenumber),"a")
        # appendfile.write('user : '+userText+'\n')
        # appendfile.write(str('bot : ')+response+'\n')
        # appendfile.close()
    return chatbot_response(userText)



if __name__ == "__main__":
    app.run()