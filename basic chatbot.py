import random
import string 
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 

with open('german.txt', 'r', encoding='utf8', errors='ignore') as german_file, \
     open('chatbotData.txt', 'r', encoding='utf8', errors='ignore') as chatbot_file:
    german_content = german_file.read()
    chatbot_content = chatbot_file.read()
    
raw_text = (german_content + chatbot_content).lower()

sent_tokens = nltk.sent_tokenize(raw_text)# converts to list of sentences (nltk stands for Natural Language ToolKit)
word_tokens = nltk.word_tokenize(raw_text)# converts to list of words and characters 
#ex: raw_text = "hello, world"
    #sent = ["Hello, world"]
    #word = ["Hello", ",", "world"]

# Preprocessing
lemmer = WordNetLemmatizer() # Lemmatization is the process of reducing words to their base or root form, called a "lemma." ex: ["runs"] becomes ["run"]
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens] #lemmatizes the words to amke it faster and easier for the model to search
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation) #removes puncuation to make it easier
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

'''A token is the smallest individual unit in a python program. All statements and instructions in a program are built with tokens.'''

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey", "bonjour", "hallo")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    Luci_response=''
    sent_tokens.append(user_response) #add the sentane input to the sentance tokens so it cna be used fr comparison
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens) # TF-IDF stands for Term Frequency-Inverse Document Frequency, where it will take all teh sent tokens and make it into a matrix and gives a value of how relevent each word is to teh inputted sentance
    vals = cosine_similarity(tfidf[-1], tfidf) #computes the cosine similarity between the last sentence (inputted sentance) and all other sentences in the corpus. The vals variable is assigned these similarity scores, where each value represents how similar the user's query (the inputted sentance) is to each other sentence in the corpus.
    idx=vals.argsort()[0][-2] #take the index of the sentance to return, the sentance is the second best since you added the user responce
    flat = vals.flatten() # flatten teh return value
    flat.sort()
    req_tfidf = flat[-2] #the return sentance since it is the second most recognized
    if(req_tfidf==0):
        Luci_response=Luci_response+"what are you yapping about?" #if not found then return this
        return Luci_response
    else:
        Luci_response = Luci_response+sent_tokens[idx]  # add teh "Luci:" to the setneance returned
        return Luci_response


flag=True
print("Luci: My name is Luci. I can answer questions about the German Language and the history of Chatbots, Type by to leave,type Bye!")
while(flag==True):
    user_response = input("You: ")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Luci: You are welcome...")
        else:
            if(greeting(user_response)!=None):
                print("Luci: " + greeting(user_response))
            else:
                print("Luci: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Luci: Bye! take care..")    
        
        