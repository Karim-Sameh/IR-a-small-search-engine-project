# nltk.download()
from re import T
from typing import AnyStr, Counter
import nltk 
from collections import Counter
from nltk.corpus import stopwords
import math
from nltk.sem.logic import printtype
import pandas as pd

tokenizedWords = []
files = ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt",
         "6.txt", "7.txt", "8.txt", "9.txt", "10.txt"]

for i in files:
    f = open(i, "r")
    tokenizedWords += nltk.word_tokenize(f.read())

stopWord = set(stopwords.words('english')) - set(['in', 'to', 'where'])

tokenizedWords = [
    search for search in tokenizedWords if search not in stopWord]

print(tokenizedWords)


positional_index = {}
for word in tokenizedWords:
    for file in files:
        f = open(file, "r")
        file_content = f.read()
        file_content_list = file_content.split()
        file_content_list = [word.lower() for word in file_content_list]
        file_content_list = [
            search for search in file_content_list if search not in stopWord]
        for index, file_word in enumerate(file_content_list):
            if file_word[-1] == '.' or file_word[-1] == ',':
                file_word = file_word[:-1]
            index += 1
            if word == file_word:
                if len(positional_index) == 0:
                    positional_index[word] = {file: [index]}
                else:
                    if word in positional_index:
                        if file in positional_index[word]:
                            if index in positional_index[word][file]:
                                break
                            else:
                                positional_index[word][file].append(index)
                        else:
                            positional_index[word][file] = [index]
                    else:
                        positional_index[word] = {file: [index]}

for word in positional_index:
    for file in positional_index[word]:
        length = 0
        length += len(positional_index[word])
    print("< " + word + "," + str(length)+";")
    for file in positional_index[word]:
        indexes = ', '.join(map(str, positional_index[word][file]))
        print(file + ": " + indexes + ";")
    print(">")
    print()


query = input("Enter Your Query: ")
query_tokenization = nltk.word_tokenize(query)
query_tokenization = [word.lower() for word in query_tokenization]
sw = set(stopwords.words('english'))
query_tokenization = [w for w in query_tokenization if not w.lower() in sw]
query_tokenization = [
    search for search in query_tokenization if search not in stopWord]
fileArray = []
printCounter = 0
for i in range(0, len(query_tokenization)):
    if query_tokenization[i] in positional_index:
        printCounter += 1
        for fileName in positional_index[query_tokenization[i]]:
            if fileName in positional_index[query_tokenization[i]]:
                fileArray.append(fileName)
if printCounter < 1:
    print("NOT FOUND !")
else:
    print("FOUND IN :")
    for x in range(0, len(fileArray)):
        counter = 0
        for y in range(x, len(fileArray)):
            if fileArray[x] == fileArray[y]:
                counter += 1
        if counter == len(query_tokenization):
            print(fileArray[x])
print("\n")


vector_space_model = {}
a1 = {}
Idf = {}
tfweight = {}
tfidf = {}
for file in files:
    vector_space_model[file] = {}
    a1[file] = {}
    Idf[file] = {}
    tfweight[file] = {}
    tfidf[file] = {}
    for term in positional_index:
        if file in positional_index[term]:
            dft = len(positional_index[term])
            idf = round((math.log10(len(files) / dft)), 5)
            tf = len(positional_index[term][file])
            tf_weight = round((1 + math.log(tf)), 5)
            tf_idf = round((tf_weight * idf), 5)
            vector_space_model[file][term] = [tf, tf_weight, idf, tf_idf]
            a1[file][term] = [tf]
            tfweight[file][term] = [tf_weight]
            Idf[file][term] = [idf]
            tfidf[file][term] = [tf_idf]
        else:
            vector_space_model[file][term] = [0, 0, 0, 0]
            a1[file][term] = [0]
            tfweight[file][term] = [0]
            Idf[file][term] = [0]
            tfidf[file][term] = [0]

print("____________________________tf_______________________________________") # term frequency 
print(pd.DataFrame(a1))
print("____________________________tfWeight_______________________________________") # W = 1 + log(tf)
print(pd.DataFrame(tfweight))
print("____________________________IDF_______________________________________") # idf = log(N / df) .. N = number of files
print(pd.DataFrame(Idf))
print("____________________________TF-IDF_______________________________________") # tf-idf = (1 + log(tf)) x log(N / df)
print(pd.DataFrame(tfidf))
print("_____________________")


for file in vector_space_model:
    tf_idf_square_summation = 0
    for term in vector_space_model[file]:
        tf_idf_square_summation += vector_space_model[file][term][3] * \
            vector_space_model[file][term][3]
    length_of_document = math.sqrt(tf_idf_square_summation)
    print("the length of document")
    print(file, "=", length_of_document)
    for term in vector_space_model[file]:
        if sum(vector_space_model[file][term]) == 0:
            vector_space_model[file][term].append(0)
        else:
            vector_space_model[file][term].append(
                vector_space_model[file][term][3]/length_of_document)
print("\n")
query_vector_space_model = {}
for word in query_tokenization:
    idf = math.log(len(files) / len(positional_index[term]))
    tf = 0
    for qeury_word in query_tokenization:
        tf += 1
    tf_weight = 1 + math.log(tf)
    tf_idf = tf_weight * idf
    query_vector_space_model[word] = [idf, tf, tf_weight, tf_idf]

for word in query_vector_space_model:
    tf_idf_square_summation = 0
    for qeury_word in query_vector_space_model:
        tf_idf_square_summation += query_vector_space_model[qeury_word][3] * \
            query_vector_space_model[qeury_word][3]
        length_of_query = math.sqrt(tf_idf_square_summation)
        query_vector_space_model[word].append(
            query_vector_space_model[word][3]/length_of_query)


sim = {}
fake = 0
for file in files:
    similarity = 0
    for word in query_vector_space_model:
        if word in vector_space_model[file]:
            similarity += query_vector_space_model[word][4] * \
                vector_space_model[file][word][4]
    sim[file] = [similarity]
    print("Similarity between query and document " +
          file + ": " + str(similarity))
    if similarity == 0:
        fake += 1
print("\n")
if fake < len(files):
    print("The rank of docs is : ")
    sort_sim = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    for i in sort_sim:
        print(i[0])
    print("\n")
else:
    print("There is no rank because the query wasn't found !")
    print("\n")

