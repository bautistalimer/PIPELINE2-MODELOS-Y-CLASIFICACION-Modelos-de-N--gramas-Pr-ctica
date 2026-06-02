
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ingresarCorpus(txt):
    #Carga corpus en la misma carpeta (o poner direccion):
    with open(txt, "r", encoding="latin-1") as f:
        corpus = f.read()
    return corpus

def limpiarCorpus(c):

    #Carga el lenguaje:
    nlp = spacy.load("es_core_news_sm")

    for l in c: #Revisa caracter por caracter para eliminar los indeseados 
        if not l.isalnum() and not l.isspace() and l != "\n" : #Dejo los espacios y los saltos ("\n")
           c = c.replace(l, "") #Reemplaza el caracter por "" (nada) 

    c = c.split("\n") #Genera una lista de oraciones o parrafos usando los saltos como separador

    c_limpio = [] #Prepara una lista donde retornar el resultado

    for parrafo in nlp.pipe(c): #nlp.pipe() tokeniza las palabras de cada parrafo
        linea_actual = [] #Lista temporal
        for token in parrafo:
            if not token.is_stop and not token.is_punct and not token.is_space: #Ultimo filtrado
                linea_actual.append(token.lemma_.lower()) #Ingresa a la lista temporal los lemas de los parrafos
        
        c_limpio.append(" ".join(linea_actual)) #Ingresa el parrafo lematizado a la lista

    return c_limpio

def generarNgramas(corpus, rango1, rango2, apariciones):
    
    #Define vectorizador con sus condiciones:
    vect_ngram = CountVectorizer(ngram_range=(rango1, rango2), min_df=apariciones) #ngramas que aparezcan al menos n veces

    #Genera matriz:
    matriz_ngram = vect_ngram.fit_transform(corpus)
    
    return (vect_ngram, matriz_ngram) #Retorna tupla con resultados

def mostrarResultadosNgrama(ngrama):
    #Grafico de barras:
    pd.DataFrame(ngrama[1].sum(axis=0).T, #ngrama[n] usa el 2do elemento de la tupla
             index=ngrama[0].get_feature_names_out(), #ngrama[n] usa el 1er elemento de la tupla
             columns=['Frecuencia']).sort_values(by='Frecuencia',
                                            ascending=True).plot(kind='barh', 
                                                                 title='N-gramas',
                                                                 figsize=(14, 7))

    print(f"""Matriz Ngramas:\n
    {ngrama[1].toarray()}\n
    Se compone de los siguientes Ngramas:\n
    {ngrama[0].get_feature_names_out()}\n""")

    plt.tight_layout() #Ajusta el grafico a los limites de la ventana
    plt.show() #Mostrar grafico

entrada = "corpus_educacion.txt" #Posible input()

mostrarResultadosNgrama(generarNgramas(limpiarCorpus(ingresarCorpus(entrada)), 2, 2, 2))
mostrarResultadosNgrama(generarNgramas(limpiarCorpus(ingresarCorpus(entrada)), 3, 3, 2))
