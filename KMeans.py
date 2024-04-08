#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:04:01 2024

@author: tiago
"""

import numpy as np
import matplotlib.pyplot as plt

def gera_dados ( n_centros, n_features = 2, pontos = 10, s = 1.0 ):
    
    # Gera os centros dos clusters
    centros = np.random.rand(n_centros,n_features) * 100
    
    # Gera as dispersões dos dados
    dados = np.random.randn(pontos,n_features) * s
    
    # Loop para somar os valores das centros às dispersões aleatoriamente
    for i in range(pontos):
        # Sorteia o centro a ser usado
        p = int( np.floor( np.random.rand() * n_centros ) )
        # Soma os valores
        dados[i,:] = dados[i,:] + centros[p,:]
    
    # Retorna os valores finais
    return dados


def clustering_kmeans( dados, n_clusters = 2 , tol = 1e-3, verbose = False):
    
    # Recupera valores de número de dados e de características
    n_dados, n_features = dados.shape
    
    # Aloca espaço na memória para os centros dos agrupamentos no KMeans
    Xc = np.zeros( (n_clusters, n_features) )
    
    # Inicializa os centros por meio de sorteio aleatório de n_clusters dados
    index = np.floor( np.random.rand(n_clusters) * n_dados ).astype(int)
    for i in range(n_clusters):
        Xc[i,:] = dados[index[i],:]
    
    teste = tol
    while( teste >= tol):
        
        # Reserva espaço na memória para calcular as distências
        dist = np.zeros( (n_dados,n_clusters) )
        
        # Calcula as distâncias para cada centro de agrupamento
        for cluster in range(n_clusters):
            for dado in range(n_dados):
                dist[dado,cluster] = np.sqrt( ((dados[dado,:] - Xc[cluster,:])**2).sum() )
        
        # Atribui a cada cada um agrupamento pela menor distância ao centro
        categorias = dist.argmin(axis=1).astype(int)
        
        # Calcula a nova posição dos centros
        Xc_new = np.zeros( (n_clusters, n_features) )
        for cluster in range(n_clusters):
            Xc_new[cluster,:] = dados[ categorias == cluster ,:].mean(axis=0)
        
        # Calcula a maior distância entre o centro novo e o anterior
        teste = np.sqrt( ( ( Xc - Xc_new )**2 ).sum(axis=1) ).max()
        
        if verbose == True:
            print(teste)
            plt.figure()
            plt.scatter(dados[:,0], dados[:,1], c = categorias)
            plt.scatter(Xc[:,0], Xc[:,1], c = 'r', s = 100)
            plt.scatter(Xc_new[:,0], Xc_new[:,1], c = 'b', s = 100)

        # Atualiza as coordenadas do centro
        Xc = Xc_new
    
    return categorias

# Como usar o mesmo padrão do Scikit-Learn?
class KMeans():
    n_clusters = 0
    tol = 0
    
    def __init__(self, n_clusters = 2, tol = 1e-3):
        self.n_clusters = n_clusters
        self.tol = tol
        return None
    
    def fit( self, dados ):
        return clustering_kmeans( dados, n_clusters = self.n_clusters, tol = self.tol )


dados = gera_dados(n_centros = 3, pontos = 40, s = 5)

#categorias = clustering_kmeans (dados, n_clusters = 3, verbose = True)

clustering = KMeans(n_clusters = 3)
categorias = clustering.fit(dados)

plt.scatter(dados[:,0], dados[:,1], c = categorias)

