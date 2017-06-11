#!/usr/bin/env python
# coding: utf8

# versão do python usada: Python 2.7.6

#### Gera todos os boxplots
#### uso: ./boxplots.py

## Maquina de teste: mercurio.eclipse.ime.usp.br

#===================== Referencia de boxplot===================== 
# http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/

from __future__ import unicode_literals
import numpy as np
import matplotlib as mpl 
mpl.use('agg') ## agg backend is used to create plot as a .png file
import matplotlib.pyplot as plt 
import re
import os

lista_modos = ['cuda', 'sequencial']
lista_algs = ['aes', 'des', 'blowfish']

ntreads_min = ""
ntreads_max = ""

###################### parser dos tempos de cada arquivo #############################



for modo in lista_modos:
	for alg in lista_algs:
		lista_arqs = os.listdir("results/"+modo+"/"+alg)

		for filename in lista_arqs:
			tamArquivo = os.path.getsize ("../sample_files/"+filename[0:-4])
	
			arq = open("results/"+modo+"/"+alg+"/"+filename, 'r')
			fcontent = arq.read()
			tempos = re.findall(r"\d+[,.]\d+ seconds time elapsed", fcontent)

			print "\n"
			print "results/"+modo+"/"+alg+"/"+filename
			dados = []

			for x in tempos:
				aux = x.split(" seconds time elapsed")
				dados.append(float(aux[0].replace(",", ".")))

			print dados

			arq.close()
			
			title = '10 execuções;  Teste: '

			if modo == 'cuda':
				title += 'GPU+CPU;  '
				ntreads_max = 1024
				ntreads_min = 8
			else:
				title += 'CPU;  '
				ntreads_max = 1
				ntreads_min = 1

			title += ('algoritmo: '+alg)
			title += (';  entrada: '+filename[0:-4])
			title += (';  size file: '+str(tamArquivo/1000)+' kB')


			data_to_plot = []
			max_value = 0
			entradas = []
			nt = ntreads_min
			j = 0
			while nt <= ntreads_max:
				entradas.append(str(nt))
				amostra = dados[(j * 10):((j + 1) * 10)]
				amostra.sort()
				
				if amostra[-1] > max_value:
					max_value = amostra[-1]

				data_to_plot.append(amostra)

				nt *= 2
				j += 1

			# Cria uma instancia de figura e eixos
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))

			# Cria o boxplot
			box_plot = ax.boxplot(data_to_plot)

			ax.set_xticklabels(entradas)
			ax.yaxis.grid(True)
		
			plt.xlabel('Numero de threads', fontsize=10, color='red')
			plt.ylabel('Tempo Em Segundos', fontsize=10, color='red')
			plt.title(title, fontsize=10)

			## Remove marcadores de eixos
			ax.get_xaxis().tick_bottom()
			ax.get_yaxis().tick_left()

			max_value = round(max_value, 5) 
			interval = round(max_value/30.0, 5)
			print "maximo = ", max_value
			print "interval = ", interval
			print "*************************"
			
			plt.yticks(np.arange(0, max_value + 3 * interval, interval))
	
			# Salva a figura
			fig.savefig("graphics/"+modo+"/"+alg+"/"+filename[0:-8]+'.png', bbox_inches='tight')
			

