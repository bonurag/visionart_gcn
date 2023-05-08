<h1 align="center">Graph Neural Network</h1>

<p align="center"><img src="img/graph_data.png" height="400"/></p>

## 1. Introduzione

All'interno di questo repository sono presenti alcune slide prese dal corso [CS224W: Machine Learning with Graphs](http://snap.stanford.edu/class/cs224w-2020/) tenuto dal Prof. [Jure Leskovec](https://profiles.stanford.edu/jure-leskovec).
Non sono state soggette a modifiche se non a qualche taglio utile per evitare di aggiungere informazioni poco necessarie ai fini di questo breve seminario, il cui scopo è quello di fornire una panoramica generale su questo nuovo
modello di reti, che negli ultimi anni sono diventate oggetto di molti studi in diversi campi applicativi.

Per chi volesse approfondire, il corso indicato (CS224W: Machine Learning with Graphs) è molto ampio e ben fatto, un piccolo riassunto di quello che verrà introdotto è indicato nella tabella in calce:

| <p align="center">Lezione</p> |                               <p align="center">Descrizione</p>                           |   <p align="center">Sintesi</p>   |
| ----------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------- |
| <p align="center">1</p>       | [Introduzione](Slide/Lezione_1/01-intro_clean.pdf)                                        |                                   |
| <p align="center">2</p>       | [Metodi tradizionali di ML su grafi](Slide/Lezione_2/02-tradition-ml_clean.pdf)           |                                   |
| <p align="center">3</p>       | [Rappresentazione vettoriale di nodi](Slide/Lezione_3/03-nodeemb_clean.pdf)               |                                   |
| <p align="center">4</p>       | [Propagazione label per la classificazione dei nodi](Slide/Lezione_4/04-message_clean.pdf)|                                   |
| <p align="center">5</p>       | [Reti neurali su grafo 1: Modello GNN](Slide/Lezione_5/05-GNN1_clean.pdf)                 |                                   |
| <p align="center">6</p>       | [Reti neurali su grafo 2: Spazio di progettazione](Slide/Lezione_6/06-GNN1_clean.pdf)     |                                   |

## 2. Codice

Per facilitare la comprensione di quanto riportato all'interno delle slide, è stato realizzato un piccolo progetto prettamente didattico che sfrutta uno dei dataset messo a disposizione all'interno del framework [PyG](https://pytorch-geometric.readthedocs.io/en/latest/). E' il framework di riferimento per applicazioni di questo tipo, ve ne sono molti altri, ma essendo un estensione del framework di PyTorch è molto utilizzato in ambito scientifico/accademico. Il dataset utilizzato è una versione revisionata del ben noto MNIST, denominato MNIST superpixels. Il dataset MNIST superpixels, tratto dall'articolo ["Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs"](https://arxiv.org/pdf/1611.08402.pdf), contiene 70.000 grafi con 75 nodi ciascuno. Ogni grafo è etichettato da una delle 10 classi.

E' possibile lanciare il codice direttamente da terminale con lo script sottostante per la fase di training del modello:

```python
!python train_model.py
```

mentre per quando riguarda il test del modello lanciare quello sottostante.:

```python
!python test_model.py
```

Diversamente è stato realizzato un jupyter notebook [MNIST_SuperPixel_GCN](src/MNIST_SuperPixel_GCN.ipynb) che permette di visualizzare anche le curve di accuracy e loss tramite [tensorboardX](https://github.com/lanpa/tensorboardX).

# 3. Modello

Il modello costruito è quello rappresentato nell'immagine sottostante presa come riferimento dal paper [A Graph Neural Network for superpixel image
classification](https://iopscience.iop.org/article/10.1088/1742-6596/1871/1/012071/pdf), e prevede l'utilizzo di tre layer [GATConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html) più lo skip di delle connessioni tra i vari layer che vengono nuovamente aggregati nello steato finale con i nodi degli strati precedenti. Il GATConv, è la classe che modella una Graph Attention Network, il cui coefficiente di attention è calcolato secondo oppurtuna relazione matematica. Una trattazione rigorosa del parametro di attention è presente all'interno del paper [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf).

<p align="center"><img src="img/gnn_model.png" height="200"/></p>

# 4.Approfondimenti

In questa sezione sono riportati alcuni documenti di interesse utili per eventuali approfondimenti.
