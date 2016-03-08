.. pyStatKs documentation master file, created by
   sphinx-quickstart on Tue Mar  8 09:54:13 2016.

Welcome to pyStatKs's documentation!
====================================

pyStatKs is a statistics package to process NHL© game data.
It uses the data scraper `nhlscrapi <https://github.com/robhowley/nhlscrapi>`_
to fetch the data and does some analysis using packages as `pyBrain <http://pybrain.org/>`_
for neural networks and `scipy <http://www.scipy.org/>`_ and `numpy <http://www.numpy.org/>`_
for statistics.

pyStatKs in use:
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
I used the modules in a `dataMining ipython sheet <https://github.com/sjjko/nhlAnalysis/blob/master/ipython/dataMiningNHL.ipynb>`_
and a Neural Network `ipython sheet <https://github.com/sjjko/nhlAnalysis/blob/master/ipython/neuralNHL.ipynb>`_. Modules are
furthermore used in my `R Jupyter notebooks <https://github.com/sjjko/nhlAnalysis/tree/master/R>`_.

Structure of pyStatKs
++++++++++++++++++++++++++++++++++++++++++++++++++

**The pyStatKs module consists of two classes for data harvesting:**

* :ref:`MinerC`

the class fetching data using nhlscrapi

* :ref:`setupNhlDataC`

class for formatting the data read

**and classes for data processing and neural network analysis:**

* :ref:`AnalysisC`
as the base class for all the classes below.

* :ref:`NeuralNetworkAnalysisC`
contains routines for the setup, training and analysis of neural networks

* :ref:`WaveletAnalysisC`
contains routines for performing wavelet analysis on the dataset

* :ref:`BivariateAnalysisC`
doing some bivariate data analysis

* :ref:`MultivariateStatisticsC`
doing some multivariate statistics


Contents:

.. toctree::
   :maxdepth: 2
   
   
Class-documentation 
++++++++++++++++++++++++++++++++++++++++++++++++++
   

.. automodule:: pyStatKs
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

extlinks = {'nhlscrapi': (' ')}