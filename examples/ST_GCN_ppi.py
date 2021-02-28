import numpy as np
import tensorflow as tf
from spektral.datasets.graphsage import PPI


ppi_data = PPI()
ppi_data.download()
ppi_graph = ppi_data.read()
print(ppi_graph)

