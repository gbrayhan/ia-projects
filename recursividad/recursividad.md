# Recursividad


Explicación del algoritmo:
Procesamiento delegado a sí mismo:

En cada paso, el algoritmo verifica si el nodo actual es el nodo objetivo.
Si no lo es, marca el nodo como visitado.
Procesamiento delegado a la siguiente recursión:

Para cada nodo vecino del nodo actual, llama recursivamente a la función para explorar ese vecino.
La recursión continúa hasta encontrar el objetivo o terminar de explorar todas las opciones.
Condición de parada:

Si el nodo actual es igual al nodo objetivo, hemos encontrado el camino.
Si no quedan nodos por visitar, la búsqueda termina.
