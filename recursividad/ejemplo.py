def dfs_path(graph, current_node, target_node, visited, path):
    # Agregar el nodo actual al camino
    path.append(current_node)

    # Condición de parada: se encontró el objetivo
    if current_node == target_node:
        return path

    # Marcar el nodo como visitado
    visited.add(current_node)

    # Recorrer los vecinos
    for neighbor in graph[current_node]:
        if neighbor not in visited:
            result = dfs_path(graph, neighbor, target_node, visited, path)
            if result:  # Si se encuentra el camino, devolverlo
                return result

    # Si no se encuentra el camino en esta rama, retroceder
    path.pop()
    return None


# Árbol más complejo
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['G', 'H'],
    'F': [],
    'G': [],
    'H': []
}

# Búsqueda de un camino entre dos nodos
start_node = 'A'
target_node = 'H'
visited = set()
path = []

result = dfs_path(graph, start_node, target_node, visited, path)

if result:
    print(f"Camino de {start_node} a {target_node}: {result}")
else:
    print(f"No hay camino de {start_node} a {target_node}.")
