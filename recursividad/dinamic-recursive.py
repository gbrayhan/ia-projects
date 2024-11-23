import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os

# Definir el grafo
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

# Parámetros de búsqueda
start_node = 'A'
target_node = 'H'
visited = set()
path = []

# Crear el grafo dirigido usando networkx
G = nx.DiGraph(graph)
pos = nx.spring_layout(G, seed=42)  # Posiciones fijas para consistencia en el GIF

# Lista para almacenar los frames
frames = []

# Directorio temporal para almacenar las imágenes
temp_dir = 'temp_dfs_frames'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

frame_count = 0  # Contador de frames


def save_frame(current_node, visited, path, frame_count, message=''):
    plt.figure(figsize=(8, 6))

    # Dibujar nodos
    node_colors = []
    for node in G.nodes():
        if node == current_node:
            node_colors.append('orange')  # Nodo actual
        elif node in path:
            node_colors.append('red')  # Camino actual
        elif node in visited:
            node_colors.append('lightblue')  # Visitados
        else:
            node_colors.append('lightgrey')  # No visitados

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500)

    # Dibujar aristas
    edge_colors = []
    for edge in G.edges():
        if edge in list(zip(path, path[1:])):
            edge_colors.append('red')  # Aristas en el camino
        else:
            edge_colors.append('grey')  # Otras aristas

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True)

    # Dibujar etiquetas
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Título con el mensaje actual
    plt.title(message, fontsize=14)
    plt.axis('off')

    # Guardar el frame actual
    filename = os.path.join(temp_dir, f'frame_{frame_count:03d}.png')
    plt.savefig(filename)
    plt.close()
    frames.append(filename)


def dfs_path_gif(graph, current_node, target_node, visited, path, frame_count):
    # Agregar el nodo actual al camino
    path.append(current_node)
    frame_count = frame_count + 1
    save_frame(current_node, visited, path, frame_count, f"Visitando {current_node}")

    # Condición de parada: se encontró el objetivo
    if current_node == target_node:
        frame_count = frame_count + 1
        save_frame(current_node, visited, path, frame_count, f"¡Encontrado {target_node}!")
        return path.copy(), frame_count

    # Marcar el nodo como visitado
    visited.add(current_node)

    # Recorrer los vecinos
    for neighbor in graph[current_node]:
        if neighbor not in visited:
            result, frame_count = dfs_path_gif(graph, neighbor, target_node, visited, path, frame_count)
            if result:  # Si se encuentra el camino, devolverlo
                return result, frame_count

    # Si no se encuentra el camino en esta rama, retroceder
    path.pop()
    frame_count = frame_count + 1
    save_frame(current_node, visited, path, frame_count, f"Retrocediendo desde {current_node}")
    return None, frame_count


# Ejecutar DFS y generar frames
result, frame_count = dfs_path_gif(graph, start_node, target_node, visited, path, frame_count)

# Crear el GIF usando imageio
gif_filename = 'dfs_search.gif'
with imageio.get_writer(gif_filename, mode='I', duration=2) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Limpiar los archivos temporales
for frame in frames:
    os.remove(frame)
os.rmdir(temp_dir)

# Mostrar el resultado
if result:
    print(f"Camino de {start_node} a {target_node}: {result}")
    print(f"El GIF se ha guardado como '{gif_filename}'.")
else:
    print(f"No hay camino de {start_node} a {target_node}.")
    print(f"El GIF se ha guardado como '{gif_filename}'.")
