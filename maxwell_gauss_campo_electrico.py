import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros de la animación
frames = 60
radius = 1
num_lines = 20

# Crear figura y ejes
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_title('Ley de Gauss para el Campo Eléctrico')

# Carga positiva en el origen
charge = 1  # Carga positiva

# Crear líneas de campo
lines = []
angles = np.linspace(0, 2*np.pi, num_lines, endpoint=False)

for angle in angles:
    line, = ax.plot([], [], 'r-')
    lines.append(line)

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(i):
    for idx, angle in enumerate(angles):
        # Simular expansión de las líneas de campo
        r = radius + (i / frames) * 1.5
        x = [0, r * np.cos(angle)]
        y = [0, r * np.sin(angle)]
        lines[idx].set_data(x, y)
    return lines

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=frames, interval=100, blit=True)

# Guardar la animación como GIF
ani.save('ley_gauss_campo_electrico.gif', writer='pillow')

plt.close()
