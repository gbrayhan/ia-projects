from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch  # Asegúrate de importar torch

# Verificar si PyTorch está utilizando GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

# Cargar el tokenizador y el modelo preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
model.to(device)  # Mover el modelo al dispositivo adecuado

# Preparar una oración de ejemplo
sentence = "La inteligencia artificial está transformando el mundo."

# Tokenizar la oración
inputs = tokenizer(sentence, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}  # Mover los inputs al dispositivo adecuado

# Obtener las salidas del modelo, incluyendo las atenciones
with torch.no_grad():  # Desactivar el cálculo de gradientes para ahorrar memoria
    outputs = model(**inputs)

# Obtener las atenciones del último bloque de capas
attentions = outputs.attentions  # Tuple de tensores

# Seleccionar la atención de la última capa y la primera cabeza de atención
last_layer_attention = attentions[-1][0]  # (num_heads, seq_length, seq_length)
head_attention = last_layer_attention[0].detach().cpu().numpy()  # Primera cabeza y mover a CPU

# Obtener los tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())

# Visualizar el mapa de atención
plt.figure(figsize=(10, 8))
sns.heatmap(head_attention, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
plt.title('Mapa de Atención de la Primera Cabeza en la Última Capa')
plt.xlabel('Tokens de Entrada')
plt.ylabel('Tokens de Salida')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()  # Ajustar el diseño para evitar recortes
plt.show()
