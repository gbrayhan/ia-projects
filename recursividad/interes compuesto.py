def compound_interest_daily_recursive(principal, annual_rate, days, total_days=365):
    """
    Calcula el interés compuesto diario usando recursividad.

    :param principal: Cantidad inicial invertida.
    :param annual_rate: Tasa de interés anual (como decimal).
    :param days: Número de días a calcular.
    :param total_days: Días en un año para calcular la tasa diaria (por defecto 365).
    :return: Valor acumulado después de los días especificados.
    """
    if days == 0:  # Caso base: si no quedan días, devuelve el capital inicial.
        return principal
    # Tasa diaria
    daily_rate = annual_rate / total_days
    # Llamada recursiva con el capital actualizado
    return compound_interest_daily_recursive(principal * (1 + daily_rate), annual_rate, days - 1, total_days)


# Ejemplo de uso
principal = 20000  # Capital inicial
annual_rate = 0.12  # Tasa de interés anual (5%)
days = 365 * 4  # Número de días (3 años)

final_amount = compound_interest_daily_recursive(principal, annual_rate, days)
print(f"El monto acumulado después de {days} días es: {final_amount:.2f}")
