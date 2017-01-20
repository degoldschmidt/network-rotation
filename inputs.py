    if t > T/10: 
        x[i] = (0.9995*x[i-1] + 0.0005)
    else:
        x[i] = x[i-1]