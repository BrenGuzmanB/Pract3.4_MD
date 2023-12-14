"""
Created on Sun Dec 10 01:36:36 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""

import numpy as np


class support_vector_machine:
    def __init__(self, learning_rate=0.001, lambda_=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_  # Factor de penaluzación
        self.epochs = epochs  # num iteraciones
        self.w = None  # weights
        self.b = None  # intercept

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # inicializar en ceros
        self.b = 0  # inicializar en 0

        # Mapear etiquetas de clase {-1, 1}
        y_mapped = np.where(y <= 0, -1, 1)

        for epoch in range(self.epochs): # Épocas para el descenso del gradiente
            self.gradient_descent_step(X, y_mapped)

        return self.w, self.b

    def gradient_descent_step(self, X, y):  # Descenso del gradiente
        for i, Xi in enumerate(X):
            # Condición basada en la diferencia entre la puntuación predicha y el margen
            condition = y[i] * (np.dot(Xi, self.w) - self.b) >= 1
            
            # Actualizar los weights (w) 
            self.w -= self.learning_rate * (2 * self.lambda_ * self.w) if condition else self.learning_rate * (
                    2 * self.lambda_ * self.w - np.dot(Xi, y[i]))
            
            # Actualizar el sesgo (b)
            self.b -= self.learning_rate * y[i] if not condition else 0

