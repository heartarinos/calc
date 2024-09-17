import sympy as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

class UniMathAI:
    def __init__(self):
        # Initialize SymPy symbols
        self.symbols = sp.symbols('x y z')

        # Initialize TensorFlow variables
        self.x = tf.Variable(np.array([]), dtype=tf.float64)

        # Placeholder for the equation
        self.eq = []

    def add_equation(self, equation):
        self.eq.append(equation)

    def solve_system(self):
        # Solve the system of equations
        results = sp.dsolve(self.eq, self.symbols)

        # Substitute the solved values into the x, y, z variables
        self.x = tf.constant(results[self.symbols[0]], dtype=tf.float64)
        if len(results) > 1:
            self.y = tf.constant(results[self.symbols[1]], dtype=tf.float64)
            if len(results) > 2:
                self.z = tf.constant(results[self.symbols[2]], dtype=tf.float64)

    def graph_it(self, expression, var='x'):
        # Convert the SymPy expression into a TensorFlow expression
        tf_expr = tf.constant(sp.sympify(expression), dtype=tf.float64)

        # Generate data for the graph
        data = tf.numerical_grad(tf_expr, self.x)

        # Generate the graph
        plt.plot(self.x.numpy(), data.numpy())
        plt.xlabel(var)
        plt.ylabel("Value")
        plt.title("Graph of {}", expression)
        plt.show()

# Initialize the University Math AI
ai = UniMathAI()

while True:
    user_input = input("You: ")

    if "exit" in user_input.lower():
        break

    elif "equation" in user_input:
        # Add the equation to be solved
        ai.add_equation(user_input)

    elif "solve" in user_input:
        # Solve the system of equations
        ai.solve_system()

        # Print the solutions
        print("Solutions:")
        print(f"x = {ai.x.numpy()}")
        if hasattr(ai, 'y'):
            print(f"y = {ai.y.numpy()}")
        if hasattr(ai, 'z'):
            print(f"z = {ai.z.numpy()}")

    elif "graph" in user_input:
        # Graph the expression
        expression = user_input.split("graph ")[1].strip()
        ai.graph_it(expression)

    else:
        # Invalid input, let's joke about it!
        print("I'm lost! Make sure you're not giving me quantum mechanics problems already. :)")
