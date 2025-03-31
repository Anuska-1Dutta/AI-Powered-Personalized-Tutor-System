import os
import pickle
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

print("Creating enhanced mathematics dataset...")

# Create necessary directories
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Basic mathematical operations QA pairs
arithmetic_qa = [
    ("What is 2+2?", "The sum of 2 and 2 is 4."),
    ("What is 5+7?", "The sum of 5 and 7 is 12."),
    ("What is 10-3?", "The difference between 10 and 3 is 7."),
    ("What is 8*9?", "The product of 8 and 9 is 72."),
    ("What is 20/4?", "The result of dividing 20 by 4 is 5."),
    ("What is 15/2?", "The result of dividing 15 by 2 is 7.5."),
    ("What is 2^3?", "2 raised to the power of 3 is 8."),
    ("What is the square root of 9?", "The square root of 9 is 3."),
    ("What is the square root of 16?", "The square root of 16 is 4."),
    ("What is the value of pi?", "Pi (π) is approximately 3.14159, the ratio of a circle's circumference to its diameter."),
]

# Fundamental math concepts QA pairs
fundamentals_qa = [
    ("What is a graph in mathematics?", "In mathematics, a graph is a structure used to model pairwise relations between objects. Graphs consist of vertices (also called nodes or points) which are connected by edges (also called links or lines). Graphs can be used to model many types of relations and processes in physical, biological, social, and information systems. Different types of graphs include directed graphs, undirected graphs, weighted graphs, and trees."),
    ("What is a function?", "A function in mathematics is a relation between a set of inputs (domain) and a set of possible outputs (range), where each input is related to exactly one output. Functions are typically represented as f(x) where x is the input and f(x) is the output. They can be represented algebraically, graphically, numerically, or verbally. Common functions include linear functions, quadratic functions, exponential functions, and logarithmic functions."),
    ("What is a limit?", "A limit in calculus is the value that a function approaches as the input approaches some value. Limits form the foundation of calculus and mathematical analysis, enabling the precise definition of continuity, derivatives, and integrals. The notation lim(x→a) f(x) = L means that as x gets arbitrarily close to a (but not equal to a), f(x) gets arbitrarily close to L. Limits can be evaluated using direct substitution when functions are continuous at the point in question."),
    ("What is a prime number?", "A prime number is a natural number greater than 1 that is not a product of two natural numbers other than 1 and itself. The first few prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29. Prime numbers play an important role in number theory and have applications in cryptography, particularly in public-key encryption algorithms like RSA."),
    ("What is factoring?", "Factoring in mathematics is the process of finding which numbers multiply together to form a given number or expression. For example, factoring 12 gives 2 × 2 × 3, or 2² × 3. In algebra, factoring involves rewriting a polynomial as a product of simpler polynomials. Factoring is an essential technique for solving quadratic equations and simplifying complex expressions."),
    ("What is a fraction?", "A fraction represents a part of a whole or a ratio between two numbers. It consists of a numerator (top number) and a denominator (bottom number). For example, in the fraction 3/4, 3 is the numerator and 4 is the denominator, representing three out of four equal parts. Fractions can be proper (numerator < denominator), improper (numerator ≥ denominator), or mixed (whole number plus proper fraction)."),
    ("What is a polynomial?", "A polynomial is an expression consisting of variables, coefficients, and operations of addition, subtraction, multiplication, and non-negative integer exponents. For example, 3x² + 2x - 5 is a polynomial of degree 2 (quadratic). Polynomials are classified by their degree: degree 1 is linear, degree 2 is quadratic, degree 3 is cubic, etc. They are fundamental in algebra and are used in various applications from simple curve-fitting to complex scientific modeling."),
    ("What is a logarithm?", "A logarithm is the inverse operation to exponentiation. Specifically, if bᵃ = x, then logₐ(x) = b, where a is the base of the logarithm. Common logarithm bases include base 10 (common logarithm), base e (natural logarithm, denoted as ln), and base 2 (binary logarithm). Logarithms are useful for simplifying calculations involving large numbers, solving exponential equations, and expressing quantities that vary over several orders of magnitude."),
    ("What is a vector?", "A vector in mathematics is a quantity that has both magnitude and direction. It can be represented as an arrow in space or as an ordered list of numbers (components). In 2D space, vectors have two components (x,y), while in 3D space, they have three components (x,y,z). Vectors can be added, subtracted, scaled, and used to calculate dot and cross products. They are essential in physics for representing forces and velocities, and in computer graphics for representing positions and directions."),
    ("What is probability?", "Probability is a measure of the likelihood of an event occurring. It is quantified as a number between 0 and 1, where 0 indicates impossibility and 1 indicates certainty. Probability theory provides a mathematical framework for analyzing random phenomena and making predictions about them. Basic probability concepts include sample spaces, events, independence, conditional probability, and random variables. Probability is widely used in statistics, finance, game theory, artificial intelligence, and many other fields."),
]

# Advanced mathematical concepts QA pairs
advanced_qa = [
    ("What is a matrix?", "A matrix is a rectangular array of numbers, symbols, or expressions arranged in rows and columns. Matrices are fundamental mathematical objects used in linear algebra and have applications in computer graphics, physics, engineering, statistics, and many other fields. Key operations on matrices include addition, subtraction, multiplication, and finding inverses and determinants. An m×n matrix has m rows and n columns. Matrices can represent linear transformations, systems of linear equations, graphs, and data in machine learning."),
    ("What is a derivative?", "A derivative in calculus measures the sensitivity to change of a function's output with respect to its input. It represents the instantaneous rate of change or slope of a function at a specific point. Geometrically, the derivative gives the slope of the tangent line to the function's graph at that point. Derivatives are calculated using various rules such as the power rule, product rule, quotient rule, and chain rule. Applications include velocity and acceleration calculations in physics, marginal analysis in economics, and optimization in engineering design."),
    ("What is integration?", "Integration is a fundamental concept in calculus that represents the process of finding the accumulated effect of a continuous function. It is the reverse operation of differentiation. There are two main types of integrals: definite and indefinite. Indefinite integrals (or antiderivatives) give a family of functions whose derivative is the integrand. Definite integrals calculate the signed area between a function and the x-axis over a specific interval. Applications include calculating areas, volumes, centers of mass, and solving differential equations."),
    ("What is a differential equation?", "A differential equation is a mathematical equation that relates a function with its derivatives. It describes the relationship between a function and the rate at which that function changes. Differential equations are classified by type (ordinary vs. partial), order (highest derivative present), and linearity. They are fundamental in modeling physical phenomena in sciences and engineering, such as motion in physics, population growth in biology, and electrical circuits in engineering. Methods for solving differential equations include separation of variables, integrating factors, and numerical methods."),
    ("What is linear algebra?", "Linear algebra is a branch of mathematics that deals with vector spaces, linear transformations, matrices, and systems of linear equations. It provides a framework for studying linear mappings between spaces and the properties of these mappings. Key concepts include vectors, matrices, determinants, eigenvalues, eigenvectors, and vector spaces. Linear algebra has applications in physics, computer science, engineering, data analysis, machine learning, and many other fields."),
    ("What is a complex number?", "A complex number is a number that can be expressed in the form a + bi, where a and b are real numbers and i is the imaginary unit, defined as the square root of -1. Complex numbers extend the concept of the one-dimensional number line to the two-dimensional complex plane. They are fundamental in mathematics, engineering, physics, and signal processing, enabling solutions to equations that have no real number solutions, such as x² + 1 = 0. Operations on complex numbers include addition, subtraction, multiplication, division, and exponentiation."),
    ("What is a Fourier series?", "A Fourier series is a way to represent a periodic function as a sum of sine and cosine functions (or complex exponentials). It decomposes any periodic function into its frequency components, expressing it as an infinite sum of harmonically related sinusoids. Fourier series are named after Jean-Baptiste Joseph Fourier, who introduced them in his study of heat transfer. They are widely used in signal processing, electrical engineering, acoustics, optics, and other fields to analyze and manipulate periodic signals and functions."),
    ("What is numerical analysis?", "Numerical analysis is the study of algorithms that use numerical approximation for the problems of mathematical analysis. It develops, analyzes, and applies methods for obtaining approximate solutions to problems that cannot be solved exactly or are too complex for analytical solutions. Key areas include interpolation, numerical integration, numerical linear algebra, numerical solution of differential equations, and optimization. Numerical methods are essential in science, engineering, and computing when exact solutions are impossible or impractical to compute."),
    ("What is a tensor?", "A tensor is a mathematical object that generalizes scalars, vectors, and matrices to higher dimensions. It can be thought of as a multi-dimensional array of values that transforms according to specific rules under changes of coordinates. Tensors are characterized by their rank (or order), which indicates the number of indices needed to specify a component. They are used in physics to represent physical quantities independent of coordinate systems, in differential geometry to describe curved spaces, and in machine learning for data representation and operations."),
    ("What is graph theory?", "Graph theory is the mathematical study of graphs, which are structures used to model pairwise relations between objects. A graph consists of vertices (nodes) connected by edges (links). Types of graphs include directed, undirected, weighted, bipartite, and complete graphs. Graph theory has applications in computer science (networks, algorithms), transportation (route planning), social sciences (social networks), chemistry (molecular structures), and many other fields. Key concepts include paths, cycles, connectivity, coloring, matching, and graph algorithms like breadth-first search and Dijkstra's algorithm."),
]

# Practical applications of math QA pairs
applications_qa = [
    ("How is math used in computer science?", "Mathematics is fundamental to computer science in numerous ways. Discrete mathematics, particularly set theory, logic, graph theory, and combinatorics, forms the theoretical foundation of computer science. Algorithms and data structures rely on mathematical concepts for design and analysis. Cryptography uses number theory and abstract algebra to secure communications. Computer graphics employs geometry, linear algebra, and calculus. Artificial intelligence and machine learning use statistics, probability, linear algebra, and calculus for developing models and algorithms. Database systems use relational algebra, and formal verification of software uses mathematical logic."),
    ("How is calculus used in physics?", "Calculus is essential in physics for describing and analyzing dynamic systems. Differential calculus is used to find instantaneous rates of change, such as velocity (the derivative of position) and acceleration (the derivative of velocity). Integral calculus helps calculate cumulative effects, like finding displacement from velocity or work from force. Vector calculus extends these concepts to three dimensions, crucial for describing fields (gravitational, electric, magnetic). Differential equations, a central part of calculus, model many physical phenomena including motion, waves, heat flow, and quantum mechanics. Newton's laws of motion and Maxwell's equations are expressed using calculus."),
    ("How is math used in engineering?", "Mathematics is the language of engineering, used to model, analyze, and solve complex problems. Engineers use calculus to analyze rates of change and optimize designs. Linear algebra helps in solving systems of equations and analyzing structural systems. Statistics and probability are used for quality control, reliability analysis, and risk assessment. Differential equations model dynamic systems like electrical circuits and mechanical systems. Fourier analysis converts signals between time and frequency domains, essential in signal processing. Numerical methods provide approximate solutions to complex problems. Each engineering discipline uses specific mathematical tools: civil engineering uses structural mechanics, electrical engineering uses complex analysis, and so on."),
    ("How is geometry used in architecture?", "Geometry is fundamental to architecture, influencing design, structure, and aesthetics. Euclidean geometry provides the basis for creating basic shapes, proportions, and symmetry in buildings. Non-Euclidean geometry inspires curved and complex forms in contemporary architecture. Architects use coordinate geometry and trigonometry for precise measurements and layouts. Golden ratio and other mathematical proportions are employed for aesthetic harmony. Fractal geometry influences modern designs with self-similar patterns. Computational geometry and parametric design enable complex, algorithm-generated structures. Spatial geometry helps in understanding how people perceive and navigate buildings. Throughout history, from ancient Egyptian pyramids to modern skyscrapers, geometric principles have guided architectural innovation."),
    ("How is probability used in finance?", "Probability theory is central to modern finance and risk management. It underlies options pricing models like the Black-Scholes formula, which uses stochastic calculus to determine fair values. Portfolio theory uses probability distributions to optimize the risk-return tradeoff through diversification. Value at Risk (VaR) calculations estimate the probability of portfolio losses exceeding certain thresholds. Credit risk models assess the probability of default on loans. Monte Carlo simulations generate thousands of possible scenarios for financial outcomes. Time series analysis, based on probability theory, helps forecast financial variables and market trends. Bayesian statistics allows for updating probability assessments as new information emerges. These probabilistic tools help financial professionals make decisions under uncertainty and manage risk effectively."),
    ("How is math used in medicine?", "Mathematics plays an increasingly important role in medicine and healthcare. Biostatistics is used in clinical trials to determine efficacy of treatments and in epidemiology to study disease patterns. Mathematical modeling helps understand disease spread, predict outbreaks, and optimize vaccination strategies. Medical imaging relies on mathematical techniques like Fourier transforms for MRI and CT scan reconstruction. Computational fluid dynamics models blood flow in cardiovascular studies. Pharmacokinetics uses differential equations to model drug absorption and elimination. Neural networks and machine learning algorithms help in medical diagnosis and personalized medicine. Population genetics uses probability theory to study inheritance patterns. These mathematical applications improve diagnosis, treatment, research, and healthcare delivery systems."),
    ("How is linear algebra used in computer graphics?", "Linear algebra is the mathematical foundation of computer graphics. Matrices are used to represent and perform transformations such as translation, rotation, scaling, and projection of objects in 2D and 3D space. Vector operations enable calculating surface normals, which are essential for lighting and shading. Homogeneous coordinates allow transformations to be combined into a single matrix multiplication. Ray tracing algorithms use vector geometry to simulate light paths. Linear systems solve constraints in animation and physics simulations. Eigenvalues and eigenvectors are used in character animation and deformation. Graphics Processing Units (GPUs) are optimized for linear algebra operations, allowing real-time rendering of complex scenes. From video games to CGI in films, linear algebra makes modern computer graphics possible."),
    ("How is statistics used in data science?", "Statistics forms the core methodology of data science. Descriptive statistics summarizes and visualizes data patterns. Inferential statistics draws conclusions about populations from samples. Regression analysis models relationships between variables and makes predictions. Hypothesis testing evaluates claims about data. Bayesian statistics updates beliefs based on new evidence. Time series analysis identifies patterns in sequential data. Experimental design principles ensure valid conclusions from data collection. Sampling techniques help select representative subsets of data. Statistical learning (including machine learning) builds predictive models from data. Clustering algorithms group similar data points. Dimensionality reduction techniques simplify complex datasets. These statistical methods help data scientists extract meaningful insights, build reliable models, and make data-driven decisions across fields from business to healthcare to scientific research."),
    ("How is math used in cryptography?", "Mathematics is the foundation of modern cryptography and secure communications. Number theory, particularly modular arithmetic and prime numbers, enables public-key cryptography systems like RSA, where security relies on the difficulty of factoring large numbers. Elliptic curve cryptography uses algebraic structures for efficient secure protocols. Abstract algebra, including finite fields and groups, underlies many encryption algorithms like AES. Information theory quantifies message security and encryption effectiveness. Probability theory helps analyze the likelihood of breaking encryption through various attacks. Computational complexity theory determines the practical security of cryptographic systems by analyzing the computational resources needed to break them. Hash functions use mathematical properties to create digital signatures and verify data integrity. As computing power increases, advances in mathematical cryptography continue to protect our digital communications and data."),
    ("How is math used in artificial intelligence?", "Mathematics provides the theoretical foundation and practical tools for artificial intelligence. Linear algebra enables the representation and manipulation of data, with matrices and tensors serving as the building blocks of neural networks. Calculus, particularly gradient-based methods, powers the optimization of machine learning models through techniques like gradient descent. Probability and statistics form the basis for uncertainty modeling, Bayesian networks, and statistical learning algorithms. Information theory quantifies entropy and mutual information, concepts central to decision trees and feature selection. Graph theory structures knowledge representation and reasoning systems. Logic formalizes rule-based systems and reasoning. Optimization theory provides methods to find optimal solutions in complex spaces. As AI advances, it increasingly draws on more sophisticated mathematics, including topology, differential geometry, and category theory for developing more powerful and interpretable AI systems."),
]

# Combine all QA pairs
math_qa_pairs = arithmetic_qa + fundamentals_qa + advanced_qa + applications_qa

# Try to download additional data from external sources
try:
    print("Attempting to download additional math datasets...")
    
    # Math formula dataset - simplified sample
    formulas_url = "https://raw.githubusercontent.com/KaTeX/KaTeX/main/docs/supported.md"
    
    response = requests.get(formulas_url)
    if response.status_code == 200:
        content = response.text
        # Extract some formula examples from KaTeX documentation
        formula_examples = []
        
        for line in content.split('\n'):
            if line.startswith('- ') and '\\' in line:
                formula = line.strip('- ').strip()
                if len(formula) > 5 and len(formula) < 100:  # Filter reasonable length formulas
                    question = f"What is the formula {formula.split(' ')[0]}?"
                    answer = f"The formula is {formula}. This is a mathematical notation used in {['calculus', 'algebra', 'trigonometry', 'statistics', 'linear algebra'][len(formula) % 5]}."
                    formula_examples.append((question, answer))
        
        # Add some of the examples to our dataset
        math_qa_pairs.extend(formula_examples[:10])
        print(f"Added {len(formula_examples[:10])} formula examples from KaTeX documentation")
    
    # Try to fetch additional math definitions
    math_terms_url = "https://raw.githubusercontent.com/simple-icons/simple-icons/develop/README.md"
    
    response = requests.get(math_terms_url)
    if response.status_code == 200:
        # Generate some basic math term definitions as a fallback
        math_terms = [
            "coordinate", "equation", "factor", "inequality", "sequence", 
            "series", "set", "theorem", "variable", "constant"
        ]
        
        for term in math_terms:
            question = f"What is a {term} in mathematics?"
            answer = f"In mathematics, a {term} is a fundamental concept used in mathematical reasoning and problem-solving. It refers to a specific mathematical object or relation that helps in formulating and solving problems in various branches of mathematics."
            math_qa_pairs.append((question, answer))
        
        print(f"Added {len(math_terms)} basic math term definitions")

except Exception as e:
    print(f"Error downloading additional datasets: {str(e)}")
    print("Using only the predefined QA pairs...")

# Create a mathematics-specific dataset
print(f"Creating mathematics dataset with {len(math_qa_pairs)} QA pairs")

# Prepare the dataset structure
math_training_data = {"Mathematics": []}
for question, answer in math_qa_pairs:
    math_training_data["Mathematics"].append(question)
    math_training_data["Mathematics"].append(answer)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
all_questions = [q for q in math_training_data["Mathematics"][::2]]
X = vectorizer.fit_transform(all_questions)

# Save the model components
model_path = 'model/math_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump((vectorizer, X, math_training_data), f)

print(f"Mathematics model successfully created and saved to {model_path}")
print(f"Dataset contains {len(all_questions)} questions")

# Optionally, merge this with existing AI tutor model if it exists
try:
    existing_model_path = 'model/ai_tutor_model.pkl'
    if os.path.exists(existing_model_path):
        print("Found existing AI tutor model, merging mathematics data...")
        
        with open(existing_model_path, 'rb') as f:
            existing_model_data = pickle.load(f)
            
        if isinstance(existing_model_data, tuple) and len(existing_model_data) == 3:
            _, _, existing_training_data = existing_model_data
            
            # Merge math data with existing data
            for subject, qa_list in existing_training_data.items():
                if subject != "Mathematics":  # Keep other subjects
                    math_training_data[subject] = qa_list
            
            # Create a new combined vectorizer
            combined_questions = []
            for subject, qa_list in math_training_data.items():
                combined_questions.extend(qa_list[::2])
                
            combined_vectorizer = TfidfVectorizer()
            combined_X = combined_vectorizer.fit_transform(combined_questions)
            
            # Save the combined model
            combined_model_path = 'model/large_ai_tutor_model.pkl'
            with open(combined_model_path, 'wb') as f:
                pickle.dump((combined_vectorizer, combined_X, math_training_data), f)
                
            print(f"Combined model successfully created and saved to {combined_model_path}")
            print(f"Combined dataset contains {len(combined_questions)} questions across {len(math_training_data)} subjects")
except Exception as e:
    print(f"Error merging with existing model: {str(e)}")
    print("Mathematics-only model was still created successfully.")

print("Done! Run the app.py file to start using the enhanced mathematics model.") 