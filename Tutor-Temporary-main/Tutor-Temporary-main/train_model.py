import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

print("Training AI Tutor model...")

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Expanded training data for different subjects
# In a real application, you would have much more comprehensive training data
training_data = [
    # Mathematics
    ("How do I solve quadratic equations?", "To solve a quadratic equation ax² + bx + c = 0, you can use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a. Start by identifying the values of a, b, and c, then substitute them into the formula."),
    ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse equals the sum of squares of the other two sides. If a and b are the legs and c is the hypotenuse, then a² + b² = c²."),
    ("How do I find the derivative of a function?", "To find the derivative of a function, you apply the differentiation rules. For example, the derivative of x^n is n*x^(n-1). For more complex functions, you can use chain rule, product rule, or quotient rule depending on the function structure."),
    ("What is factoring in algebra?", "Factoring is the process of finding expressions that multiply together to give the original expression. For example, factoring x² + 5x + 6 gives (x + 2)(x + 3). It's essentially the reverse of multiplication."),
    ("How do I find the area of a circle?", "The area of a circle is calculated using the formula A = πr², where r is the radius of the circle and π (pi) is approximately 3.14159."),
    ("Can you explain logarithms?", "Logarithms are the inverse of exponentiation. If a^x = b, then log_a(b) = x. The logarithm tells you what power you need to raise the base to get the number. Common logarithms use base 10, while natural logarithms use base e (≈ 2.71828)."),
    ("What are trigonometric functions?", "Trigonometric functions relate the angles of a triangle to the lengths of its sides. The main functions are sine (sin), cosine (cos), and tangent (tan). They are fundamental in studying periodic phenomena and wave motions."),
    
    # Science
    ("Explain Newton's laws of motion", "Newton's three laws of motion are: 1) An object at rest stays at rest, and an object in motion stays in motion unless acted upon by an external force. 2) Force equals mass times acceleration (F=ma). 3) For every action, there is an equal and opposite reaction."),
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water. It converts light energy into chemical energy, producing oxygen as a byproduct."),
    ("How does DNA replication work?", "DNA replication is the process where DNA makes a copy of itself during cell division. It follows a semi-conservative model where each strand of the original DNA serves as a template for the new strand, resulting in two identical DNA molecules."),
    ("What is the periodic table?", "The periodic table is a tabular arrangement of chemical elements, organized by their atomic number, electron configuration, and chemical properties. Elements with similar properties are placed in the same column (group), allowing for predictions about chemical behavior."),
    ("How do black holes form?", "Black holes form when very massive stars die. After a supernova explosion, the core of the star collapses under its own gravity. If the core's mass is more than about 3 times the sun's mass, the collapse continues until it forms a black hole with gravity so strong that not even light can escape."),
    ("What is the theory of evolution?", "The theory of evolution by natural selection, proposed by Charles Darwin, explains how species change over time. It states that organisms with traits better suited to their environment are more likely to survive and reproduce, passing those traits to offspring."),
    ("Explain the water cycle", "The water cycle is the continuous movement of water within Earth and its atmosphere. It includes processes like evaporation (water turning to vapor), condensation (vapor forming clouds), precipitation (rain, snow), and collection (water gathering in oceans, lakes, and groundwater)."),
    
    # History
    ("What caused World War I?", "World War I was triggered by the assassination of Archduke Franz Ferdinand of Austria in June 1914, but underlying causes included militarism, alliances, imperialism, and nationalism across European powers, creating tensions that erupted into global conflict."),
    ("Who was Cleopatra?", "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt. She was known for her intelligence, political acumen, and relationships with Julius Caesar and Mark Antony. Her reign marked the end of the Hellenistic period and the beginning of Roman Egypt."),
    ("What was the Renaissance?", "The Renaissance was a period of European history from the 14th to the 17th century, marking the transition from the Middle Ages to modernity. It began in Italy and was characterized by renewed interest in classical learning, arts, and scientific discovery."),
    ("When did the Industrial Revolution occur?", "The Industrial Revolution began in Britain in the late 18th century (around 1760) and spread to Europe and North America by the early 19th century. It marked a major shift from manual production to machine manufacturing, transforming economies and societies."),
    ("Who was Genghis Khan?", "Genghis Khan was the founder and first Great Khan of the Mongol Empire, which became the largest contiguous empire in history after his death. Born around 1162, he united many nomadic tribes and conquered vast territories across Asia and Eastern Europe through military campaigns."),
    ("What was the Cold War?", "The Cold War was a period of geopolitical tension between the United States and the Soviet Union, along with their respective allies, from approximately 1947 to 1991. It was characterized by political and military tensions, proxy wars, and competition in areas like nuclear arms and space exploration."),
    ("Tell me about Ancient Greece", "Ancient Greece was a civilization that existed from the 8th century BC to the 6th century AD. It laid foundations for modern Western democracy, philosophy, arts, and sciences. City-states like Athens and Sparta had different political systems, while thinkers like Socrates, Plato, and Aristotle shaped philosophical thought."),
    
    # Programming
    ("How do I write a for loop in Python?", "In Python, you can write a for loop using the syntax: 'for variable in iterable:' followed by indented code to execute. For example, 'for i in range(5):' will loop with i values from 0 to 4."),
    ("What is object-oriented programming?", "Object-oriented programming (OOP) is a programming paradigm based on the concept of 'objects' that contain data and code. Objects can inherit properties and behaviors from other objects. The four main principles of OOP are encapsulation, abstraction, inheritance, and polymorphism."),
    ("How do I handle exceptions in Python?", "In Python, you can handle exceptions using try-except blocks. Place the code that might raise an exception in the try block, and the code to handle the exception in the except block. You can also use finally for code that should run regardless of whether an exception occurred."),
    ("What is a function in programming?", "A function in programming is a reusable block of code designed to perform a specific task. Functions accept inputs (parameters), process them, and return outputs. They help organize code, make it more readable, and promote reusability."),
    ("Explain what an API is", "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other. APIs define the methods and data formats that applications can use to request and exchange information."),
    ("What is a database?", "A database is an organized collection of structured information or data, typically stored electronically in a computer system. Databases are managed by database management systems (DBMS) and can be relational (like SQL databases) or non-relational (NoSQL)."),
    ("How do I create a website?", "Creating a website involves several steps: 1) Plan your site's purpose and content. 2) Choose a domain name and hosting service. 3) Design your site's layout and appearance. 4) Develop using HTML, CSS, and JavaScript or use a website builder. 5) Test functionality across different browsers and devices. 6) Publish your site."),
    
    # Additional examples for better coverage
    ("What is calculus?", "Calculus is a branch of mathematics focused on the study of rates of change and accumulation. It consists of two main branches: differential calculus (concerning rates of change and slopes of curves) and integral calculus (concerning accumulation of quantities and areas under curves)."),
    ("What are prime numbers?", "Prime numbers are natural numbers greater than 1 that cannot be formed by multiplying two smaller natural numbers. A natural number greater than 1 that is not prime is called a composite number. For example, 2, 3, 5, 7, 11, and 13 are prime numbers."),
    ("How does the immune system work?", "The immune system is a complex network of cells, tissues, and organs that defend the body against infections. It identifies foreign substances (antigens) and produces antibodies to neutralize them. It consists of innate immunity (the body's first line of defense) and adaptive immunity (which learns and remembers specific pathogens)."),
    ("What is climate change?", "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels which increases heat-trapping greenhouse gases in Earth's atmosphere."),
    ("Who was Albert Einstein?", "Albert Einstein (1879-1955) was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He is best known to the general public for his mass–energy equivalence formula E = mc²."),
    ("What is artificial intelligence?", "Artificial intelligence (AI) is the simulation of human intelligence in machines programmed to think and learn like humans. It encompasses various techniques including machine learning, natural language processing, computer vision, and robotics to enable systems to perform tasks that typically require human intelligence."),
]

# Extract questions and answers
X = [item[0] for item in training_data]
y = [item[1] for item in training_data]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Random Forest
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")

# Save the model
with open('model/ai_tutor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to model/ai_tutor_model.pkl")

# Testing the model with a few queries
test_queries = [
    "How do I solve for x in a quadratic equation?",
    "Can you explain how DNA works?",
    "What was the cause of World War II?",
    "How do I create a function in Python?"
]

print("\nTesting the model with sample queries:")
for query in test_queries:
    prediction = model.predict([query])[0]
    print(f"\nQuery: {query}")
    print(f"Response: {prediction[:100]}...")  # Show just the beginning of the response 