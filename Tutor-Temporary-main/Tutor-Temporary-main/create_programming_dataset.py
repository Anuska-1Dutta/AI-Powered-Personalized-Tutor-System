import os
import pickle
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

print("Creating enhanced programming dataset...")

# Create necessary directories
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Python programming QA pairs
python_qa = [
    ("What is Python?", "Python is a high-level, interpreted programming language known for its readability and simplicity. Created by Guido van Rossum and released in 1991, Python emphasizes code readability with significant whitespace. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python has a comprehensive standard library and a large ecosystem of third-party packages."),
    ("What are Python lists?", "In Python, lists are ordered, mutable collections of items that can be of different types. Lists are defined using square brackets, like my_list = [1, 2, 'three', 4.0]. They support operations like indexing (my_list[0]), slicing (my_list[1:3]), appending (my_list.append(5)), extending (my_list.extend([6, 7])), and other methods for manipulation. Lists are one of Python's most versatile data structures."),
    ("What are Python dictionaries?", "Dictionaries in Python are unordered collections of key-value pairs. They are defined using curly braces with key-value pairs separated by colons, like my_dict = {'name': 'John', 'age': 30}. Dictionaries provide fast lookups based on keys and support operations like adding items (my_dict['email'] = 'john@example.com'), removing items (del my_dict['age']), and checking for keys (if 'name' in my_dict:). They are very useful for representing structured data."),
    ("What are Python functions?", "Functions in Python are defined blocks of code that perform specific tasks and can be reused. They are defined using the 'def' keyword, like 'def greet(name): return f\"Hello, {name}!\"'. Functions can have parameters with default values, return values, and can be passed as arguments to other functions. Python also supports anonymous functions called lambda functions, like 'square = lambda x: x**2'."),
    ("What is object-oriented programming in Python?", "Object-oriented programming (OOP) in Python is a programming paradigm based on classes and objects. Classes are blueprints for creating objects, and objects are instances of classes. Python supports OOP concepts like inheritance, encapsulation, and polymorphism. Classes are defined using the 'class' keyword, and objects are created by calling the class. Python's OOP implementation is flexible and supports multiple inheritance."),
    ("What are Python modules and packages?", "Python modules are files containing Python code that can be imported and used in other Python programs. A module can define functions, classes, and variables. Packages are collections of related modules organized in directories with an __init__.py file. They provide a way to organize code hierarchically. The standard library offers many built-in modules like os, sys, math, and datetime. Third-party packages can be installed using package managers like pip."),
    ("What are Python decorators?", "Decorators in Python are functions that modify the behavior of other functions or methods. They use the @decorator syntax and are a form of metaprogramming. Decorators take a function as input, add some functionality, and return the modified function. Common uses include logging, authentication, timing functions, and adding methods to classes. Built-in decorators include @property, @classmethod, and @staticmethod."),
    ("What are Python comprehensions?", "Python comprehensions are concise ways to create collections like lists, dictionaries, and sets. List comprehensions create lists: [x**2 for x in range(10)]. Dictionary comprehensions create dictionaries: {x: x**2 for x in range(5)}. Set comprehensions create sets: {x%3 for x in range(10)}. Generator expressions, similar to list comprehensions but enclosed in parentheses, create generators: (x**2 for x in range(10)). These provide a more readable and often faster alternative to using loops."),
    ("What is PEP 8?", "PEP 8 is the style guide for Python code. It provides conventions for writing readable code, including rules for indentation (use 4 spaces), maximum line length (79 characters), naming conventions (lowercase_with_underscores for functions and variables, CamelCase for classes), imports organization, whitespace usage, and more. Following PEP 8 makes Python code more consistent and easier for others to read and maintain. Tools like pylint, flake8, and black can help enforce PEP 8 compliance."),
    ("What are Python virtual environments?", "Virtual environments in Python are isolated environments where you can install packages without affecting the global Python installation. They are created using the venv module (python -m venv myenv) or third-party tools like virtualenv. After activation, packages installed with pip will be local to that environment. Virtual environments help manage dependencies for different projects, avoiding version conflicts and making projects more reproducible."),
]

# Web development QA pairs
web_dev_qa = [
    ("What is HTML?", "HTML (HyperText Markup Language) is the standard markup language used to create web pages. It defines the structure and content of a webpage using elements represented by tags like <head>, <body>, <h1>, <p>, <div>, etc. HTML documents are interpreted by web browsers to display content. HTML5, the latest major version, introduced new semantic elements, form controls, and multimedia support. HTML works with CSS for styling and JavaScript for behavior."),
    ("What is CSS?", "CSS (Cascading Style Sheets) is a style sheet language used to describe the presentation of HTML documents. It controls layout, colors, fonts, and other visual aspects of web pages. CSS uses selectors to target HTML elements and declarations to specify their styles. It supports concepts like the box model, positioning, flexbox, grid layout, and responsive design through media queries. CSS3, the current standard, introduced animations, transitions, and other advanced features."),
    ("What is JavaScript?", "JavaScript is a high-level, interpreted programming language primarily used for client-side web development. It enables interactive web pages by allowing manipulation of the DOM (Document Object Model), handling events, making asynchronous requests (AJAX), and more. JavaScript follows the ECMAScript standard, with modern versions (ES6+) adding features like arrow functions, classes, promises, and modules. It can also be used server-side (Node.js) and for mobile and desktop app development."),
    ("What is React?", "React is a JavaScript library for building user interfaces developed by Facebook. It uses a component-based architecture where UIs are composed of reusable components. React uses a virtual DOM to optimize rendering performance. Key concepts include JSX (JavaScript XML syntax), props for passing data to components, state for managing component data, and hooks for adding state and lifecycle features to functional components. React is commonly used with state management libraries like Redux or Context API."),
    ("What is Node.js?", "Node.js is a JavaScript runtime built on Chrome's V8 engine that allows executing JavaScript code outside a web browser. It uses an event-driven, non-blocking I/O model, making it efficient for data-intensive applications. Node.js excels at building scalable network applications and is widely used for server-side web development. It comes with npm, the largest ecosystem of open-source libraries. Core modules include fs for file operations, http for creating web servers, and path for handling file paths."),
    ("What is RESTful API?", "A RESTful API (Representational State Transfer) is an architectural style for designing networked applications. It uses HTTP requests to perform CRUD operations (Create, Read, Update, Delete) on resources, which are identified by URLs. RESTful APIs typically return data in JSON or XML format. They are stateless, meaning each request contains all necessary information. Key principles include using appropriate HTTP methods (GET, POST, PUT, DELETE), clear resource naming, and HTTP status codes for responses."),
    ("What is responsive web design?", "Responsive web design is an approach to web design that makes web pages render well on all devices and screen sizes. It uses fluid grids, flexible images, and CSS media queries to adapt layouts to different viewports. Key techniques include percentage-based widths, max-width property, viewport meta tag, and breakpoints where designs change based on screen width. This approach eliminates the need for separate mobile sites and improves user experience across devices."),
    ("What is AJAX?", "AJAX (Asynchronous JavaScript and XML) is a technique that allows web pages to update content asynchronously by exchanging data with a server behind the scenes. This enables updating parts of a web page without reloading the entire page. Despite the name, modern AJAX typically uses JSON instead of XML. The fetch API or XMLHttpRequest object is used to send requests to the server. AJAX is fundamental to creating dynamic, interactive web applications with smooth user experiences."),
    ("What is webpack?", "Webpack is a static module bundler for modern JavaScript applications. It takes modules with dependencies and generates optimized bundles that browsers can understand. Webpack can process various asset types (JavaScript, CSS, images) using loaders, and transform them using plugins. Key concepts include entry points, output configuration, loaders, plugins, and code splitting. Webpack helps improve performance through minification, tree shaking (dead code elimination), and efficient loading strategies."),
    ("What is CORS?", "CORS (Cross-Origin Resource Sharing) is a security feature implemented by browsers that restricts web pages from making requests to a different domain than the one that served the page. It prevents potentially malicious websites from reading sensitive data from another site. CORS works through HTTP headers: the browser sends an Origin header with requests, and the server responds with Access-Control-Allow-Origin headers indicating which origins are permitted to access the resource. Preflight requests (OPTIONS) may be sent to check if the actual request is safe to send."),
]

# Data science QA pairs
data_science_qa = [
    ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It involves algorithms that improve performance on a task through experience. The three main types are supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through reward-based feedback). Applications include image recognition, recommendation systems, fraud detection, and natural language processing."),
    ("What is data preprocessing?", "Data preprocessing is the technique of preparing raw data for analysis. It includes cleaning (handling missing values, removing duplicates), transformation (normalization, standardization), feature selection/extraction, and dimensionality reduction. Preprocessing is crucial because real-world data is often incomplete, inconsistent, and contains errors. Effective preprocessing improves model accuracy, reduces training time, and helps avoid the 'garbage in, garbage out' problem in data analysis."),
    ("What is a neural network?", "A neural network is a computational model inspired by the human brain, consisting of connected nodes (neurons) organized in layers. It includes an input layer, one or more hidden layers, and an output layer. Each connection has a weight that adjusts during learning. Neural networks excel at finding patterns in complex data and are the foundation of deep learning. Applications include image and speech recognition, language translation, and playing complex games like Go."),
    ("What is deep learning?", "Deep learning is a subset of machine learning using neural networks with many layers (deep neural networks). These networks can automatically learn hierarchical features from data. Key architectures include Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) for sequential data, and Transformers for natural language processing. Deep learning has achieved breakthrough results in areas like computer vision, speech recognition, and natural language understanding."),
    ("What is a decision tree?", "A decision tree is a supervised learning algorithm that creates a flowchart-like structure for decision making. It splits data into branches based on feature values to minimize a cost function. Decision trees are interpretable, can handle both numerical and categorical data, and require minimal preprocessing. However, they can overfit without proper constraints. Random Forests and Gradient Boosted Trees are ensemble methods that combine multiple decision trees to improve performance."),
    ("What is logistic regression?", "Logistic regression is a statistical model for binary classification (predicting one of two outcomes). Despite its name, it's used for classification, not regression. It applies a logistic function to a linear combination of features to produce a probability between 0 and 1. The decision boundary is linear. Logistic regression is simple, interpretable, computationally efficient, and less prone to overfitting than complex models. It's widely used in medicine, marketing, and credit scoring."),
    ("What is reinforcement learning?", "Reinforcement learning is a machine learning paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative rewards. The agent learns through trial and error, trading off exploration (trying new actions) and exploitation (using known good actions). Key concepts include states, actions, rewards, and policies. Applications include game playing (AlphaGo), robotics, recommendation systems, and autonomous vehicles."),
    ("What is natural language processing?", "Natural language processing (NLP) is a field combining linguistics, computer science, and AI to enable computers to understand, interpret, and generate human language. It includes tasks like sentiment analysis, named entity recognition, machine translation, and question answering. Modern NLP relies heavily on deep learning, with transformer models like BERT and GPT achieving remarkable results. Applications include virtual assistants, language translation, text summarization, and content analysis."),
    ("What is cross-validation?", "Cross-validation is a technique for evaluating machine learning models by testing them on multiple subsets of the available data. In k-fold cross-validation, the data is divided into k subsets (folds); the model is trained on k-1 folds and tested on the remaining fold, rotating through all combinations. This method provides a more reliable performance estimate than a single train-test split, helping detect overfitting and making better use of limited data. Variations include stratified cross-validation and leave-one-out cross-validation."),
    ("What is data visualization?", "Data visualization is the graphical representation of information to identify patterns, trends, and insights. Common visualization types include bar charts, line graphs, scatter plots, histograms, heatmaps, and geographic maps. Effective visualizations follow principles like clarity, simplicity, and honest representation of data. Popular tools include Matplotlib, Seaborn, and Plotly in Python, ggplot2 in R, and applications like Tableau and Power BI. Good visualization helps in data exploration, communication of findings, and decision making."),
]

# Software engineering QA pairs
software_eng_qa = [
    ("What is version control?", "Version control is a system that records changes to files over time, allowing developers to recall specific versions and collaborate efficiently. It tracks modification history, facilitates collaboration among multiple developers, creates development branches, and enables reverting to previous states if needed. Git is the most popular distributed version control system, with platforms like GitHub, GitLab, and Bitbucket providing additional collaboration features. Version control is fundamental to modern software development practices."),
    ("What is Test-Driven Development?", "Test-Driven Development (TDD) is a software development approach where tests are written before the actual code. The process follows a red-green-refactor cycle: write a failing test, implement the minimum code to pass the test, then refactor while maintaining passing tests. TDD helps ensure code meets requirements, facilitates better design, creates a regression test suite, and simplifies debugging. It's particularly effective in projects with changing requirements and complex logic."),
    ("What is continuous integration?", "Continuous Integration (CI) is a development practice where developers frequently merge code changes into a central repository, after which automated builds and tests run. It aims to detect integration issues early, improve software quality, and reduce the time to validate and release new updates. CI typically involves a version control system, build automation, automated testing, and a CI server like Jenkins, GitHub Actions, or GitLab CI. When combined with Continuous Deployment, it forms a CI/CD pipeline."),
    ("What is agile software development?", "Agile software development is an approach emphasizing iterative development, collaboration, self-organizing teams, and adapting to change. It values individuals and interactions, working software, customer collaboration, and responding to change rather than following rigid plans. Popular agile methodologies include Scrum (with sprints, daily standups, and retrospectives) and Kanban (focusing on visualizing workflow and limiting work in progress). Agile aims to deliver working software frequently while maintaining flexibility."),
    ("What are design patterns?", "Design patterns are reusable solutions to common problems in software design. They represent best practices evolved over time by experienced developers. Some key patterns include: Singleton (ensuring a class has only one instance), Factory (creating objects without specifying exact class), Observer (maintaining a list of dependents to notify of state changes), Strategy (selecting an algorithm at runtime), and Decorator (adding responsibilities to objects dynamically). Design patterns improve code organization, reusability, and communication among developers."),
    ("What is software architecture?", "Software architecture is the high-level structure of a software system, defining its components, relationships, and properties. Common architectural patterns include: Microservices (independent, deployable services), Layered Architecture (organized in horizontal layers), Event-Driven Architecture (components communicating through events), and Model-View-Controller (separating data, user interface, and control logic). Good architecture facilitates scalability, maintainability, and adaptability to changing requirements."),
    ("What is DevOps?", "DevOps is a set of practices combining software development (Dev) and IT operations (Ops) to shorten the development lifecycle and deliver features continuously. Key principles include automation (CI/CD, infrastructure as code), collaboration between development and operations, measuring everything, and sharing responsibilities. DevOps tools include Git for version control, Jenkins for CI/CD, Docker for containerization, Kubernetes for orchestration, and Ansible for configuration management. The approach aims to increase deployment frequency, reduce failures, and improve recovery times."),
    ("What is clean code?", "Clean code is code that is easy to understand, change, and maintain. Characteristics include readability, simplicity, modular design, good naming, minimal duplication, proper handling of errors, and comprehensive tests. Clean code follows principles like DRY (Don't Repeat Yourself), SOLID (Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion), and KISS (Keep It Simple, Stupid). It results in fewer bugs, easier collaboration, faster development, and reduced technical debt."),
    ("What is refactoring?", "Refactoring is the process of restructuring existing code without changing its external behavior. It aims to improve code quality, readability, and maintainability by removing duplication, simplifying complex logic, and applying design patterns. Common refactoring techniques include extracting methods, renaming variables for clarity, simplifying conditional expressions, and organizing code into classes. Refactoring should be done with a safety net of tests and in small, incremental steps to avoid introducing bugs."),
    ("What is SQL?", "SQL (Structured Query Language) is a domain-specific language for managing data in relational database management systems. It allows creating, reading, updating, and deleting data (CRUD operations) through commands like CREATE TABLE, SELECT, INSERT, UPDATE, and DELETE. SQL also enables complex queries with JOINs, aggregations (SUM, COUNT), filtering (WHERE), sorting (ORDER BY), and grouping (GROUP BY). It's used across various database systems like MySQL, PostgreSQL, SQL Server, and Oracle, with minor syntax variations among them."),
]

# Combined programming QA pairs
programming_qa_pairs = python_qa + web_dev_qa + data_science_qa + software_eng_qa

# Try to download additional data from external sources
try:
    print("Attempting to download additional programming datasets...")
    
    # Try to fetch additional programming content
    programming_urls = [
        "https://raw.githubusercontent.com/karpathy/minGPT/master/README.md",
        "https://raw.githubusercontent.com/tensorflow/tensorflow/master/README.md"
    ]
    
    programming_examples = []
    for url in programming_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                content = response.text
                lines = content.split('\n')
                
                # Extract Q&A style content from README files
                current_question = None
                answer_lines = []
                
                for line in lines:
                    # Look for potential questions (headers)
                    if line.startswith('## ') or line.startswith('# '):
                        # If we already have a question, save it with its answer
                        if current_question and answer_lines:
                            answer_text = ' '.join(answer_lines)
                            if len(answer_text) > 50:  # Only use if answer has substantial content
                                programming_examples.append((current_question, answer_text))
                        
                        # Start a new potential Q&A pair
                        current_question = line.lstrip('#').strip()
                        answer_lines = []
                    elif current_question and line.strip() and not line.startswith('```'):
                        # Collect non-empty lines that aren't code blocks as part of the answer
                        answer_lines.append(line.strip())
                
                # Add the last Q&A pair if there is one
                if current_question and answer_lines:
                    answer_text = ' '.join(answer_lines)
                    if len(answer_text) > 50:
                        programming_examples.append((current_question, answer_text))
                
                print(f"Extracted {len(programming_examples)} Q&A pairs from {url}")
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    # Add some of the examples to our dataset (limit to avoid very long answers)
    for q, a in programming_examples[:5]:
        # Truncate very long answers
        if len(a) > 500:
            a = a[:500] + "..."
        programming_qa_pairs.append((q, a))
    
    # Add fallback programming topics as a backup
    programming_concepts = [
        "API", "framework", "IDE", "compiler", "interpreter", 
        "debugger", "algorithm", "data structure", "database", "git"
    ]
    
    for concept in programming_concepts:
        question = f"What is a {concept} in programming?"
        answer = f"In programming, a {concept} is a tool or concept that helps developers create, test, and maintain software. It's an essential part of modern software development practices and workflows."
        programming_qa_pairs.append((question, answer))
    
    print(f"Added {len(programming_concepts)} basic programming concept definitions")

except Exception as e:
    print(f"Error downloading additional datasets: {str(e)}")
    print("Using only the predefined QA pairs...")

# Create a programming-specific dataset
print(f"Creating programming dataset with {len(programming_qa_pairs)} QA pairs")

# Prepare the dataset structure
programming_training_data = {"Programming": []}
for question, answer in programming_qa_pairs:
    programming_training_data["Programming"].append(question)
    programming_training_data["Programming"].append(answer)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
all_questions = [q for q in programming_training_data["Programming"][::2]]
X = vectorizer.fit_transform(all_questions)

# Save the model components
model_path = 'model/programming_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump((vectorizer, X, programming_training_data), f)

print(f"Programming model successfully created and saved to {model_path}")
print(f"Dataset contains {len(all_questions)} questions")

# Merge with existing AI tutor model
try:
    existing_model_path = 'model/large_ai_tutor_model.pkl'
    if os.path.exists(existing_model_path):
        print("Found existing large AI tutor model, merging programming data...")
        
        with open(existing_model_path, 'rb') as f:
            existing_model_data = pickle.load(f)
            
        if isinstance(existing_model_data, tuple) and len(existing_model_data) == 3:
            combined_vectorizer, combined_X, combined_training_data = existing_model_data
            
            # Merge programming data with existing data
            combined_training_data["Programming"] = programming_training_data["Programming"]
            
            # Create a new combined vectorizer and feature matrix
            combined_questions = []
            for subject, qa_list in combined_training_data.items():
                combined_questions.extend(qa_list[::2])
                
            new_vectorizer = TfidfVectorizer()
            new_X = new_vectorizer.fit_transform(combined_questions)
            
            # Save the combined model
            with open(existing_model_path, 'wb') as f:
                pickle.dump((new_vectorizer, new_X, combined_training_data), f)
                
            print(f"Combined model successfully updated at {existing_model_path}")
            print(f"Combined dataset now contains {len(combined_questions)} questions across {len(combined_training_data)} subjects")
except Exception as e:
    print(f"Error merging with existing model: {str(e)}")
    print("Programming-only model was still created successfully.")

print("Done! Run the app.py file to start using the enhanced programming model.") 