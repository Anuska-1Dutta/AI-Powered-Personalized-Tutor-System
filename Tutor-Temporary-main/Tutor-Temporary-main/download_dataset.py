import requests
import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Create necessary directories
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Downloading educational QA datasets...")

# Function to download and parse datasets
def download_datasets():
    """Download educational datasets from GitHub"""
    # Educational QA dataset URLs - these are small datasets available without authentication
    math_dataset_url = "https://raw.githubusercontent.com/cognitivefactory/courseware-nlp-training/main/data/datasets/math-qa-sample.json"
    science_dataset_url = "https://raw.githubusercontent.com/cognitivefactory/courseware-nlp-training/main/data/datasets/science-qa-sample.json"
    
    # Create a combined training dataset
    training_data = {
        "Mathematics": [
            "What is a quadratic equation?", "A quadratic equation is a second-degree polynomial equation in the form ax² + bx + c = 0, where a, b, and c are constants and a ≠ 0. These equations are fundamental in algebra and are used to model many real-world scenarios. Quadratic equations can have zero, one, or two real solutions, and can be solved using factoring, completing the square, or the quadratic formula. The graph of a quadratic equation is a parabola, which can open upward or downward depending on the sign of the 'a' coefficient.",
            "How do I solve quadratic equations?", "To solve a quadratic equation ax² + bx + c = 0, you can use several methods:\n\n1. The Quadratic Formula: x = (-b ± √(b² - 4ac)) / 2a\n\n2. Factoring: If the equation can be written as (x+p)(x+q)=0, then x=-p or x=-q\n\n3. Completing the Square: Rearrange the equation to isolate all terms with variables on one side, then add terms to create a perfect square trinomial\n\n4. Graphical Method: Plot the quadratic function and find where it intersects the x-axis\n\nThe discriminant (b² - 4ac) tells you how many real solutions exist: if positive, two solutions; if zero, one solution; if negative, no real solutions.",
            "What is the Pythagorean theorem?", "The Pythagorean theorem is a fundamental principle in Euclidean geometry that states that in a right-angled triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the other two sides. Expressed mathematically: a² + b² = c², where c is the hypotenuse and a and b are the other two sides. This theorem is named after the ancient Greek mathematician Pythagoras, though it was known in various forms by other civilizations. It has countless applications in mathematics, physics, engineering, architecture, and navigation, and forms the basis for the distance formula in coordinate geometry.",
            "What is calculus?", "Calculus is a branch of mathematics focused on the study of change and motion. It consists of two major branches: differential calculus and integral calculus. Differential calculus examines rates of change and slopes of curves, introducing the concept of the derivative as an instantaneous rate of change. Integral calculus studies the accumulation of quantities and areas under or between curves, using integrals to calculate these quantities. These two branches are connected by the Fundamental Theorem of Calculus. Calculus is essential for modeling dynamic systems in physics, engineering, economics, statistics, and many other scientific disciplines. It was developed independently by Sir Isaac Newton and Gottfried Wilhelm Leibniz in the late 17th century.",
            "What is algebra?", "Algebra is a branch of mathematics dealing with symbols and the rules for manipulating these symbols to represent relationships between quantities and solve equations. It forms the foundation for advanced mathematical study and has applications in virtually every field that uses mathematics. Algebra introduces the concept of variables (symbols that represent unknown values) and equips us with methods to solve equations, inequalities, and systems of equations. Key algebraic concepts include expressions, equations, functions, polynomials, and matrices. The history of algebra dates back to ancient civilizations, with significant contributions from Persian mathematician al-Khwarizmi, whose work 'Al-jabr' gave algebra its name. Modern algebra has expanded to include abstract algebra, linear algebra, and many other specialized fields.",
            "What is geometry?", "Geometry is a branch of mathematics concerned with questions of shape, size, relative position of figures, and the properties of space. It began as a practical way for people to measure land, but evolved into a sophisticated field studying points, lines, angles, surfaces, solids, and higher-dimensional analogs. Euclidean geometry, based on Euclid's axioms and postulates, dominated for over 2000 years until the development of non-Euclidean geometries in the 19th century. Modern geometry includes projective geometry, differential geometry, topology, algebraic geometry, and computational geometry. Geometry connects to other mathematical fields like calculus, number theory, and provides essential tools for physics, engineering, computer graphics, architecture, and art. The subject combines visual intuition with rigorous logical reasoning, making it both practical and intellectually profound.",
            "What are derivatives in calculus?", "A derivative in calculus measures the sensitivity to change of a function's output with respect to its input. It represents the instantaneous rate of change or slope of a function at a specific point. Geometrically, the derivative gives the slope of the tangent line to the function's graph at that point. Derivatives are calculated using various rules such as the power rule, product rule, quotient rule, and chain rule. Derivatives are fundamental in understanding motion, optimization problems, approximation methods, and differential equations. Higher-order derivatives measure how the rate of change itself changes. Applications include velocity and acceleration calculations in physics, marginal analysis in economics, and optimization in engineering design. The process of finding derivatives is called differentiation, which forms one of the two main branches of calculus alongside integration.",
            "What is integration in calculus?", "Integration is a fundamental concept in calculus that represents the process of finding the accumulated effect of a continuous function. It is the reverse operation of differentiation. There are two main types of integrals: definite and indefinite. Indefinite integrals (or antiderivatives) give a family of functions whose derivative is the integrand. Definite integrals calculate the signed area between a function and the x-axis over a specific interval, represented as ∫[a,b] f(x)dx. Integration techniques include substitution, integration by parts, partial fractions, and trigonometric substitution. The Fundamental Theorem of Calculus connects differentiation and integration, showing that a definite integral can be evaluated by finding the antiderivative and evaluating it at the boundaries. Applications of integration include calculating areas, volumes, centers of mass, work, fluid pressure, and solving differential equations. Integration is used extensively in physics, engineering, economics, statistics, and many other fields of science.",
            "What is a limit in calculus?", "A limit in calculus is the value that a function approaches as the input approaches some value. Limits form the foundation of calculus and mathematical analysis, enabling the precise definition of continuity, derivatives, and integrals. The notation lim(x→a) f(x) = L means that as x gets arbitrarily close to a (but not equal to a), f(x) gets arbitrarily close to L. Limits can be evaluated using direct substitution when functions are continuous at the point in question. When direct substitution yields an indeterminate form (like 0/0 or ∞/∞), algebraic manipulation, factoring, rationalization, L'Hôpital's rule, or other techniques may be applied. Limits can exist even when a function is undefined at a point, and one-sided limits examine function behavior from only one direction. Limits at infinity describe a function's end behavior as its input grows without bound. The concept of limits enables mathematicians and scientists to analyze behavior near discontinuities, asymptotes, and other critical points in functions.",
            "How do I find the area of a circle?", "The area of a circle is calculated using the formula A = πr², where r is the radius of the circle. This formula can be derived by considering a circle as the limit of a regular polygon with an infinite number of sides, or through calculus using integration. The constant π (pi) is approximately 3.14159, but using the exact value π in formulas gives precise results. For practical calculations, if you know the diameter (d) instead of the radius, you can use A = π(d/2)², since r = d/2. If you know the circumference (C), you can use A = C²/4π, since C = 2πr. When measuring real-world circles, be careful to use consistent units - the area will be in square units (square centimeters, square meters, etc.). The area of a circle increases as the square of its radius, so doubling the radius quadruples the area."
        ],
        "Science": [
            "What is biology?", "Biology is the scientific study of life and living organisms. It examines the structure, function, growth, origin, evolution, and distribution of living things. Biology is divided into various specialized fields such as anatomy, physiology, botany, zoology, microbiology, genetics, ecology, and more. Each field focuses on different aspects of life, from molecular processes within cells to interactions between entire ecosystems. The core principles of biology include cell theory (all living things are made of cells), evolution (populations change over time through natural selection), genetics (traits are passed from parents to offspring through genes), homeostasis (organisms maintain internal stability), and energy processing (organisms require energy for survival). Understanding biology is essential for advances in medicine, agriculture, environmental conservation, and biotechnology.",
            "What is an amoeba?", "An amoeba is a type of single-celled organism (protozoan) that belongs to the phylum Sarcodina. Characterized by their ability to change shape, amoebas move by extending temporary foot-like projections called pseudopodia ('false feet'). They are typically found in freshwater environments such as ponds, lakes, and slow-moving streams, though some species live in soil or as parasites in animals. Amoebas feed by engulfing food particles through phagocytosis, where the cell membrane surrounds the food and brings it inside the cell within a food vacuole. They reproduce asexually through binary fission, where one cell divides into two identical daughter cells. While most amoebas are harmless, certain species like Entamoeba histolytica can cause diseases such as amoebic dysentery in humans. Amoebas are studied extensively in biology as examples of simple cellular organisms and are important in understanding cellular processes and evolution.",
            "What is chemistry?", "Chemistry is the scientific discipline that studies the composition, structure, properties, and changes of matter. It focuses on atoms, ions, and molecules which make up substances and how they interact with each other. Chemistry encompasses understanding the behavior of particles at the atomic and molecular levels to explain observations at the macroscopic level. The five main branches of chemistry are analytical chemistry, biochemistry, inorganic chemistry, organic chemistry, and physical chemistry. Some of the fundamental concepts in chemistry include the periodic table of elements, chemical bonds, reactions and equations, stoichiometry, and thermodynamics. Chemistry is vital to many aspects of modern life, from developing new materials and medicines to addressing environmental challenges. The systematic study of chemistry began with the development of the modern scientific method and has evolved through significant contributions from scientists like Antoine Lavoisier, John Dalton, Dmitri Mendeleev, and Marie Curie.",
            "What is Newton's first law?", "Newton's first law of motion, also known as the law of inertia, states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force. This principle directly challenges the previously held Aristotelian view that objects naturally seek rest. It explains why objects in space continue moving indefinitely, why passengers feel pushed back when a car accelerates, and why seat belts are essential safety devices. The law highlights the property of inertia - the resistance of any physical object to change its state of motion. Objects with greater mass have greater inertia. The first law is foundational to classical mechanics and leads logically to Newton's other laws. It reflects the fundamental symmetry of the universe and the conservation of momentum, setting the framework for understanding equilibrium and non-equilibrium states in physics.",
            "What is photosynthesis?", "Photosynthesis is the process by which green plants, algae, and certain bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars. This vital process takes place primarily in the chloroplasts of plant cells, particularly in the leaves. The overall equation for photosynthesis is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. The process occurs in two main stages: light-dependent reactions, which capture energy from sunlight and convert it into ATP and NADPH, and the Calvin cycle (light-independent reactions), which uses this energy to build sugar molecules from carbon dioxide. Photosynthesis is essential for life on Earth as it provides the oxygen we breathe, the food we eat (directly or indirectly), and helps regulate the planet's carbon cycle and climate. The efficiency of photosynthesis is affected by light intensity, carbon dioxide concentration, temperature, and water availability.",
            "What is the periodic table?", "The periodic table is a tabular arrangement of chemical elements, organized on the basis of their atomic numbers and chemical properties.",
            "What is DNA?", "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions.",
            "What are the states of matter?", "The main states of matter are solid, liquid, gas, and plasma. Each state has different properties based on the arrangement and movement of particles.",
            "What is the theory of evolution?", "The theory of evolution by natural selection explains how species change over time through variations that increase the individual's ability to survive and reproduce.",
            "What is gravity?", "Gravity is the force by which objects with mass attract one another. On Earth, gravity gives weight to physical objects and causes objects to fall toward the ground.",
            "What are chemical reactions?", "Chemical reactions are processes that lead to the transformation of one set of chemical substances to another, involving the breaking and formation of chemical bonds.",
            "What is the scientific method?", "The scientific method is a systematic approach to investigating phenomena, acquiring new knowledge, or correcting and integrating previous knowledge through observation, hypothesis, experimentation, and conclusion.",
            "What is cellular respiration?", "Cellular respiration is the process by which cells convert nutrients into ATP, water, and carbon dioxide. It's essentially the opposite of photosynthesis."
        ],
        "History": [
            "When did World War II start?", "World War II began in Europe on September 1, 1939, when Germany invaded Poland.",
            "Who was Alexander the Great?", "Alexander the Great was a king of the ancient Greek kingdom of Macedon and a member of the Argead dynasty, who conquered most of the known world of his time.",
            "What was the Renaissance?", "The Renaissance was a period of European cultural, artistic, political, scientific, and intellectual 'rebirth' following the Middle Ages, spanning the 14th to 17th centuries.",
            "What was the Industrial Revolution?", "The Industrial Revolution was a period from the 18th to 19th centuries where major changes in agriculture, manufacturing, mining, and transportation had a profound effect on economic and social conditions.",
            "Who was Cleopatra?", "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt, known for her relationships with Julius Caesar and Mark Antony.",
            "What was the Cold War?", "The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their allies from the mid-1940s until the early 1990s.",
            "What was the French Revolution?", "The French Revolution was a period of radical social and political upheaval in France from 1789 to 1799 that had a major impact on French and modern history.",
            "Who was Genghis Khan?", "Genghis Khan was the founder and first Great Khan of the Mongol Empire, which became the largest contiguous empire in history after his death.",
            "What was the Silk Road?", "The Silk Road was a network of trade routes connecting the East and West, facilitating trade and cultural exchange between ancient civilizations.",
            "What were the Crusades?", "The Crusades were a series of religious wars sanctioned by the Latin Church in the medieval period, specifically aimed at recovering the Holy Land from Islamic rule.",
            "Who was Abraham Lincoln?", "Abraham Lincoln was the 16th President of the United States, serving from March 1861 until his assassination in April 1865. Lincoln led the United States through the American Civil War, preserving the Union, abolishing slavery, strengthening the federal government, and modernizing the U.S. economy. He is remembered for his character, his speeches and letters, and for issuing the Emancipation Proclamation (1863) that began the process of ending slavery. His Gettysburg Address of 1863 became an iconic statement of America's dedication to the principles of nationalism, republicanism, equal rights, liberty, and democracy. Lincoln is consistently ranked by scholars and the public as one of the greatest U.S. presidents. His assassination made him a martyr for the ideals of national unity and equality.",
            "Who was George Washington?", "George Washington was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed as the commander of the Continental Army in 1775, Washington led Patriot forces to victory in the American Revolutionary War and served as the president of the Constitutional Convention of 1787, which created the U.S. Constitution and the federal government. Washington has been called the 'Father of the Country' for his manifold leadership in the nation's founding. He set many precedents for the presidency, including the title 'Mr. President', and established traditions like delivering an inaugural address and forming a cabinet. After retiring from the presidency, Washington returned to Mount Vernon, where he died in 1799. He was eulogized as 'first in war, first in peace, and first in the hearts of his countrymen'.",
            "Who was Winston Churchill?", "Sir Winston Leonard Spencer Churchill was a British statesman, soldier, and writer who served as Prime Minister of the United Kingdom from 1940 to 1945, during the Second World War, and again from 1951 to 1955. As Prime Minister, Churchill led Britain to victory over Nazi Germany. He was a noted statesman, orator, and strategist, and he was also an officer in the British Army and a writer who won the Nobel Prize in Literature. Churchill is widely regarded as one of the greatest wartime leaders of the 20th century. He was also a historian, a writer, and an artist. He is the only British Prime Minister to have received the Nobel Prize in Literature, and was also the first person to be made an honorary citizen of the United States."
        ],
        "Programming": [
            "What is Python?", "Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple programming paradigms and has a comprehensive standard library.",
            "How do I create a function in JavaScript?", "In JavaScript, you can create a function using the function keyword: function myFunction() { ... }, arrow functions: const myFunction = () => { ... }, or function expressions: const myFunction = function() { ... }",
            "What is object-oriented programming?", "Object-oriented programming (OOP) is a programming paradigm based on the concept of 'objects', which can contain data (attributes) and code (methods). The four main principles are encapsulation, abstraction, inheritance, and polymorphism.",
            "What is a variable in programming?", "A variable is a storage location paired with an associated name, which contains a value. Variables are used to store data that can be referenced and manipulated in a program.",
            "What is a loop in programming?", "A loop is a programming construct that repeats a sequence of instructions until a specific condition is met. Common types include for loops, while loops, and do-while loops.",
            "What is an array?", "An array is a data structure consisting of a collection of elements, each identified by an index. Arrays store multiple values in a single variable and provide ordered storage of items.",
            "What is a function in programming?", "A function is a block of organized, reusable code that performs a specific task. Functions help in breaking down complex problems into simpler pieces and improve code reusability.",
            "What is a database?", "A database is an organized collection of data stored and accessed electronically. Databases are designed to offer efficient retrieval, insertion, and deletion of data.",
            "What is an API?", "An Application Programming Interface (API) is a set of rules that allows different software applications to communicate with each other. APIs define the methods and data formats that programs can use to communicate.",
            "What is version control?", "Version control is a system that records changes to a file or set of files over time so you can recall specific versions later. It helps in tracking history, collaboration, and managing different versions of projects."
        ]
    }
    
    # Try to download additional data from URLs
    try:
        print("Attempting to download additional educational datasets...")
        
        # Math dataset
        response = requests.get(math_dataset_url)
        if response.status_code == 200:
            math_data = response.json()
            # Process the data and add to training data
            for item in math_data[:20]:  # Expanded to 20 items
                if 'question' in item and 'answer' in item:
                    training_data["Mathematics"].append(item['question'])
                    training_data["Mathematics"].append(item['answer'])
            print(f"Added {min(20, len(math_data))} mathematics Q&A pairs from online dataset")
        
        # Science dataset
        response = requests.get(science_dataset_url)
        if response.status_code == 200:
            science_data = response.json()
            # Process the data and add to training data
            for item in science_data[:20]:  # Expanded to 20 items
                if 'question' in item and 'answer' in item:
                    training_data["Science"].append(item['question'])
                    training_data["Science"].append(item['answer'])
            print(f"Added {min(20, len(science_data))} science Q&A pairs from online dataset")
            
            # Try to fetch additional biology Q&A
            bio_dataset_url = "https://raw.githubusercontent.com/cognitivefactory/courseware-nlp-training/main/data/datasets/bio-qa-sample.json"
            try:
                bio_response = requests.get(bio_dataset_url)
                if bio_response.status_code == 200:
                    bio_data = bio_response.json()
                    for item in bio_data[:15]:
                        if 'question' in item and 'answer' in item:
                            training_data["Science"].append(item['question'])
                            training_data["Science"].append(item['answer'])
                    print(f"Added biology Q&A pairs from online dataset")
            except:
                print("Could not retrieve additional biology dataset")
    
    except Exception as e:
        print(f"Error downloading additional datasets: {str(e)}")
        print("Using only the predefined training data...")
    
    # Count and display statistics
    total_qa_pairs = sum(len(training_data[subject])//2 for subject in training_data)
    print(f"Total Q&A pairs in dataset: {total_qa_pairs}")
    for subject in training_data:
        print(f"  - {subject}: {len(training_data[subject])//2} Q&A pairs")
        
    return training_data

# Main function to create the model
def create_model():
    print("Creating AI Tutor model...")
    
    # Get training data
    training_data = download_datasets()
    
    try:
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        all_questions = [q for subj in training_data for q in training_data[subj][::2]]
        X = vectorizer.fit_transform(all_questions)
        
        # Save the model components
        model_path = 'model/ai_tutor_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump((vectorizer, X, training_data), f)
        
        print(f"Model successfully created and saved to {model_path}")
        print(f"Dataset contains {len(all_questions)} questions across {len(training_data)} subjects")
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        print("Creating basic model instead...")
        
        # Import BasicAITutor from app.py
        from app import BasicAITutor
        
        # Create and save the basic model
        basic_model = BasicAITutor()
        with open('model/ai_tutor_model.pkl', 'wb') as f:
            pickle.dump(basic_model, f)
        
        print("Basic model created successfully")

if __name__ == "__main__":
    create_model()
    print("Done! Run the app.py file to start the AI Tutor with the enhanced model.") 