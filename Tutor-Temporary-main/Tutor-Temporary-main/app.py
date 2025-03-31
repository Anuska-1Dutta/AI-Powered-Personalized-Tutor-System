import streamlit as st
import pickle
import os
import random
import logging
import traceback
from datetime import datetime
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import streamlit.components.v1 as components
import re

# Configure logging
logging.basicConfig(filename='logs/ai_tutor.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Direct debug logs to a file per session
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('logs', exist_ok=True)
debug_log_path = f'logs/ai_tutor_{current_time}.log'
debug_logger = logging.getLogger('debug')
debug_handler = logging.FileHandler(debug_log_path)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.DEBUG)

def debug_log(message):
    """Write debug messages to log file"""
    debug_logger.debug(message)

# Function to get progress data for a user
def get_progress(username):
    """Load progress data for the given username"""
    try:
        progress_file = f"data/user_progress_{username}.json"
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        debug_log(f"Error loading progress: {str(e)}")
        return {}

# Function to update progress data
def update_progress(username, subject, question, progress_value=5):
    """Update the progress data for a user with the new format."""
    try:
        # Load existing progress data
        progress_file = 'data/user_progress.pkl'
        
        # Check if file exists first
        if not os.path.exists(progress_file):
            debug_log("Creating new progress file")
            with open(progress_file, 'wb') as f:
                pickle.dump({}, f)
        
        # Load the data
        user_progress = {}
        with open(progress_file, 'rb') as f:
            user_progress = pickle.load(f)
        
        # Initialize user if not exists
        if username not in user_progress:
            user_progress[username] = {}
        
        # Initialize subject if not exists
        if subject not in user_progress[username]:
            user_progress[username][subject] = {
                "sessions": [],
                "last_session": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "questions_asked": 0,
                "mastery_level": 0
            }
        
        # Create a new session or add to the most recent one
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if there's a session in the last hour, otherwise create new one
        if user_progress[username][subject]["sessions"]:
            last_session = user_progress[username][subject]["sessions"][-1]
            last_timestamp = last_session.get("timestamp", "")
            
            # Parse the timestamp
            try:
                last_time = datetime.strptime(last_timestamp, "%Y-%m-%d %H:%M:%S")
                time_diff = datetime.now() - last_time
                
                # If less than an hour, add to existing session
                if time_diff.total_seconds() < 3600:
                    if "questions" not in last_session:
                        last_session["questions"] = []
                    last_session["questions"].append(question)
                else:
                    # Create new session
                    user_progress[username][subject]["sessions"].append({
                        "timestamp": current_time,
                        "questions": [question],
                        "duration": 0
                    })
            except Exception as e:
                debug_log(f"Error parsing timestamp: {str(e)}")
                # Create new session as fallback
                user_progress[username][subject]["sessions"].append({
                    "timestamp": current_time,
                    "questions": [question],
                    "duration": 0
                })
        else:
            # No sessions yet, create first one
            user_progress[username][subject]["sessions"].append({
                "timestamp": current_time,
                "questions": [question],
                "duration": 0
            })
        
        # Update last session time and questions count
        user_progress[username][subject]["last_session"] = current_time
        user_progress[username][subject]["questions_asked"] += 1
        
        # Update mastery level - simple algorithm based on number of questions
        questions_asked = user_progress[username][subject]["questions_asked"]
        mastery_thresholds = {10: 20, 25: 40, 50: 60, 100: 80, 200: 95}
        
        for threshold, level in mastery_thresholds.items():
            if questions_asked >= threshold:
                user_progress[username][subject]["mastery_level"] = level
        
        # Save updated progress
        with open(progress_file, 'wb') as f:
            pickle.dump(user_progress, f)
        
        debug_log(f"Progress saved for {username} in {subject} with question: {question[:30]}...")
        return True
    except Exception as e:
        debug_log(f"Error updating progress: {str(e)}")
        return False

# Add this function close to the top of the file, after other imports
def generate_ai_response(subject, question, model, vectorizer, questions, answers):
    """Generate AI response with improved context handling and caching prevention"""
    
    debug_log(f"Generating response for: '{question}' in subject: {subject}")
    
    # Clean and process the user question
    cleaned_question = question.strip().lower()
    
    # Generate a unique request ID to prevent response caching
    request_id = str(uuid.uuid4())[:8]
    debug_log(f"Request ID: {request_id}")
    
    # Check for empty question
    if not cleaned_question:
        debug_log("Empty question detected")
        return "Please ask me a question about " + subject
    
    try:
        # Convert the question to vector
        question_vector = vectorizer.transform([cleaned_question])
        
        # Get all training vectors for comparison
        train_vectors = vectorizer.transform(questions)
        
        # Calculate similarity scores between the question and training data
        similarity_scores = cosine_similarity(question_vector, train_vectors)[0]
        
        # Get the index with the highest similarity
        most_similar_idx = np.argmax(similarity_scores)
        similarity_value = similarity_scores[most_similar_idx]
        
        debug_log(f"Most similar question: '{questions[most_similar_idx]}' with score: {similarity_value}")
        
        # Check if there's a good match
        if similarity_value > 0.5:  # A reasonable threshold
            answer = answers[most_similar_idx]
            debug_log(f"Using matched answer: '{answer[:50]}...'")
            return answer
        else:
            # If no good match, generate a response based on the subject
            # This ensures we don't always return a default response
            
            debug_log("No good match found, using subject-specific response")
            
            # Subject-specific responses for low-match cases
            subject_responses = {
                "Mathematics": [
                    f"Request {request_id}: That's an interesting math question. While I don't have an exact match in my knowledge base, I can tell you that mathematics involves numbers, quantities, and logical reasoning. Could you provide more details or rephrase your question?",
                    f"Request {request_id}: In mathematics, we often approach problems step-by-step. Your question seems to be about a math concept I'm not fully trained on yet. Could you try a simpler version or a different topic in mathematics?",
                    f"Request {request_id}: Mathematics has many branches like algebra, geometry, calculus, and statistics. To better answer your question, could you specify which area of math you're asking about?"
                ],
                "Science": [
                    f"Request {request_id}: That's a fascinating science question. While I don't have a specific answer in my database, science is about understanding the natural world through observation and experimentation. Could you clarify or rephrase your question?",
                    f"Request {request_id}: Science includes fields like biology, chemistry, physics, and earth science. To give you a better answer, could you specify which scientific field your question relates to?",
                    f"Request {request_id}: Scientific knowledge evolves through research and discovery. Your question touches on concepts I don't have complete information about. Could you provide more context or ask about a related topic?"
                ],
                "History": [
                    f"Request {request_id}: History helps us understand the past and its influence on our present. While I don't have specific information about your question in my database, could you provide more details about the time period or region you're interested in?",
                    f"Request {request_id}: Historical events are connected through cause and effect relationships. To better answer your question about history, could you specify the era, civilization, or historical figure you're asking about?",
                    f"Request {request_id}: Understanding history requires examining multiple perspectives and sources. Your question seems to be about a historical topic I don't have complete information on. Could you try a different historical question?"
                ],
                "Programming": [
                    f"Request {request_id}: Programming involves writing instructions for computers to follow. While I don't have a specific answer to your programming question in my database, could you specify which programming language or concept you're asking about?",
                    f"Request {request_id}: In programming, problems can be solved in multiple ways. Your question seems to be about a programming concept I don't have complete information on. Could you provide code examples or more context?",
                    f"Request {request_id}: Programming languages have their own syntax and best practices. To better help with your programming question, could you specify if you're asking about algorithms, syntax, debugging, or another aspect of coding?"
                ]
            }
            
            # Default fallback if subject not in predefined list
            default_responses = [
                f"Request {request_id}: That's an interesting question. While I don't have a specific answer in my knowledge base, I'm designed to help with educational topics. Could you provide more details or try a different question?",
                f"Request {request_id}: I'm still learning about various educational topics. Your question seems to be about something I don't have complete information on yet. Could you try rephrasing or asking about something else?",
                f"Request {request_id}: I aim to provide helpful educational information. To better assist you, could you provide more context or details about what you're trying to learn?"
            ]
            
            # Get responses for the current subject or use default
            response_options = subject_responses.get(subject, default_responses)
            
            # Randomly select one to add variety
            import random
            response = random.choice(response_options)
            
            debug_log(f"Generated fallback response: '{response[:50]}...'")
            return response
    
    except Exception as e:
        debug_log(f"Error generating response: {str(e)}")
        return f"I encountered an error while processing your question (Error ID: {request_id}). Please try asking again or try a different question."

# Define the AITutor class properly
class AITutor:
    def __init__(self, vectorizer=None, X=None, training_data=None, model_path=None):
        try:
            debug_log("Initializing AITutor")
            if model_path:
                with open(model_path, 'rb') as f:
                    debug_log(f"Loading model from {model_path}")
                    data = pickle.load(f)
                    
                    # Check if new format model with all components
                    if isinstance(data, tuple) and len(data) == 3:
                        self.vectorizer, self.X, self.training_data = data
                        debug_log(f"Loaded model in expanded format with {sum(len(v)//2 for v in self.training_data.values() if isinstance(v, list))} QA pairs")
                    else:
                        # Old format - just Q&A pairs
                        self.vectorizer = TfidfVectorizer()
                        self.training_data = data
                        questions = [q for q, _ in data]
                        self.X = self.vectorizer.fit_transform(questions)
                        debug_log("Loaded model in legacy format")
            elif vectorizer is not None and X is not None and training_data is not None:
                self.vectorizer = vectorizer
                self.X = X
                self.training_data = training_data
                debug_log(f"Using provided vectorizer, X, and training_data with {sum(len(v)//2 for v in training_data.values() if isinstance(v, list))} QA pairs")
            else:
                debug_log("No model provided, creating comprehensive model with external datasets if available")
                # Try loading from external datasets first
                try:
                    large_dataset_path = 'model/large_ai_tutor_model.pkl'
                    if os.path.exists(large_dataset_path):
                        debug_log(f"Loading large dataset from {large_dataset_path}")
                        with open(large_dataset_path, 'rb') as f:
                            large_data = pickle.load(f)
                            if isinstance(large_data, tuple) and len(large_data) == 3:
                                self.vectorizer, self.X, self.training_data = large_data
                                debug_log(f"Loaded large dataset with {sum(len(v)//2 for v in self.training_data.values() if isinstance(v, list))} QA pairs")
                                
                                # Create subject lookup dictionary
                                self.training_data_dict = {}
                                for subject, qa_list in self.training_data.items():
                                    pairs = []
                                    for i in range(0, len(qa_list), 2):
                                        if i+1 < len(qa_list):
                                            pairs.append((qa_list[i], qa_list[i+1]))
                                    self.training_data_dict[subject] = pairs
                                
                                debug_log(f"Created structured QA dictionary with {sum(len(pairs) for pairs in self.training_data_dict.values())} total pairs")
                                return
                except Exception as e:
                    debug_log(f"Error loading large dataset: {str(e)}")
                
                # Create a minimal fallback model with comprehensive dataset
                self.vectorizer = TfidfVectorizer()
                
                # Comprehensive QA dataset for all subjects
                minimal_qa = [
                    # Mathematics
                    ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the other two sides. It is represented by the equation: a² + b² = c², where c is the length of the hypotenuse and a and b are the lengths of the other two sides."),
                    ("How do you solve a quadratic equation?", "Quadratic equations can be solved using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a, where ax² + bx + c = 0. Alternatively, you can solve by factoring, completing the square, or graphing, depending on the specific equation."),
                    ("What are matrices?", "Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are used in linear algebra for representing linear transformations and solving systems of linear equations."),
                    ("What is calculus?", "Calculus is a branch of mathematics that focuses on the study of continuous change. It has two main branches: differential calculus (concerning rates of change and slopes of curves) and integral calculus (concerning accumulation of quantities and areas under curves)."),
                    ("What is algebra?", "Algebra is a branch of mathematics that uses symbols and letters to represent numbers and quantities in formulas and equations. It introduces the concept of variables and provides tools for solving equations."),
                    ("What are equations?", "Equations are mathematical statements that assert the equality of two expressions. They typically contain variables and state that the expressions on either side of the equals sign have the same value."),
                    ("What is trigonometry?", "Trigonometry is a branch of mathematics that studies the relationships between the sides and angles of triangles. It defines trigonometric functions such as sine, cosine, and tangent, which relate the angles of a triangle to the lengths of its sides."),
                    ("What is geometry?", "Geometry is a branch of mathematics concerned with questions of shape, size, relative position of figures, and the properties of space. It includes the study of points, lines, angles, surfaces, and solids."),
                    
                    # Science
                    ("What is photosynthesis?", "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars. Plants take in carbon dioxide and water, and with the energy from sunlight, convert them into glucose and oxygen."),
                    ("What are the states of matter?", "The four primary states of matter are solid, liquid, gas, and plasma. Each state has unique properties based on the arrangement and energy of their particles. Solids have fixed shape and volume, liquids have fixed volume but take the shape of their container, gases expand to fill their container, and plasma is an ionized gas that conducts electricity."),
                    ("What is the scientific method?", "The scientific method is a systematic approach to research that involves making observations, formulating a hypothesis, testing the hypothesis through experiments, analyzing data, and drawing conclusions. It is the foundation of scientific inquiry and ensures that findings are based on evidence rather than assumptions."),
                    ("What is cellular respiration?", "Cellular respiration is the process by which cells convert nutrients into energy in the form of ATP. It involves three main stages: glycolysis, the Krebs cycle (citric acid cycle), and the electron transport chain. This process requires oxygen and produces carbon dioxide as a waste product."),
                    ("What is biology?", "Biology is the scientific study of living organisms and their interactions with each other and their environments. It encompasses various specialized fields such as molecular biology, cellular biology, genetics, ecology, evolutionary biology, and physiology."),
                    ("What is chemistry?", "Chemistry is the scientific discipline that studies the composition, structure, properties, and changes of matter. It examines atoms, the elements, how they bond to form molecules and compounds, and how substances interact with energy."),
                    ("What is physics?", "Physics is the natural science that studies matter, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines, with its main goal being to understand how the universe behaves."),
                    ("What is ecology?", "Ecology is the branch of biology that studies the relationships between living organisms, including humans, and their physical environment. It examines how organisms interact with each other and with their environment, including the distribution and abundance of organisms."),
                    
                    # History
                    ("Who was Albert Einstein?", "Albert Einstein (1879-1955) was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He is best known for his mass–energy equivalence formula E = mc²."),
                    ("When did World War II end?", "World War II ended in Europe on May 8, 1945 (V-E Day) when Nazi Germany surrendered, and in Asia on September 2, 1945 (V-J Day) when Japan formally surrendered. The war claimed an estimated 70-85 million lives and was the deadliest conflict in human history."),
                    ("Who was Rana Pratap Singh?", "Maharana Pratap Singh (1540-1597) was a Hindu Rajput king of Mewar in Rajasthan, India. He is known for his resistance against the expansionist policy of the Mughal Emperor Akbar and for the Battle of Haldighati in 1576, where he fought bravely despite being outnumbered."),
                    ("What was the Renaissance?", "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity, spanning roughly from the 14th to the 17th century. It was characterized by renewed interest in classical learning and values, artistic and architectural innovations, scientific discoveries, and increased cultural and intellectual exchange."),
                    ("What was the Industrial Revolution?", "The Industrial Revolution was a period of major industrialization and innovation that took place during the late 1700s and early 1800s. It began in Great Britain and spread to other parts of Europe and North America, fundamentally changing economic and social organization through the development of machine-based manufacturing, new energy sources, and transportation systems."),
                    ("Who was Mahatma Gandhi?", "Mahatma Gandhi (1869-1948) was an Indian lawyer, anti-colonial nationalist, and political ethicist who employed nonviolent resistance to lead the successful campaign for India's independence from British rule. His philosophy of nonviolent civil disobedience inspired movements for civil rights and freedom across the world."),
                    ("What was the Cold War?", "The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies from approximately 1947 to 1991. It was characterized by proxy wars, an arms race, ideological competition between capitalism and communism, and a constant threat of nuclear war."),
                    ("What were the Crusades?", "The Crusades were a series of religious wars initiated, supported, and sometimes directed by the Latin Church in the medieval period. The best-known Crusades were those to the Holy Land in the period between 1095 and 1291, which were fought to recover Jerusalem and other holy sites from Islamic rule.")
                ]
                
                # Structure the minimal QA pairs by subject
                minimal_qa_by_subject = {
                    "Mathematics": minimal_qa[:8],
                    "Science": minimal_qa[8:16],
                    "History": minimal_qa[16:24],
                    "Programming": minimal_qa[24:]
                }
                
                self.training_data = minimal_qa_by_subject
                
                # Create a structured dictionary for subject-specific lookups
                self.training_data_dict = {
                    "Mathematics": [(q, a) for q, a in minimal_qa[:8]],  # First 8 items are Mathematics
                    "Science": [(q, a) for q, a in minimal_qa[8:16]],    # Next 8 items are Science
                    "History": [(q, a) for q, a in minimal_qa[16:24]],   # Next 8 items are History
                    "Programming": [(q, a) for q, a in minimal_qa[24:]]  # Remaining items are Programming
                }
                
                # Prepare vectorizer with all questions
                all_questions = [q for subj in self.training_data_dict.keys() for q, _ in self.training_data_dict[subj]]
                self.X = self.vectorizer.fit_transform(all_questions)
                
                debug_log(f"Created comprehensive model with {len(all_questions)} QA pairs")
                # Log the number of QA pairs for each subject
                for subject, qa_pairs in self.training_data_dict.items():
                    debug_log(f"Subject {subject} has {len(qa_pairs)} QA pairs")
            
            debug_log(f"AITutor initialized successfully")
        except Exception as e:
            debug_log(f"Error in AITutor.__init__: {str(e)}")
            raise

    def handle_arithmetic(self, question):
        """
        Process math questions and calculate results
        """
        try:
            # Replace words with math symbols
            math_words = {
                "plus": "+", 
                "add": "+", 
                "added to": "+",
                "minus": "-", 
                "subtract": "-", 
                "take away": "-",
                "times": "*", 
                "multiplied by": "*", 
                "multiply": "*",
                "divided by": "/", 
                "divide": "/"
            }
            
            # Clean the question text
            cleaned = question.strip().lower()
            
            # Convert words to symbols
            for word, symbol in math_words.items():
                cleaned = cleaned.replace(word, symbol)
            
            # Check if it's a math question
            calculation_starters = ["what is", "what's", "whats", "calculate", "compute", "solve", "find", "="]
            is_calculation = False
            for starter in calculation_starters:
                if cleaned.startswith(starter):
                    cleaned = cleaned.replace(starter, "", 1).strip()
                    is_calculation = True
                    break
            
            # Look for math patterns
            if is_calculation or re.search(r'\d+\s*[\+\-\*\/]\s*\d+', cleaned):
                # Keep only numbers and operators
                expression = re.sub(r'[^0-9+\-*/().]', '', cleaned)
                
                # Make sure it's a valid expression
                if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', expression):
                    # Safety check - limit length
                    if len(expression) > 100:
                        return "That expression is too long to calculate safely."
                    
                    # Try to evaluate
                    try:
                        # Use safe eval for math
                        result = eval(expression)
                        
                        # Format result based on type
                        if isinstance(result, int):
                            return f"The answer is {result}."
                        else:
                            # Round to 4 decimal places for floats
                            return f"The answer is approximately {result:.4f}."
                    except Exception as e:
                        debug_log(f"Error evaluating expression: {str(e)}")
                        return "I couldn't evaluate that expression. Please check the format and try again."
            
            # Not a math question
            return None
        
        except Exception as e:
            debug_log(f"Error in handle_arithmetic: {str(e)}")
            return None

    def get_response(self, question, subject):
        try:
            # Clean the input and generate a unique request ID for this query
            cleaned_question = question.strip().lower()
            request_id = str(uuid.uuid4())[:8]
            
            debug_log(f"[{request_id}] Getting response for question: '{cleaned_question}' in subject: {subject}")
            
            # SPECIAL CASE: Handle specific formulas directly
            if subject == "Mathematics":
                # Check for quadratic formula specifically
                if "quadratic" in cleaned_question and ("formula" in cleaned_question or "equation" in cleaned_question):
                    debug_log(f"[{request_id}] Direct match for quadratic formula")
                    return "The quadratic formula is used to solve equations in the form ax² + bx + c = 0. The formula is: x = (-b ± √(b² - 4ac)) / 2a, where a, b, and c are coefficients in the quadratic equation. The discriminant (b² - 4ac) determines the number of solutions: if positive, there are two real solutions; if zero, there is one real solution; if negative, there are two complex solutions."
                
                # Handle general formula questions
                if ("formula" in cleaned_question or "fromula" in cleaned_question):
                    debug_log(f"[{request_id}] General formula query detected")
                    return "A formula in mathematics is a fact or rule written with mathematical symbols. It typically uses an equals sign (=) to show that two expressions have the same value. Formulas express relationships between various quantities and provide a concise way to solve problems. Common mathematical formulas include the quadratic formula (x = (-b ± √(b² - 4ac)) / 2a), the area of a circle (A = πr²), the Pythagorean theorem (a² + b² = c²), and many others specific to different branches of mathematics."
            
            # SPECIAL CASE: Handle arithmetic operations if the subject is Mathematics
            if subject == "Mathematics":
                arithmetic_result = self.handle_arithmetic(cleaned_question)
                if arithmetic_result:
                    debug_log(f"[{request_id}] Handled as arithmetic operation: '{cleaned_question}'")
                    return arithmetic_result
                
                # Handle graph-related questions
                if "graph" in cleaned_question:
                    return "In mathematics, a graph is a structure used to model pairwise relations between objects. Graphs consist of vertices (also called nodes or points) which are connected by edges (also called links or lines). Graphs can be used to model many types of relations and processes in physical, biological, social, and information systems. In mathematics, graphs are used in the study of graph theory."
            
            # SPECIAL CASE HANDLING: Check for direct keyword matches first
            # This is a more reliable approach for common questions
            
            # Keywords specific to each subject
            subject_keywords = {
                "Mathematics": {
                    "pythagorean": "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the other two sides. It is represented by the equation: a² + b² = c², where c is the length of the hypotenuse and a and b are the lengths of the other two sides.",
                    "quadratic": "Quadratic equations can be solved using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a, where ax² + bx + c = 0. Alternatively, you can solve by factoring, completing the square, or graphing, depending on the specific equation.",
                    "matrices": "Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are used in linear algebra for representing linear transformations and solving systems of linear equations.",
                    "matrix": "Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are used in linear algebra for representing linear transformations and solving systems of linear equations.",
                    "calculus": "Calculus is a branch of mathematics that focuses on the study of continuous change. It has two main branches: differential calculus (concerning rates of change and slopes of curves) and integral calculus (concerning accumulation of quantities and areas under curves).",
                    "algebra": "Algebra is a branch of mathematics that uses symbols and letters to represent numbers and quantities in formulas and equations. It introduces the concept of variables and provides tools for solving equations.",
                    "equation": "Equations are mathematical statements that assert the equality of two expressions. They typically contain variables and state that the expressions on either side of the equals sign have the same value.",
                    "trigonometry": "Trigonometry is a branch of mathematics that studies the relationships between the sides and angles of triangles. It defines trigonometric functions such as sine, cosine, and tangent, which relate the angles of a triangle to the lengths of its sides.",
                    "geometry": "Geometry is a branch of mathematics concerned with questions of shape, size, relative position of figures, and the properties of space. It includes the study of points, lines, angles, surfaces, and solids.",
                },
                "Science": {
                    "photosynthesis": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars. Plants take in carbon dioxide and water, and with the energy from sunlight, convert them into glucose and oxygen.",
                    "states of matter": "The four primary states of matter are solid, liquid, gas, and plasma. Each state has unique properties based on the arrangement and energy of their particles. Solids have fixed shape and volume, liquids have fixed volume but take the shape of their container, gases expand to fill their container, and plasma is an ionized gas that conducts electricity.",
                    "scientific method": "The scientific method is a systematic approach to research that involves making observations, formulating a hypothesis, testing the hypothesis through experiments, analyzing data, and drawing conclusions. It is the foundation of scientific inquiry and ensures that findings are based on evidence rather than assumptions.",
                    "cellular respiration": "Cellular respiration is the process by which cells convert nutrients into energy in the form of ATP. It involves three main stages: glycolysis, the Krebs cycle (citric acid cycle), and the electron transport chain. This process requires oxygen and produces carbon dioxide as a waste product.",
                    "biology": "Biology is the scientific study of living organisms and their interactions with each other and their environments. It encompasses various specialized fields such as molecular biology, cellular biology, genetics, ecology, evolutionary biology, and physiology.",
                    "chemistry": "Chemistry is the scientific discipline that studies the composition, structure, properties, and changes of matter. It examines atoms, the elements, how they bond to form molecules and compounds, and how substances interact with energy.",
                    "physics": "Physics is the natural science that studies matter, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines, with its main goal being to understand how the universe behaves.",
                    "ecology": "Ecology is the branch of biology that studies the relationships between living organisms, including humans, and their physical environment. It examines how organisms interact with each other and with their environment, including the distribution and abundance of organisms.",
                }
            }
            
            # Check for direct keyword matches in the current subject
            if subject in subject_keywords:
                for keyword, answer in subject_keywords[subject].items():
                    if keyword in cleaned_question:
                        debug_log(f"[{request_id}] Found direct keyword match: '{keyword}' in subject: {subject}")
                        return answer
            
            # Use the subject-specific QA pairs dictionary if available
            if hasattr(self, 'training_data_dict') and isinstance(self.training_data_dict, dict):
                # Make sure we're only looking at QA pairs for the current subject
                subject_qa_pairs = self.training_data_dict.get(subject, [])
                if not subject_qa_pairs:
                    debug_log(f"[{request_id}] No QA pairs found for subject: {subject}")
                else:
                    debug_log(f"[{request_id}] Found {len(subject_qa_pairs)} QA pairs for subject {subject}")
                    
                    questions = [q for q, _ in subject_qa_pairs]
                    answers = [a for _, a in subject_qa_pairs]
                    
                    # Check for exact matches first (case insensitive)
                    for i, q in enumerate(questions):
                        q_lower = q.lower()
                        if cleaned_question == q_lower:
                            debug_log(f"[{request_id}] Found exact match with: '{q}'")
                            return answers[i]
                    
                    # Check for substring matches
                    for i, q in enumerate(questions):
                        q_lower = q.lower()
                        if cleaned_question in q_lower:
                            debug_log(f"[{request_id}] Found question contains user query: '{q}'")
                            return answers[i]
                            
                    # Try the reverse - if the question contains important keywords from our database
                    # For "what is/are X" questions, extract the X and match
                    if cleaned_question.startswith("what is") or cleaned_question.startswith("what are"):
                        topic = cleaned_question.replace("what is", "").replace("what are", "").strip()
                        debug_log(f"[{request_id}] Extracted topic: '{topic}'")
                        
                        for i, q in enumerate(questions):
                            q_lower = q.lower()
                            if topic in q_lower:
                                debug_log(f"[{request_id}] Found topic match with: '{q}'")
                                return answers[i]
                    
                    # Similarly handle who/when/where/why/how questions
                    question_starters = ["who", "when", "where", "why", "how"]
                    for starter in question_starters:
                        if cleaned_question.startswith(starter):
                            topic = cleaned_question[len(starter):].strip()
                            if topic:
                                debug_log(f"[{request_id}] Extracted topic from {starter} question: '{topic}'")
                                for i, q in enumerate(questions):
                                    q_lower = q.lower()
                                    if topic in q_lower:
                                        debug_log(f"[{request_id}] Found topic match with: '{q}'")
                                        return answers[i]
                    
                    # Use vectorization for semantic matching only within this subject
                    try:
                        # Create a new vectorizer just for this subject's questions
                        subject_vectorizer = TfidfVectorizer(stop_words='english')
                        subject_vectors = subject_vectorizer.fit_transform(questions)
                        user_vector = subject_vectorizer.transform([cleaned_question])
                        
                        similarities = cosine_similarity(user_vector, subject_vectors)[0]
                        best_idx = similarities.argmax()
                        best_score = similarities[best_idx]
                        
                        debug_log(f"[{request_id}] Best match: '{questions[best_idx]}' with score {best_score:.4f}")
                        
                        # Only return if the match is reasonably good
                        if best_score > 0.3:
                            return answers[best_idx]
                    except Exception as e:
                        debug_log(f"[{request_id}] Error in vectorization: {str(e)}")
                        # Continue to fallback responses
            
            # Try using the global vectorizer with all questions but filter by subject
            try:
                if hasattr(self, 'training_data') and isinstance(self.training_data, dict) and hasattr(self, 'vectorizer'):
                    all_subject_questions = []
                    all_subject_answers = []
                    subject_indices = []
                    
                    # Rebuild a list of all questions and answers with subject tracking
                    for subj, qa_pairs in self.training_data_dict.items():
                        if subj == subject:  # Only include the current subject
                            for q, a in qa_pairs:
                                all_subject_questions.append(q)
                                all_subject_answers.append(a)
                                subject_indices.append(subj)
                    
                    if all_subject_questions:
                        # Create new vectors for just this subject's questions
                        subject_only_vectorizer = TfidfVectorizer(stop_words='english')
                        subject_only_vectors = subject_only_vectorizer.fit_transform(all_subject_questions)
                        user_vector = subject_only_vectorizer.transform([cleaned_question])
                        
                        subject_similarities = cosine_similarity(user_vector, subject_only_vectors)[0]
                        subject_best_idx = subject_similarities.argmax()
                        subject_best_score = subject_similarities[subject_best_idx]
                        
                        debug_log(f"[{request_id}] Subject-only best match: '{all_subject_questions[subject_best_idx]}' with score {subject_best_score:.4f}")
                        
                        if subject_best_score > 0.3:
                            return all_subject_answers[subject_best_idx]
            except Exception as e:
                debug_log(f"[{request_id}] Error in global vectorization: {str(e)}")
            
            # Fallback responses by subject
            fallback_responses = {
                "Mathematics": "I don't have specific information about that mathematical concept. Please try asking about the Pythagorean theorem, quadratic equations, matrices, calculus, algebra, equations, trigonometry, or geometry.",
                "Science": "I don't have specific information about that scientific concept. Please try asking about photosynthesis, states of matter, the scientific method, cellular respiration, biology, chemistry, physics, or ecology.",
                "History": "I don't have specific information about that historical topic. Please try asking about Albert Einstein, World War II, Rana Pratap Singh, the Renaissance, the Industrial Revolution, Mahatma Gandhi, the Cold War, or the Crusades.",
                "Programming": "I don't have specific information about that programming concept. Please try asking about variables, object-oriented programming, functions, data structures, Python, algorithms, debugging, or databases."
            }
            
            return fallback_responses.get(subject, f"I don't have enough information about that in {subject}. Could you try asking something else?")
        
        except Exception as e:
            debug_log(f"Error in get_response: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

# Main app
def main():
    debug_log("Entering main function")
    
    # Create necessary directories
    try:
        # Ensure data directory structure exists
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("model", exist_ok=True)
        
        # Create user_progress.pkl if it doesn't exist
        user_progress_file = 'data/user_progress.pkl'
        if not os.path.exists(user_progress_file):
            with open(user_progress_file, 'wb') as f:
                pickle.dump({}, f)
            debug_log(f"Created empty user progress file at {user_progress_file}")
    except Exception as e:
        debug_log(f"Error ensuring data directories: {str(e)}")
    
    # Modern UI styling improved to fix text visibility issues
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- Additional styles for debug -->
    <style>
    .st-emotion-cache-13oz2x0 {
        color: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = ''
    
    if 'current_subject' not in st.session_state:
        st.session_state.current_subject = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    
    # Initialize AITutor model once and properly
    if 'ai_tutor_model' not in st.session_state:
        try:
            debug_log("Initializing AI Tutor model")
            st.session_state.ai_tutor_model = AITutor()
            debug_log("AI Tutor model initialized successfully")
        except Exception as e:
            debug_log(f"Error initializing AI Tutor model: {str(e)}")
            st.session_state.ai_tutor_model = None

    # Fix issue with chat submission by renaming the function to submit_chat
    def submit_chat(user_question, subject):
        try:
            debug_log(f"Processing chat submission: '{user_question}' for subject: {subject}")
            
            # Get AI response
            ai_response = st.session_state.ai_tutor_model.get_response(user_question, subject)
            debug_log(f"Received AI response: {ai_response[:50]}...")
            
            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Update progress
            update_progress(st.session_state.username, subject, user_question)
            
            return True
        except Exception as e:
            debug_log(f"Error in submit_chat: {str(e)}")
            return False

    # Login section
    if not st.session_state.logged_in:
        debug_log("User not logged in - showing login page")
        # Hide the sidebar when not logged in
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Circle logo at the top
        st.markdown("""
        <div class="login-logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 32 32" fill="none" stroke="#e2e8f0" stroke-width="2">
                <circle cx="16" cy="16" r="12" />
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="text-center mb-2" style="font-size: 1.25rem; color: #e2e8f0 !important; font-weight: 600;">Sign up to our AI Tutorial Program</h2>', unsafe_allow_html=True)
        st.markdown('<p class="text-center mb-4" style="color: #94a3b8 !important; font-size: 0.875rem;">We just need a few details to get you started.</p>', unsafe_allow_html=True)
        
        # Username field
        st.markdown('<label for="username" style="display: block; margin-bottom: 0.25rem; color: #e2e8f0 !important; font-weight: 500;">Full name</label>', unsafe_allow_html=True)
        username = st.text_input("Full name", placeholder="Enter your name", key="username_input", label_visibility="collapsed")
        
        # Email field (not used but kept for UI consistency)
        st.markdown('<label for="email" style="display: block; margin-bottom: 0.25rem; color: #e2e8f0 !important; font-weight: 500;">Email</label>', unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="Enter your email", key="email_input", label_visibility="collapsed")
        
        # Password field
        st.markdown('<label for="password" style="display: block; margin-bottom: 0.25rem; color: #e2e8f0 !important; font-weight: 500;">Password</label>', unsafe_allow_html=True)
        password = st.text_input("Password", placeholder="Enter your password", type="password", key="password_input", label_visibility="collapsed")
        
        # Login/Sign up button
        if st.button("Sign up", key="login_button"):
            debug_log(f"Login button clicked with username: {username}")
            if username.strip():
                if len(password) >= 8:
                    debug_log(f"Login successful for: {username}")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    # Show qualification selection page instead of welcome page
                    st.session_state.current_subject = "qualification"
                    st.success("Sign up successful!")
                    debug_log("Rerunning after successful login")
                    st.rerun()
                else:
                    debug_log("Login failed: Password too short")
                    st.error("Password must be at least 8 characters long")
            else:
                debug_log("Login failed: Empty username")
                st.warning("Please enter your name")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        debug_log(f"User logged in: {st.session_state.username}")
        # Configure sidebar
        with st.sidebar:
            # Fix the sidebar header with visible code
            # Replace the existing style declaration in main sidebar area
            st.sidebar.markdown("""
            <style>
            .sidebar-header {
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: #e2e8f0;
            }
            </style>
            <div class="sidebar-header">AI Tutor</div>
            """, unsafe_allow_html=True)

            if st.session_state.logged_in:
                st.sidebar.markdown(f"""
                <div style="margin-bottom: 20px;">
                    <h3 style="font-size: 1.2rem; font-weight: 500; color: #e2e8f0; margin-bottom: 0.5rem;">Welcome, {st.session_state.username}!</h3>
                    <p style="color: #94a3b8; font-size: 0.9rem;">Select a subject below to start learning</p>
                </div>
                """, unsafe_allow_html=True)

            # Subject buttons
            subjects = ["Mathematics", "Science", "History", "Programming"]
            
            for subject in subjects:
                if st.button(subject, key=f"subject_{subject}", help=f"Start learning {subject}", use_container_width=True):
                    debug_log(f"Subject selected: {subject}")
                    st.session_state.current_subject = subject
                    st.session_state.chat_history = []  # Reset chat history when changing subjects
                    debug_log("Rerunning after subject selection")
                    st.rerun()
            
            st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
            
            # Progress button
            if st.button("View My Progress", key="view_progress", use_container_width=True):
                debug_log("Progress view selected")
                st.session_state.current_subject = "progress"
                debug_log("Rerunning after progress selection")
                st.rerun()
            
            # Logout button
            if st.button("Sign Out", key="logout", use_container_width=True):
                debug_log("User logging out")
                st.session_state.logged_in = False
                st.session_state.username = ''
                st.session_state.current_subject = None
                st.session_state.chat_history = []
                debug_log("Rerunning after logout")
                st.rerun()
                
            # Add debug button at the bottom of sidebar
            st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
            if st.button("Show Debug Logs", key="show_debug"):
                # Read the debug log file directly
                if 'show_debug_logs' not in st.session_state:
                    st.session_state.show_debug_logs = True
                else:
                    st.session_state.show_debug_logs = not st.session_state.show_debug_logs
                
                debug_log("Debug log display toggled")

            # Display debug logs if enabled
            if 'show_debug_logs' in st.session_state and st.session_state.show_debug_logs:
                try:
                    st.markdown("### Debug Logs")
                    # Read the most recent log file
                    log_files = sorted([f for f in os.listdir('logs') if f.startswith('ai_tutor_')], reverse=True)
                    if log_files:
                        latest_log = os.path.join('logs', log_files[0])
                        with open(latest_log, 'r') as f:
                            log_content = f.readlines()
                            # Display the last 20 log entries
                            for line in log_content[-20:]:
                                st.text(line.strip())
                    else:
                        st.text("No log files found")
                except Exception as e:
                    st.error(f"Error reading logs: {str(e)}")
        
        # Main content area
        st.markdown('<h1 style="font-size: 1.75rem; margin-bottom: 1.5rem; color: #e2e8f0 !important; font-weight: 600;">AI Tutor</h1>', unsafe_allow_html=True)

        # Use existing model from session state or create new one if needed
        if 'ai_tutor_model' not in st.session_state or st.session_state.ai_tutor_model is None:
            debug_log("Creating new AITutor instance in main content")
            try:
                st.session_state.ai_tutor_model = AITutor()
                debug_log("Successfully created new AITutor instance")
            except Exception as e:
                debug_log(f"Error creating AITutor instance: {str(e)}")
                st.error("Error initializing AI Tutor. Some features may not work correctly.")

        # Log the model status
        if 'ai_tutor_model' in st.session_state and st.session_state.ai_tutor_model is not None:
            debug_log("AITutor model is available")
        else:
            debug_log("AITutor model is NOT available")
        
        # Qualification selection page
        if st.session_state.current_subject == "qualification":
            debug_log("Displaying qualification selection page")
            
            st.markdown("""
            <style>
            .qualification-container {
                background: linear-gradient(145deg, #1e293b, #0f172a);
                border-radius: 1.5rem;
                padding: 3rem;
                margin: 2rem 0;
                text-align: center;
                border: 1px solid #334155;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            }
            
            .qualification-title {
                font-size: 2.5rem;
                color: #e2e8f0;
                margin-bottom: 1.5rem;
                font-weight: 700;
            }
            
            .qualification-description {
                font-size: 1.2rem;
                color: #94a3b8;
                margin-bottom: 2.5rem;
                max-width: 700px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.7;
            }
            </style>
            
            <div class="qualification-container">
                <h2 class="qualification-title">Tell us about your educational background</h2>
                <p class="qualification-description">
                    This will help us personalize your learning experience and recommend appropriate content.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<label style="display: block; margin-bottom: 0.5rem; color: #e2e8f0; font-weight: 500; font-size: 1.1rem;">Current Educational Level</label>', unsafe_allow_html=True)
            
            qualification = st.selectbox(
                "Select your current educational level",
                [
                    "Class 8",
                    "Class 9",
                    "Class 10",
                    "Class 11",
                    "Class 12",
                    "Undergraduate",
                    "Postgraduate",
                    "Doctorate",
                    "Professional",
                    "Other"
                ],
                label_visibility="collapsed"
            )
            
            st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
            
            if st.button("Continue to AI Tutor", use_container_width=True):
                debug_log(f"User qualification selected: {qualification}")
                # Store the qualification in session state for future use
                st.session_state.user_qualification = qualification
                # Now show the welcome page
                st.session_state.current_subject = None
                debug_log("Redirecting to welcome page")
                st.rerun()
        
        # Fix the welcome page rendering by using st.components.v1.html instead of st.markdown
        elif st.session_state.current_subject is None or st.session_state.current_subject == "":
            debug_log("Displaying welcome page using components.html method")
            
            # Add the CSS styling first
            st.markdown("""
            <style>
            /* Global styles for welcome page */
            .welcome-page h1, .welcome-page h2, .welcome-page h3 {
                color: white !important;
                font-weight: 700 !important;
            }
            
            /* Hero banner */
            .hero-banner {
                background: linear-gradient(to right, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.8)), 
                            url('https://images.unsplash.com/photo-1501504905252-473c47e087f8?auto=format&fit=crop&w=1674&q=80');
                background-size: cover;
                background-position: center;
                border-radius: 1.5rem;
                padding: 6rem 3rem;
                margin-bottom: 4rem;
                text-align: center;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.1);
                animation: gradientShift 10s ease infinite alternate;
            }
            
            @keyframes gradientShift {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
            
            .hero-title {
                font-size: 5rem !important;
                font-weight: 900 !important;
                margin-bottom: 2rem !important;
                color: white !important;
                text-shadow: 0 4px 8px rgba(0, 0, 0, 0.5) !important;
                line-height: 1.1 !important;
                animation: titleFadeIn 1.5s ease-out;
            }
            
            @keyframes titleFadeIn {
                0% {opacity: 0; transform: translateY(-20px);}
                100% {opacity: 1; transform: translateY(0);}
            }
            
            .hero-subtitle {
                font-size: 2.25rem !important;
                max-width: 900px;
                margin: 0 auto 3rem auto !important;
                color: #e2e8f0 !important;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
                line-height: 1.4 !important;
                animation: subtitleFadeIn 1.5s ease-out 0.5s both;
            }
            
            @keyframes subtitleFadeIn {
                0% {opacity: 0; transform: translateY(20px);}
                100% {opacity: 1; transform: translateY(0);}
            }
            
            .quote-container {
                background-color: rgba(15, 23, 42, 0.7);
                max-width: 900px;
                margin: 0 auto;
                padding: 2.5rem;
                border-radius: 1.5rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                animation: quoteFadeIn 1.5s ease-out 1s both;
            }
            
            @keyframes quoteFadeIn {
                0% {opacity: 0; transform: scale(0.95);}
                100% {opacity: 1; transform: scale(1);}
            }
            
            .quote-text {
                font-style: italic;
                color: #e2e8f0 !important;
                font-size: 1.75rem !important;
                margin: 0 !important;
                line-height: 1.6 !important;
            }
            
            .quote-author {
                color: #94a3b8 !important;
                margin-top: 1.5rem !important;
                font-size: 1.25rem !important;
                text-align: right !important;
            }
            
            /* Feature cards */
            .features-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2.5rem;
                margin: 4rem 0;
            }
            
            .feature-card {
                background: linear-gradient(145deg, #1e293b, #111827);
                padding: 3rem 2rem;
                border-radius: 1.5rem;
                border: 1px solid #334155;
                box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                position: relative;
                overflow: hidden;
                z-index: 1;
            }
            
            .feature-card::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
                z-index: -1;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
            }
            
            .feature-card:hover::before {
                opacity: 1;
            }
            
            .feature-icon {
                font-size: 4rem !important;
                margin-bottom: 2rem !important;
                text-align: center !important;
                display: block !important;
            }
            
            .feature-title {
                font-size: 2rem !important;
                margin-bottom: 1rem !important;
                color: #e2e8f0 !important;
                font-weight: 700 !important;
                text-align: center !important;
            }
            
            .feature-description {
                color: #94a3b8 !important;
                line-height: 1.7 !important;
                font-size: 1.2rem !important;
                text-align: center !important;
            }
            
            /* Subject cards */
            .subjects-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 2.5rem;
                margin: 4rem 0;
            }
            
            .subject-card {
                background-color: #1e293b;
                border-radius: 1.5rem;
                overflow: hidden;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
                transition: transform 0.4s ease, box-shadow 0.4s ease;
                cursor: pointer;
                position: relative;
                border: 1px solid #334155;
                height: 420px; /* Fixed height for consistency */
            }
            
            .subject-card:hover {
                transform: translateY(-15px) scale(1.02);
                box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.4);
            }
            
            .subject-banner {
                height: 180px;
                background-size: cover;
                background-position: center;
                position: relative;
            }
            
            .subject-banner::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(to bottom, rgba(15, 23, 42, 0.4), rgba(15, 23, 42, 0.9));
            }
            
            .math-banner { 
                background-image: url('https://images.unsplash.com/photo-1635070041078-e363dbe005cb?auto=format&fit=crop&w=800&q=80');
            }
            
            .science-banner { 
                background-image: url('https://images.unsplash.com/photo-1564325724739-bae0bd08762c?auto=format&fit=crop&w=800&q=80');
            }
            
            .history-banner { 
                background-image: url('https://images.unsplash.com/photo-1461360370896-922624d12aa1?auto=format&fit=crop&w=800&q=80');
            }
            
            .programming-banner { 
                background-image: url('https://images.unsplash.com/photo-1555066931-4365d14bab8c?auto=format&fit=crop&w=800&q=80');
            }
            
            .subject-content {
                padding: 2.5rem;
                position: relative;
            }
            
            .subject-icon {
                position: absolute;
                top: -35px;
                left: 50%;
                transform: translateX(-50%);
                width: 70px;
                height: 70px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2rem;
                z-index: 10;
                box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.4);
            }
            
            .icon-math {
                background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            }
            
            .icon-science {
                background: linear-gradient(135deg, #14b8a6, #0f766e);
            }
            
            .icon-history {
                background: linear-gradient(135deg, #8b5cf6, #6d28d9);
            }
            
            .icon-programming {
                background: linear-gradient(135deg, #10b981, #047857);
            }
            
            .subject-title {
                font-size: 2rem !important;
                text-align: center !important;
                margin-top: 1.5rem !important;
                margin-bottom: 1rem !important;
                color: #e2e8f0 !important;
                font-weight: 700 !important;
            }
            
            .subject-description {
                color: #94a3b8 !important;
                text-align: center !important;
                margin-bottom: 1.5rem !important;
                font-size: 1.1rem !important;
                line-height: 1.7 !important;
            }
            
            .subject-button {
                display: block !important;
                background-color: #334155 !important;
                color: white !important;
                padding: 1rem 0 !important;
                border-radius: 0.75rem !important;
                text-align: center !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
                text-decoration: none !important;
                border: none !important;
                cursor: pointer !important;
                width: 100% !important;
                font-size: 1.15rem !important;
                position: absolute;
                bottom: 2.5rem;
                left: 0;
                right: 0;
                margin: 0 2.5rem;
                width: calc(100% - 5rem) !important;
            }
            
            .math-button:hover { background-color: #3b82f6 !important; }
            .science-button:hover { background-color: #14b8a6 !important; }
            .history-button:hover { background-color: #8b5cf6 !important; }
            .programming-button:hover { background-color: #10b981 !important; }
            
            /* Call to action */
            .cta-container {
                background: linear-gradient(145deg, #1e293b, #0f172a);
                border-radius: 1.5rem;
                padding: 4rem 3rem;
                margin: 4rem 0;
                text-align: center;
                border: 1px solid #334155;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                position: relative;
                overflow: hidden;
            }
            
            .cta-container::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 6px;
                background: linear-gradient(to right, #3b82f6, #8b5cf6, #14b8a6, #10b981);
                z-index: 1;
            }
            
            .cta-title {
                font-size: 3rem !important;
                color: #e2e8f0 !important;
                margin-bottom: 1.5rem !important;
                font-weight: 800 !important;
            }
            
            .cta-description {
                font-size: 1.5rem !important;
                color: #94a3b8 !important;
                margin-bottom: 2.5rem !important;
                max-width: 900px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                line-height: 1.7 !important;
            }
            
            .background-lights {
                position: absolute;
                width: 15rem;
                height: 15rem;
                border-radius: 50%;
                filter: blur(100px);
                opacity: 0.15;
                z-index: -1;
            }
            
            .light-1 {
                background-color: #3b82f6;
                top: -5rem;
                left: -5rem;
                animation: floatLight 15s ease-in-out infinite alternate;
            }
            
            .light-2 {
                background-color: #8b5cf6;
                bottom: -5rem;
                right: -5rem;
                animation: floatLight 18s ease-in-out infinite alternate-reverse;
            }
            
            .light-3 {
                background-color: #10b981;
                top: 50%;
                right: 10%;
                animation: floatLight 20s ease-in-out infinite alternate;
            }
            
            @keyframes floatLight {
                0% {transform: translate(0, 0) scale(1);}
                50% {transform: translate(100px, 50px) scale(1.2);}
                100% {transform: translate(50px, 100px) scale(1);}
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Use the HTML component approach
            components.html("""
            <div class="welcome-page" style="color: white; font-family: Inter, sans-serif;">
                <div style="background: linear-gradient(to right, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.8)); 
                            background-size: cover;
                            background-position: center;
                            border-radius: 1.5rem;
                            padding: 6rem 3rem;
                            margin-bottom: 4rem;
                            text-align: center;
                            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                            border: 1px solid rgba(255, 255, 255, 0.1);">
                    <h1 style="font-size: 5rem; font-weight: 900; margin-bottom: 2rem; color: white; 
                              text-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); line-height: 1.1;">
                        Welcome to AI Tutor
                    </h1>
                    <p style="font-size: 2.25rem; max-width: 900px; margin: 0 auto 3rem auto; 
                            color: #e2e8f0; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); line-height: 1.4;">
                        Experience personalized education with your intelligent AI tutor companion
                    </p>
                    
                    <div style="background-color: rgba(15, 23, 42, 0.7);
                            max-width: 900px;
                            margin: 0 auto;
                            padding: 2.5rem;
                            border-radius: 1.5rem;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            backdrop-filter: blur(10px);">
                        <p style="font-style: italic;
                               color: #e2e8f0;
                               font-size: 1.75rem;
                               margin: 0;
                               line-height: 1.6;">
                            "Education is not the filling of a pail, but the lighting of a fire."
                        </p>
                        <p style="color: #94a3b8;
                               margin-top: 1.5rem;
                               font-size: 1.25rem;
                               text-align: right;">
                            — William Butler Yeats
                        </p>
                    </div>
                </div>
                
                <h2 style="font-size: 3.5rem; font-weight: 800; text-align: center; margin: 3rem 0 2rem 0; color: #e2e8f0;">
                    Your Personal AI Learning Assistant
                </h2>
                <p style="text-align: center; font-size: 1.5rem; max-width: 900px; margin: 0 auto 3rem auto; color: #94a3b8; line-height: 1.7;">
                    Discover a new way of learning with our advanced AI tutor that adapts to your needs and helps you master any subject at your own pace.
                </p>
                
                <div style="display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 2.5rem;
                        margin: 4rem 0;">
                    <!-- Learn Card -->
                    <div style="background: linear-gradient(145deg, #1e293b, #111827);
                            padding: 3rem 2rem;
                            border-radius: 1.5rem;
                            border: 1px solid #334155;
                            box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.3);">
                        <div style="font-size: 4rem; margin-bottom: 2rem; text-align: center; display: block;">📚</div>
                        <h3 style="font-size: 2rem; margin-bottom: 1rem; color: #e2e8f0; font-weight: 700; text-align: center;">
                            Learn Anything
                        </h3>
                        <p style="color: #94a3b8; line-height: 1.7; font-size: 1.2rem; text-align: center;">
                            Explore multiple subjects from mathematics and science to history and programming. Our AI tutor provides in-depth knowledge and personalized explanations.
                        </p>
                    </div>
                    
                    <!-- Interact Card -->
                    <div style="background: linear-gradient(145deg, #1e293b, #111827);
                            padding: 3rem 2rem;
                            border-radius: 1.5rem;
                            border: 1px solid #334155;
                            box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.3);">
                        <div style="font-size: 4rem; margin-bottom: 2rem; text-align: center; display: block;">💬</div>
                        <h3 style="font-size: 2rem; margin-bottom: 1rem; color: #e2e8f0; font-weight: 700; text-align: center;">
                            Interactive Dialogue
                        </h3>
                        <p style="color: #94a3b8; line-height: 1.7; font-size: 1.2rem; text-align: center;">
                            Engage in natural conversations with your AI tutor. Ask questions, get clarification, and dive deeper into topics that interest you most.
                        </p>
                    </div>
                    
                    <!-- Progress Card -->
                    <div style="background: linear-gradient(145deg, #1e293b, #111827);
                            padding: 3rem 2rem;
                            border-radius: 1.5rem;
                            border: 1px solid #334155;
                            box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.3);">
                        <div style="font-size: 4rem; margin-bottom: 2rem; text-align: center; display: block;">📊</div>
                        <h3 style="font-size: 2rem; margin-bottom: 1rem; color: #e2e8f0; font-weight: 700; text-align: center;">
                            Track Progress
                        </h3>
                        <p style="color: #94a3b8; line-height: 1.7; font-size: 1.2rem; text-align: center;">
                            Monitor your learning journey with detailed progress tracking. Review past sessions, see your improvement, and identify areas for further study.
                        </p>
                    </div>
                </div>
                
                <h2 style="font-size: 3.5rem; font-weight: 800; text-align: center; margin: 5rem 0 2rem 0; color: #e2e8f0;">
                    Start Learning Now
                </h2>
                <p style="text-align: center; font-size: 1.5rem; max-width: 900px; margin: 0 auto 3rem auto; color: #94a3b8; line-height: 1.7;">
                    Choose from our wide range of subjects and begin your personalized learning experience with our AI tutor.
                </p>
            </div>
            """, height=1200)
            
            # Add basic subject selection buttons
            st.markdown("<h2 style='text-align: center; margin-top: 30px; color: #e2e8f0;'>Select a Subject to Start</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Mathematics", use_container_width=True):
                    st.session_state.current_subject = "Mathematics"
                    st.rerun()
                
                if st.button("History", use_container_width=True):
                    st.session_state.current_subject = "History"
                    st.rerun()
            
            with col2:
                if st.button("Science", use_container_width=True):
                    st.session_state.current_subject = "Science"
                    st.rerun()
                
                if st.button("Programming", use_container_width=True):
                    st.session_state.current_subject = "Programming"
                    st.rerun()
            
            debug_log("Welcome page display complete with fixed rendering using components.html")
        
        # Subject specific chat interface
        elif st.session_state.current_subject != "progress":
            debug_log(f"Displaying chat interface for subject: {st.session_state.current_subject}")
            st.markdown(f'<h2 style="font-size: 1.5rem; margin-bottom: 1rem; color: #e2e8f0 !important; font-weight: 600;">Learning {st.session_state.current_subject}</h2>', unsafe_allow_html=True)
            
            # Initialize chat history if not present
            if "chat_history" not in st.session_state:
                debug_log("Initializing chat history")
                st.session_state.chat_history = []
                
            # Generate a unique key for the input field on each submission
            if "input_key" not in st.session_state:
                st.session_state.input_key = 0
                
            # Display chat history
            debug_log(f"Displaying chat history: {len(st.session_state.chat_history)} messages")
            
            # Debug output of actual chat content
            for idx, msg in enumerate(st.session_state.chat_history):
                debug_log(f"Message {idx}: role={msg['role']}, content={msg['content'][:30]}...")
            
            # Simple container for chat
            st.subheader(f"Chat with AI Tutor - {st.session_state.current_subject}")
            
            # Create a consistent chat container
            chat_container = st.container(border=True)
            
            with chat_container:
                if len(st.session_state.chat_history) == 0:
                    st.info("No messages yet. Ask your first question below!")
                else:
                    # Display each message with proper styling
                    for i, message in enumerate(st.session_state.chat_history):
                        if message["role"] == "user":
                            with st.container(border=False):
                                st.markdown(f"### You:")
                                st.write(message["content"])
                                st.write("---")
                        else:
                            with st.container(border=False):
                                st.markdown(f"### AI Tutor:")
                                st.write(message["content"])
                                st.write("---")
            
            # Add some space before the input 
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
            # Use a standard input instead of a form (simpler approach)
            debug_log("Setting up chat input without form")
            
            # Check first to show success message at the top before the input field
            if 'last_submitted' in st.session_state and st.session_state.last_submitted:
                st.success(f"Message sent successfully! The AI tutor has responded to your question.")
                # Reset after displaying
                st.session_state.last_submitted = False
            
            # Create a styled container for the chat input
            st.markdown("""
            <div style="background-color: #1e293b; border-radius: 0.5rem; padding: 1.5rem; margin-top: 1rem; border: 1px solid #334155;">
                <h3 style="font-size: 1.25rem; color: #e2e8f0 !important; margin-bottom: 1rem; font-weight: 600;">Ask Your Question</h3>
            """, unsafe_allow_html=True)
            
            # Label for the input
            st.markdown(f"""
            <label style="display: block; margin-bottom: 0.75rem; color: #e2e8f0 !important; font-weight: 500;">
                Type your question about {st.session_state.current_subject} below:
            </label>
            """, unsafe_allow_html=True)
            
            # Text input for user question - use a dynamic key to force reset
            user_input = st.text_input(
                label=f"Ask a question about {st.session_state.current_subject}",
                key=f"user_input_{st.session_state.input_key}",
                placeholder="Type your question here...",
                label_visibility="collapsed"  # Hide the label as we've added it manually above
            )
            
            # Submit button
            submit_pressed = st.button(
                "Submit Question",
                key="direct_submit",
                use_container_width=True
            )
            
            # Close the container
            st.markdown("</div>", unsafe_allow_html=True)
            
            debug_log(f"Direct submit button created with key 'direct_submit', is pressed: {submit_pressed}")
            
            if submit_pressed and user_input:
                debug_log(f"Chat direct submit with input: {user_input}")
                try:
                    # Store the user question to ensure it's saved
                    user_question = user_input
                    
                    # Use the better_submit_chat function instead
                    success, message = better_submit_chat(user_question, st.session_state.current_subject)
                    
                    if success:
                        debug_log("Chat submission successful")
                        
                        # Set flag to show success message after rerun
                        st.session_state.last_submitted = True
                        
                        # Increment the input key to reset the field on next render
                        st.session_state.input_key += 1
                        debug_log(f"Incremented input key to {st.session_state.input_key}")
                        debug_log("Rerunning after chat submission")
                        st.rerun()
                    else:
                        debug_log(f"Chat submission failed: {message}")
                        st.error(f"Failed to process your question: {message}")
                except Exception as e:
                    debug_log(f"Error processing chat submission: {str(e)}")
                    debug_log(f"Chat submission error: {traceback.format_exc()}")
                    st.error(f"An error occurred: {str(e)}")
            
            elif submit_pressed:
                debug_log("Submit pressed but no input")
                st.warning("Please enter a question")
            else:
                debug_log(f"No submit yet - input: {'empty' if not user_input else 'has content'}")
        
        # Progress page
        elif st.session_state.current_subject == "progress":
            debug_log("Displaying progress page using improved function")
            
            st.markdown("""
            <style>
            .progress-header {
                font-size: 2.5rem !important;
                font-weight: 700 !important;
                margin-bottom: 1rem !important;
                color: #e2e8f0 !important;
                text-align: center !important;
            }
            /* Keep all the CSS styling from before... */
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<h1 class="progress-header">Your Learning Progress</h1>', unsafe_allow_html=True)

            # Get user progress data using our fixed function
            username = st.session_state.username
            user_progress = get_user_progress_data(username)
            
            if not user_progress:
                st.info("No learning progress data available yet. Start learning by selecting a subject and asking questions!")
            else:
                # Overall stats section
                st.markdown('<h2 class="progress-subheader">Overall Learning Stats</h2>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                # Count total subjects
                total_subjects = len(user_progress.keys())
                
                # Count total questions
                total_questions = sum(subject_data.get("questions_asked", 0) for subject_data in user_progress.values())
                
                # Count total sessions
                total_sessions = sum(len(subject_data.get("sessions", [])) for subject_data in user_progress.values())
                
                with col1:
                    st.markdown(
                        f'<div class="progress-card">'
                        f'<div class="progress-stat-label">Subjects Explored</div>'
                        f'<div class="progress-stat-value">{total_subjects}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                with col2:
                    st.markdown(
                        f'<div class="progress-card">'
                        f'<div class="progress-stat-label">Total Questions Asked</div>'
                        f'<div class="progress-stat-value">{total_questions}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                with col3:
                    st.markdown(
                        f'<div class="progress-card">'
                        f'<div class="progress-stat-label">Learning Sessions</div>'
                        f'<div class="progress-stat-value">{total_sessions}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Subject-specific progress
                st.markdown('<h2 class="progress-subheader">Subject Progress</h2>', unsafe_allow_html=True)
                
                for subject, data in user_progress.items():
                    with st.expander(f"{subject.capitalize()} Progress", expanded=True):
                        last_session = data.get("last_session", "No session recorded")
                        questions_asked = data.get("questions_asked", 0)
                        mastery_level = data.get("mastery_level", 0)
                        sessions = data.get("sessions", [])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(
                                f'<div class="progress-card">'
                                f'<div class="progress-stat-label">Last Session</div>'
                                f'<div class="progress-stat-value">{last_session}</div>'
                                f'<div class="progress-stat-label">Questions Asked</div>'
                                f'<div class="progress-stat-value">{questions_asked}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            st.markdown(
                                f'<div class="progress-card">'
                                f'<div class="progress-stat-label">Mastery Level</div>'
                                f'<div class="mastery-container">'
                                f'<div class="mastery-label">Progress <span class="mastery-value">{mastery_level}%</span></div>'
                                f'<div class="mastery-bar-container">'
                                f'<div class="mastery-bar" style="width: {mastery_level}%;"></div>'
                                f'</div>'
                                f'<div class="mastery-explanation">Mastery percentage is calculated based on the number of questions you\'ve asked and sessions completed. Continue learning to increase your mastery level!</div>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Recent questions from sessions
                        if sessions:
                            st.markdown('<div class="progress-card">', unsafe_allow_html=True)
                            st.markdown('<div class="progress-stat-label">Recent Learning Sessions</div>', unsafe_allow_html=True)
                            
                            # Sort sessions by timestamp, newest first
                            sorted_sessions = sorted(sessions, key=lambda x: x.get("timestamp", ""), reverse=True)
                            
                            # Display the 5 most recent sessions
                            for i, session in enumerate(sorted_sessions[:5]):
                                timestamp = session.get("timestamp", "Unknown date")
                                questions = session.get("questions", [])
                                
                                st.markdown(
                                    f'<div class="session-item">'
                                    f'<div class="session-date">{timestamp}</div>',
                                    unsafe_allow_html=True
                                )
                                
                                if questions:
                                    for q in questions[:3]:  # Show up to 3 questions per session
                                        st.markdown(f'<div class="question-item">{q}</div>', unsafe_allow_html=True)
                                    
                                    if len(questions) > 3:
                                        st.markdown(f'<div class="question-item">+ {len(questions) - 3} more questions</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="question-item">No questions recorded for this session</div>', unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<div class="progress-card">'
                                '<div class="progress-stat-label">Recent Learning Sessions</div>'
                                '<div class="no-data-message" style="margin: 1rem 0;">No session data available for this subject yet.</div>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                
                # Learning tips based on progress
                st.markdown('<h2 class="progress-subheader">Learning Recommendations</h2>', unsafe_allow_html=True)
                
                st.markdown(
                    '<div class="progress-card">'
                    '<div class="progress-stat-label">Personalized Tips</div>',
                    unsafe_allow_html=True
                )
                
                # Generate some basic recommendations based on user data
                if total_questions < 10:
                    st.markdown(
                        '<div class="question-item">You\'re just getting started! Try asking more questions to build your knowledge base.</div>'
                        '<div class="question-item">Explore different subjects to broaden your learning experience.</div>'
                        '<div class="question-item">Try asking follow-up questions to deepen your understanding of topics.</div>',
                        unsafe_allow_html=True
                    )
                elif total_questions < 50:
                    st.markdown(
                        '<div class="question-item">You\'re making good progress! Continue exploring topics that interest you.</div>'
                        '<div class="question-item">Try asking more complex questions to challenge yourself.</div>'
                        '<div class="question-item">Review subjects you haven\'t visited recently to reinforce your knowledge.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="question-item">Impressive progress! You\'re becoming an expert in your subjects.</div>'
                        '<div class="question-item">Consider teaching concepts to others to solidify your understanding.</div>'
                        '<div class="question-item">Explore advanced topics to keep challenging yourself.</div>',
                        unsafe_allow_html=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)

# Function to display a clickable subject card
def subject_card(title, description, color):
    debug_log(f"Creating subject card for: {title}")
    card_html = f"""
    <div onclick="document.querySelector('button[key=\\'subject_{title}\\']').click();" 
         style="background-color: #1e293b; 
                border-radius: 0.5rem; 
                border: 1px solid #334155;
                border-top: 3px solid {color}; 
                padding: 1.25rem; 
                margin-bottom: 1rem; 
                cursor: pointer;
                transition: all 0.15s ease-in-out;">
        <h4 style="font-size: 1.125rem; color: {color} !important; margin-bottom: 0.5rem; font-weight: 600;">{title}</h4>
        <p style="font-size: 0.875rem; color: #94a3b8 !important;">{description}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    debug_log(f"Subject card rendered for: {title}")

# Add this function back - it was removed accidentally
def get_progress(username):
    debug_log(f"Loading progress for user: {username}")
    progress_file = f"data/users/{username}/progress.pkl"
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                debug_log(f"Opening progress file: {progress_file}")
                return pickle.load(f)
        except Exception as e:
            debug_log(f"Error loading progress: {str(e)}")
            return {}
    else:
        debug_log(f"No progress file found at {progress_file}")
        return {}

# Function to load the AI Tutor model
def load_model():
    """Load the AI tutor model from file."""
    try:
        # Try to load the large dataset model first
        large_model_path = "model/large_ai_tutor_model.pkl"
        standard_model_path = "model/ai_tutor_model.pkl"
        
        debug_log("Checking for available models...")
        
        # Try the large model first
        if os.path.exists(large_model_path):
            debug_log(f"Opening large model file: {large_model_path}")
            with open(large_model_path, 'rb') as f:
                model_data = pickle.load(f)
                debug_log("Large model data loaded successfully")
                
                # Detect model format - new expanded format has 3 items
                if isinstance(model_data, tuple) and len(model_data) == 3:
                    vectorizer, X, training_data = model_data
                    debug_log(f"Found expanded large model format with {sum(len(v)//2 for v in training_data.values() if isinstance(v, list))} QA pairs")
                    return AITutor(vectorizer=vectorizer, X=X, training_data=training_data)
                else:
                    debug_log("Found legacy model format")
                    return AITutor(model_path=large_model_path)
        
        # Fallback to standard model if large model not found
        elif os.path.exists(standard_model_path):
            debug_log(f"Large model not found. Opening standard model file: {standard_model_path}")
            with open(standard_model_path, 'rb') as f:
                model_data = pickle.load(f)
                debug_log("Standard model data loaded successfully")
                
                # Detect model format - new expanded format has 3 items
                if isinstance(model_data, tuple) and len(model_data) == 3:
                    vectorizer, X, training_data = model_data
                    debug_log(f"Found expanded model format with {sum(len(v)//2 for v in training_data.values() if isinstance(v, list))} QA pairs")
                    return AITutor(vectorizer=vectorizer, X=X, training_data=training_data)
                else:
                    debug_log("Found legacy model format")
                    return AITutor(model_path=standard_model_path)
        else:
            debug_log("No model files found, using fallback model")
            return AITutor()
    except Exception as e:
        debug_log(f"Error loading model: {str(e)}")
        return AITutor()

# Improve chat interface error handling
# Add better error handling and display for chat errors
def better_submit_chat(user_question, subject):
    """Enhanced version of submit_chat with better error handling"""
    debug_log(f"Processing chat submission with better error handling: '{user_question}' for subject: {subject}")
    
    try:
        # Validate inputs
        if not user_question or not user_question.strip():
            debug_log("Empty question detected")
            return False, "Please enter a question before submitting."
        
        if not subject:
            debug_log("Missing subject")
            return False, "Please select a subject first."
        
        # Get AI response with additional error handling
        try:
            ai_response = None
            if 'ai_tutor_model' in st.session_state and st.session_state.ai_tutor_model:
                debug_log("Using session state AI model")
                ai_response = st.session_state.ai_tutor_model.get_response(user_question, subject)
            else:
                debug_log("Creating new AI model instance")
                temp_model = AITutor()
                ai_response = temp_model.get_response(user_question, subject)
            
            debug_log(f"Received AI response: {ai_response[:50]}...")
            
            # Check if the response is valid
            if not ai_response or ai_response.strip() == "":
                debug_log("Empty AI response received")
                return False, "The AI returned an empty response. Please try a different question."
            
            # Update chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Update progress
            try:
                update_progress(st.session_state.username, subject, user_question)
                debug_log("Progress updated successfully")
            except Exception as progress_error:
                debug_log(f"Error updating progress (non-critical): {str(progress_error)}")
                # Don't fail the whole submission just because progress update failed
            
            return True, "Success"
        except Exception as model_error:
            debug_log(f"AI model error: {str(model_error)}")
            
            # Add the error to chat history for visibility
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred while processing your question: {str(model_error)}"})
            
            return False, f"AI model error: {str(model_error)}"
    except Exception as e:
        debug_log(f"Submission process error: {str(e)}")
        return False, f"Error in submission process: {str(e)}"

# Fix the get_response method to prevent wrong subject answers
def subject_specific_get_response(self, question, subject):
    """An improved response method with more precise question matching and diverse answers"""
    try:
        debug_log(f"Using enhanced subject_specific_get_response for question: '{question}' in subject: {subject}")
        cleaned_question = question.strip().lower()
        request_id = str(uuid.uuid4())[:8]
        
        # Create a more comprehensive QA database for each subject
        minimal_qa_by_subject = {
            "Mathematics": [
                ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the other two sides. It is represented by the equation: a² + b² = c², where c is the length of the hypotenuse and a and b are the lengths of the other two sides."),
                ("How do you solve a quadratic equation?", "Quadratic equations can be solved using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a, where ax² + bx + c = 0. Alternatively, you can solve by factoring, completing the square, or graphing, depending on the specific equation."),
                ("What are matrices?", "Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are used in linear algebra for representing linear transformations and solving systems of linear equations."),
                ("What is calculus?", "Calculus is a branch of mathematics that focuses on the study of continuous change. It has two main branches: differential calculus (concerning rates of change and slopes of curves) and integral calculus (concerning accumulation of quantities and areas under curves)."),
                ("What is algebra?", "Algebra is a branch of mathematics that uses symbols and letters to represent numbers and quantities in formulas and equations. It introduces the concept of variables and provides tools for solving equations."),
                ("What are equations?", "Equations are mathematical statements that assert the equality of two expressions. They typically contain variables and state that the expressions on either side of the equals sign have the same value."),
                ("What is trigonometry?", "Trigonometry is a branch of mathematics that studies the relationships between the sides and angles of triangles. It defines trigonometric functions such as sine, cosine, and tangent, which relate the angles of a triangle to the lengths of its sides."),
                ("What is geometry?", "Geometry is a branch of mathematics concerned with questions of shape, size, relative position of figures, and the properties of space. It includes the study of points, lines, angles, surfaces, and solids.")
            ],
            "Science": [
                ("What is photosynthesis?", "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars. Plants take in carbon dioxide and water, and with the energy from sunlight, convert them into glucose and oxygen."),
                ("What are the states of matter?", "The four primary states of matter are solid, liquid, gas, and plasma. Each state has unique properties based on the arrangement and energy of their particles. Solids have fixed shape and volume, liquids have fixed volume but take the shape of their container, gases expand to fill their container, and plasma is an ionized gas that conducts electricity."),
                ("What is the scientific method?", "The scientific method is a systematic approach to research that involves making observations, formulating a hypothesis, testing the hypothesis through experiments, analyzing data, and drawing conclusions. It is the foundation of scientific inquiry and ensures that findings are based on evidence rather than assumptions."),
                ("What is cellular respiration?", "Cellular respiration is the process by which cells convert nutrients into energy in the form of ATP. It involves three main stages: glycolysis, the Krebs cycle (citric acid cycle), and the electron transport chain. This process requires oxygen and produces carbon dioxide as a waste product."),
                ("What is biology?", "Biology is the scientific study of living organisms and their interactions with each other and their environments. It encompasses various specialized fields such as molecular biology, cellular biology, genetics, ecology, evolutionary biology, and physiology."),
                ("What is chemistry?", "Chemistry is the scientific discipline that studies the composition, structure, properties, and changes of matter. It examines atoms, the elements, how they bond to form molecules and compounds, and how substances interact with energy."),
                ("What is physics?", "Physics is the natural science that studies matter, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines, with its main goal being to understand how the universe behaves."),
                ("What is ecology?", "Ecology is the branch of biology that studies the relationships between living organisms, including humans, and their physical environment. It examines how organisms interact with each other and with their environment, including the distribution and abundance of organisms.")
            ],
            "History": [
                ("Who was Albert Einstein?", "Albert Einstein (1879-1955) was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He is best known for his mass–energy equivalence formula E = mc²."),
                ("When did World War II end?", "World War II ended in Europe on May 8, 1945 (V-E Day) when Nazi Germany surrendered, and in Asia on September 2, 1945 (V-J Day) when Japan formally surrendered. The war claimed an estimated 70-85 million lives and was the deadliest conflict in human history."),
                ("Who was Rana Pratap Singh?", "Maharana Pratap Singh (1540-1597) was a Hindu Rajput king of Mewar in Rajasthan, India. He is known for his resistance against the expansionist policy of the Mughal Emperor Akbar and for the Battle of Haldighati in 1576, where he fought bravely despite being outnumbered."),
                ("What was the Renaissance?", "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity, spanning roughly from the 14th to the 17th century. It was characterized by renewed interest in classical learning and values, artistic and architectural innovations, scientific discoveries, and increased cultural and intellectual exchange."),
                ("What was the Industrial Revolution?", "The Industrial Revolution was a period of major industrialization and innovation that took place during the late 1700s and early 1800s. It began in Great Britain and spread to other parts of Europe and North America, fundamentally changing economic and social organization through the development of machine-based manufacturing, new energy sources, and transportation systems."),
                ("Who was Mahatma Gandhi?", "Mahatma Gandhi (1869-1948) was an Indian lawyer, anti-colonial nationalist, and political ethicist who employed nonviolent resistance to lead the successful campaign for India's independence from British rule. His philosophy of nonviolent civil disobedience inspired movements for civil rights and freedom across the world."),
                ("What was the Cold War?", "The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies from approximately 1947 to 1991. It was characterized by proxy wars, an arms race, ideological competition between capitalism and communism, and a constant threat of nuclear war."),
                ("What were the Crusades?", "The Crusades were a series of religious wars initiated, supported, and sometimes directed by the Latin Church in the medieval period. The best-known Crusades were those to the Holy Land in the period between 1095 and 1291, which were fought to recover Jerusalem and other holy sites from Islamic rule.")
            ],
            "Programming": [
                ("What is a variable in programming?", "A variable in programming is a storage location paired with an associated symbolic name that contains a value. Variables can be used to store numbers, text, or more complex data structures. The value of a variable can be changed throughout the program's execution."),
                ("What is object-oriented programming?", "Object-oriented programming (OOP) is a programming paradigm based on the concept of 'objects', which can contain data (attributes) and code (methods). OOP features include encapsulation, inheritance, polymorphism, and abstraction, which help organize code and make it more reusable and maintainable."),
                ("What is a function?", "In programming, a function is a reusable block of code designed to perform a specific task. Functions can take inputs (parameters), process them, and return outputs. They help organize code, reduce repetition, and improve maintainability by breaking complex programs into smaller, manageable pieces."),
                ("What are data structures?", "Data structures are specialized formats for organizing, processing, retrieving and storing data. Common examples include arrays, linked lists, stacks, queues, trees, and hash tables. The choice of data structure affects the efficiency of algorithms and operations performed on the data."),
                ("What is Python?", "Python is a high-level, interpreted programming language known for its readability and simplicity. It emphasizes code readability with its notable use of significant indentation. Python features a dynamic type system and automatic memory management, supporting multiple programming paradigms including procedural, object-oriented, and functional programming."),
                ("What is an algorithm?", "An algorithm is a step-by-step procedure or formula for solving a problem. In programming, algorithms are the foundation for developing efficient and effective solutions. They can be expressed as pseudocode, flowcharts, programming code, or natural language, and are evaluated based on correctness, efficiency, and simplicity."),
                ("What is debugging?", "Debugging is the process of finding and fixing errors, bugs, or unexpected behavior in computer programs. It involves identifying the problem, locating the source of the error, and making necessary corrections. Debugging tools provide features like breakpoints, variable inspection, and step-by-step execution to help programmers track down issues."),
                ("What is a database?", "A database is an organized collection of structured information or data, typically stored electronically in a computer system. Databases are managed using database management systems (DBMS), which provide an interface for creating, querying, updating, and administering databases. Common types include relational, NoSQL, and object-oriented databases.")
            ]
        }
        
        # Process the question word embeddings
        question_keywords = {
            "what": ["what", "explain", "describe", "define", "tell me about"],
            "who": ["who", "person", "people", "individual"],
            "when": ["when", "time", "date", "year", "period"],
            "where": ["where", "location", "place", "country"],
            "why": ["why", "reason", "cause", "purpose"],
            "how": ["how", "process", "method", "way", "technique"]
        }
        
        # Determine question type to help with matching
        question_type = None
        for type, keywords in question_keywords.items():
            if any(keyword in cleaned_question for keyword in keywords):
                question_type = type
                break
                
        debug_log(f"Question type detected: {question_type}")
        
        # Ensure we have data for the requested subject
        if subject not in minimal_qa_by_subject:
            debug_log(f"Subject {subject} not found in predefined QA pairs")
            return f"I don't have information about the subject '{subject}'. Please try Mathematics, Science, History, or Programming."
        
        # Get QA pairs for this specific subject only
        qa_pairs = minimal_qa_by_subject[subject]
        debug_log(f"Using {len(qa_pairs)} QA pairs for subject: {subject}")
        
        # Create vectorizer and transform questions for this subject only
        vectorizer = TfidfVectorizer(stop_words='english')
        questions = [q for q, _ in qa_pairs]
        answers = [a for _, a in qa_pairs]
        
        # First try to find exact matches (case-insensitive)
        for i, q in enumerate(questions):
            if cleaned_question in q.lower():
                debug_log(f"Found exact substring match: '{q}'")
                return answers[i]
            
            # For "what is X" questions, check for more flexible matching
            if "what is" in cleaned_question or "what are" in cleaned_question:
                # Extract the subject of the question
                subject_words = cleaned_question.replace("what is", "").replace("what are", "").strip()
                if subject_words in q.lower():
                    debug_log(f"Found subject match: '{q}' for subject words '{subject_words}'")
                    return answers[i]
        
        # Fit and transform in one step
        question_vectors = vectorizer.fit_transform(questions)
        
        # Transform user question
        user_question_vector = vectorizer.transform([cleaned_question])
        
        # Calculate similarities
        similarities = cosine_similarity(user_question_vector, question_vectors)[0]
        debug_log(f"Calculated {len(similarities)} similarity scores")
        
        # Log top 3 matches
        top_indices = similarities.argsort()[-3:][::-1]
        debug_log("Top 3 matches:")
        for i, idx in enumerate(top_indices):
            debug_log(f"{i+1}. '{questions[idx]}' - Score: {similarities[idx]:.4f}")
        
        # Find best match
        best_match_idx = similarities.argmax()
        best_match_score = similarities[best_match_idx]
        
        debug_log(f"Best match: '{questions[best_match_idx]}' with score: {best_match_score:.4f}")
        
        # Only return a match if similarity is high enough
        if best_match_score > 0.3:
            return answers[best_match_idx]
        else:
            # Generic subject-specific response for no good match
            subject_responses = {
                "Mathematics": "I don't have specific information about that mathematical concept. Please try asking about the Pythagorean theorem, quadratic equations, matrices, calculus, algebra, equations, trigonometry, or geometry.",
                "Science": "I don't have specific information about that scientific concept. Please try asking about photosynthesis, states of matter, the scientific method, cellular respiration, biology, chemistry, physics, or ecology.",
                "History": "I don't have specific information about that historical topic. Please try asking about Albert Einstein, World War II, Rana Pratap Singh, the Renaissance, the Industrial Revolution, Mahatma Gandhi, the Cold War, or the Crusades.",
                "Programming": "I don't have specific information about that programming concept. Please try asking about variables, object-oriented programming, functions, data structures, Python, algorithms, debugging, or databases."
            }
            
            return subject_responses.get(subject, f"I don't have information about that in {subject}. Please try asking something else.")
            
    except Exception as e:
        debug_log(f"Error in subject_specific_get_response: {str(e)}")
        return f"An error occurred: {str(e)}"

# Fix the progress data view to properly display user progress
def get_user_progress_data(username):
    """Retrieve user progress data from the progress file"""
    try:
        progress_file = 'data/user_progress.pkl'
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
        # Check if file exists
        if not os.path.exists(progress_file):
            debug_log(f"Progress file not found: {progress_file}")
            # Initialize with empty data
            with open(progress_file, 'wb') as f:
                pickle.dump({}, f)
            return {}
        
        # Load progress data
        with open(progress_file, 'rb') as f:
            user_progress = pickle.load(f)
            
        if username not in user_progress:
            debug_log(f"No progress data found for user: {username}")
            return {}
            
        return user_progress[username]
    except Exception as e:
        debug_log(f"Error retrieving user progress: {str(e)}")
        return {}

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs("data/users", exist_ok=True)
        os.makedirs("model", exist_ok=True)
        
        # Run the main app
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        # Try to recover by providing a basic interface
        st.markdown("""
        <div style="padding: 20px; background-color: #1e293b; border-radius: 10px; margin: 20px auto; max-width: 600px; text-align: center;">
            <h2 style="color: #e2e8f0;">AI Tutor - Error Recovery Mode</h2>
            <p style="color: #e2e8f0;">We encountered an error while loading the application. Please try refreshing the page.</p>
            <p style="color: #94a3b8; margin-top: 20px;">If the issue persists, please restart the application or contact support.</p>
        </div>
        """, unsafe_allow_html=True) 