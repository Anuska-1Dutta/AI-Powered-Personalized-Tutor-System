import pickle
import os

# Recreate the BasicAITutor class exactly as in app.py
class BasicAITutor:
    """A basic AI tutor that uses a dictionary-based approach instead of ML models"""
    
    def __init__(self):
        self.responses = {
            # Mathematics
            "quadratic equation": "To solve a quadratic equation ax² + bx + c = 0, use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a.",
            "pythagorean theorem": "The Pythagorean theorem states that in a right-angled triangle, a² + b² = c², where c is the hypotenuse.",
            "derivative": "To find the derivative of a function, apply differentiation rules. For example, the derivative of x^n is n*x^(n-1).",
            "algebra": "Algebra is a branch of mathematics dealing with symbols and the rules for manipulating these symbols.",
            "calculus": "Calculus is the mathematical study of continuous change and has two major branches: differential calculus and integral calculus.",
            
            # Science
            "newton": "Newton's laws: 1) Objects at rest stay at rest unless acted upon by a force. 2) F=ma. 3) For every action, there is an equal and opposite reaction.",
            "photosynthesis": "Photosynthesis is the process where plants convert sunlight, water and carbon dioxide into oxygen and glucose.",
            "dna": "DNA replication follows a semi-conservative model where each strand of the original DNA serves as a template for the new strand.",
            "periodic table": "The periodic table is a tabular arrangement of chemical elements, organized on the basis of their atomic numbers and chemical properties.",
            
            # History
            "world war": "World War I was triggered by the assassination of Archduke Franz Ferdinand in 1914, with underlying causes including militarism, alliances, imperialism, and nationalism.",
            "cleopatra": "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt, known for her relationships with Julius Caesar and Mark Antony.",
            "renaissance": "The Renaissance (14th-17th century) was characterized by renewed interest in classical learning, arts, and scientific discovery.",
            
            # Programming
            "for loop": "In Python, write a for loop using: 'for variable in iterable:' followed by indented code to execute.",
            "object-oriented": "OOP is based on 'objects' containing data and code. The four main principles are encapsulation, abstraction, inheritance, and polymorphism.",
            "exception": "In Python, handle exceptions using try-except blocks. Place risky code in the try block, and error handling in the except block.",
            "python": "Python is a high-level, interpreted programming language known for its readability and versatility."
        }
        
        self.fallbacks = {
            "Mathematics": "That's an interesting math question. Mathematics is all about patterns, numbers, and logical reasoning.",
            "Science": "Science is about observation, hypothesis, and testing. I encourage you to apply the scientific method.",
            "History": "History helps us understand how past events shaped our present. I recommend examining primary sources.",
            "Programming": "Programming is about problem-solving through code. Try breaking down the problem into smaller components."
        }
    
    def get_response(self, subject, question):
        question_lower = question.lower()
        
        # Find matching keywords
        for keyword, response in self.responses.items():
            if keyword in question_lower:
                return response
        
        # If no match found, return a subject-specific fallback
        if subject in self.fallbacks:
            return self.fallbacks[subject]
        
        # General fallback
        return "I don't have specific information on that question. Could you ask me something else about " + subject + "?"

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Create and save the BasicAITutor model
basic_model = BasicAITutor()
with open('model/ai_tutor_model.pkl', 'wb') as f:
    pickle.dump(basic_model, f)

print("Model successfully recreated!") 