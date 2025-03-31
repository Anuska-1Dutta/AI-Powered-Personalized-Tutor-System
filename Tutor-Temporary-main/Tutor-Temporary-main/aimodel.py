import os
import pickle
import random
import re
import logging
from datetime import datetime
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join('logs', f'ai_tutor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    filemode='w'
)
logger = logging.getLogger(__name__)

class AITutor:
    def __init__(self, model_path='model/ai_tutor_model.pkl'):
        self.model_path = model_path
        self.data = self.load_model()
        self.chat_history = {}
        self.current_context = {}
        self.recent_responses = {}  # Track recent responses to avoid repetition
        logger.info("AI Tutor initialized")
        
    def load_model(self):
        """Load the model from pickle file"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Model loaded successfully from {self.model_path}")
                logger.info(f"Model data contains {len(data)} entries")
                return data
            else:
                logger.warning(f"Model file not found at {self.model_path}. Using default data.")
                # Fallback data
                return self._create_default_data()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return self._create_default_data()
    
    def _create_default_data(self):
        """Create basic data if model is missing"""
        logger.info("Creating default data")
        return {
            "mathematics": {
                "what is algebra": "Algebra is a branch of mathematics dealing with symbols and the rules for manipulating these symbols. In elementary algebra, those symbols (today written as Latin and Greek letters) represent quantities without fixed values, known as variables.",
                "explain pythagoras theorem": "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides. It is represented by the equation: a² + b² = c², where c is the length of the hypotenuse and a and b are the lengths of the other two sides."
            },
            "science": {
                "what is photosynthesis": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars. Plants take in carbon dioxide and water, and with the energy from sunlight, convert them into glucose and oxygen.",
                "explain atoms": "Atoms are the basic units of matter and the defining structure of elements. The term 'atom' comes from the Greek word for indivisible because it was once thought that atoms were the smallest things in the universe and could not be divided. We now know that atoms are composed of three particles: protons, neutrons, and electrons."
            },
            "history": {
                "who was abraham lincoln": "Abraham Lincoln was the 16th President of the United States, serving from March 1861 until his assassination in April 1865. Lincoln led the United States through the American Civil War, preserved the Union, abolished slavery, strengthened the federal government, and modernized the U.S. economy.",
                "what was the renaissance": "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It was characterized by an effort to revive and surpass ideas and achievements of classical antiquity. It saw developments in art, architecture, politics, science and literature with figures like Leonardo da Vinci and Michelangelo."
            },
            "programming": {
                "what is python": "Python is a high-level, interpreted programming language known for its readability and simplicity. It emphasizes code readability with its notable use of significant whitespace. Python features a dynamic type system and automatic memory management, supporting multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "explain object oriented programming": "Object-Oriented Programming (OOP) is a programming paradigm based on the concept of 'objects', which can contain data and code. Data is in the form of fields (attributes or properties), and code is in the form of procedures (methods). A key feature of OOP is that object's procedures can access and modify the data fields of the object they are associated with."
            }
        }
    
    def get_response(self, question, subject):
        """Generate a response to a user question about a specific subject."""
        try:
            # Log the incoming question and subject
            logger.info(f"Received question: '{question}' for subject: '{subject}'")
            
            # Clean the question
            cleaned_question = self._clean_text(question)
            
            # Check if there's exact subject data
            if subject.lower() not in self.data and subject.lower() not in [s.lower() for s in self.data.keys()]:
                logger.warning(f"Subject '{subject}' not found in model data. Using general knowledge.")
                # If subject not found, search across all subjects
                found_response = self._search_across_subjects(cleaned_question)
                if found_response:
                    return found_response
                return self._generate_fallback_response(cleaned_question, subject)
            
            # Convert to lowercase for case-insensitive matching
            subject_key = subject.lower()
            
            # Ensure subject key exists (handling case sensitivity)
            for key in self.data.keys():
                if key.lower() == subject_key:
                    subject_key = key
                    break
            
            # First, try for exact match
            if cleaned_question in self.data[subject_key]:
                logger.info(f"Found exact match for question in {subject_key}")
                response = self.data[subject_key][cleaned_question]
                
                # Check if this is a repeat of the most recent response
                recent_key = f"{subject_key}:{cleaned_question}"
                if recent_key in self.recent_responses:
                    logger.info(f"Avoiding repetition of exact response")
                    # Try to find alternative or add disclaimer
                    return self._find_alternative_response(self.data[subject_key][cleaned_question], subject_key)
                
                # Record this response as recently used
                self.recent_responses[recent_key] = response
                return response
            
            # Check for special "who" questions about people
            if cleaned_question.startswith("who is") or cleaned_question.startswith("who was"):
                person_name = cleaned_question.replace("who is", "").replace("who was", "").strip()
                logger.info(f"Processing 'who' question about: {person_name}")
                
                # Search for the person's name in the data
                person_response = self._search_for_person(person_name, subject_key)
                if person_response:
                    logger.info(f"Found information about {person_name}")
                    return person_response
            
            # For "what is" questions, look for keyword matches
            if cleaned_question.startswith("what is") or cleaned_question.startswith("what are"):
                entity = cleaned_question.replace("what is", "").replace("what are", "").strip()
                logger.info(f"Processing 'what is' question about: {entity}")
                
                # Search for the entity in the data
                entity_response = self._search_for_entity(entity, subject_key)
                if entity_response:
                    logger.info(f"Found information about {entity}")
                    return entity_response
            
            # Try fuzzy matching if no exact match found
            response = self._fuzzy_match(cleaned_question, subject_key)
            if response:
                logger.info(f"Found fuzzy match for question in {subject_key}")
                return response
            
            # If no match found, try to generate based on similar questions
            logger.info(f"No match found, trying to generate response")
            return self._generate_response(cleaned_question, subject_key)
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your question. {str(e)}"
    
    def _clean_text(self, text):
        """Clean text by removing punctuation and converting to lowercase."""
        text = text.lower()
        # Remove question marks, periods, etc.
        text = re.sub(r'[^\w\s]', '', text).strip()
        return text
    
    def _search_across_subjects(self, question):
        """Search for an answer across all subjects."""
        logger.info(f"Searching across all subjects for: '{question}'")
        
        for subject, qa_pairs in self.data.items():
            if question in qa_pairs:
                logger.info(f"Found match in {subject}")
                return qa_pairs[question]
            
            # Try keyword matching for non-exact matches
            for q, a in qa_pairs.items():
                # Check if all words in the question are in the stored question
                question_words = set(question.split())
                stored_question_words = set(q.split())
                
                # If at least 70% of words match
                intersection = question_words.intersection(stored_question_words)
                if len(intersection) >= 0.7 * len(question_words):
                    logger.info(f"Found partial match in {subject}: {q}")
                    return a
        
        logger.info("No match found across subjects")
        return None
    
    def _search_for_person(self, person_name, subject_key):
        """Search for information about a person in the data."""
        person_name = person_name.lower()
        
        # Look across all QA pairs in the subject
        for question, answer in self.data[subject_key].items():
            # Check if person's name is in the question
            if person_name in question.lower():
                return answer
            
            # Also check if the person's name is in the answer
            if person_name in answer.lower():
                # Check that this is biographical information
                biographical_indicators = ["born", "was a", "is a", "died", "lived", "known for"]
                if any(indicator in answer.lower() for indicator in biographical_indicators):
                    return answer
        
        # If not found in specific subject, try across all subjects
        for subj, qa_pairs in self.data.items():
            if subj == subject_key:
                continue  # Already checked
                
            for question, answer in qa_pairs.items():
                if person_name in question.lower() or person_name in answer.lower():
                    # Check that this is biographical information
                    biographical_indicators = ["born", "was a", "is a", "died", "lived", "known for"]
                    if any(indicator in answer.lower() for indicator in biographical_indicators):
                        return f"While this is not strictly {subject_key}, I can tell you that: {answer}"
        
        return None
    
    def _search_for_entity(self, entity, subject_key):
        """Search for information about an entity (concept, thing) in the data."""
        entity = entity.lower()
        
        # Look across all QA pairs in the subject
        for question, answer in self.data[subject_key].items():
            # Check if entity is in the question
            if entity in question.lower() and ("what is" in question.lower() or "what are" in question.lower()):
                return answer
            
            # Also check if the entity is prominently in the answer
            if entity in answer.lower().split()[:10]:  # Check if entity is in the first few words
                definitional_indicators = ["is a", "are a", "refers to", "defined as"]
                if any(indicator in answer.lower() for indicator in definitional_indicators):
                    return answer
        
        # If not found in specific subject, try across all subjects
        for subj, qa_pairs in self.data.items():
            if subj == subject_key:
                continue  # Already checked
                
            for question, answer in qa_pairs.items():
                if entity in question.lower() and ("what is" in question.lower() or "what are" in question.lower()):
                    return f"While this is not strictly {subject_key}, I can tell you that: {answer}"
        
        return None
    
    def _fuzzy_match(self, question, subject_key):
        """Find a response using fuzzy matching."""
        best_match = None
        highest_score = 0
        
        for q in self.data[subject_key]:
            # Simple word overlap score
            q_words = set(q.split())
            question_words = set(question.split())
            common_words = q_words.intersection(question_words)
            
            # Skip if no common meaningful words
            if len(common_words) < 2:
                continue
            
            # Calculate score: ratio of common words to total unique words
            score = len(common_words) / len(q_words.union(question_words))
            
            if score > highest_score and score > 0.5:  # Threshold of 50% similarity
                highest_score = score
                best_match = q
        
        if best_match:
            logger.info(f"Fuzzy match found with score {highest_score}: {best_match}")
            
            # Check if we've recently used this response
            recent_key = f"{subject_key}:{best_match}"
            if recent_key in self.recent_responses:
                logger.info(f"Avoiding repetition of fuzzy match response")
                # Try to find alternative or add disclaimer
                return self._find_alternative_response(self.data[subject_key][best_match], subject_key)
            
            # Record this response as recently used
            self.recent_responses[recent_key] = self.data[subject_key][best_match]
            return self.data[subject_key][best_match]
        
        return None
    
    def _find_alternative_response(self, original_response, subject_key):
        """Find an alternative response or modify the original to avoid repetition."""
        # Try to find a different response with similar content
        for q, a in self.data[subject_key].items():
            if a != original_response and self._similarity_score(a, original_response) > 0.3:
                return f"Another perspective on this: {a}"
        
        # If no alternative found, add a note to the original response
        variations = [
            "As I mentioned earlier, ",
            "To reiterate, ",
            "Just to confirm what I said before, ",
            "To expand on my previous answer, "
        ]
        prefix = random.choice(variations)
        return f"{prefix}{original_response}"
    
    def _similarity_score(self, text1, text2):
        """Calculate a simple similarity score between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _generate_response(self, question, subject_key):
        """Generate a response based on the question and available data."""
        # Parse the question to identify the topic
        question_words = question.split()
        content_words = [w for w in question_words if w not in ['what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with']]
        
        if not content_words:
            return self._generate_fallback_response(question, subject_key)
        
        # Search for related answers based on content words
        potential_answers = []
        
        for q, a in self.data[subject_key].items():
            for word in content_words:
                if word in q or word in a:
                    potential_answers.append((q, a, self._calculate_relevance(word, q, a)))
        
        # Sort by relevance score
        potential_answers.sort(key=lambda x: x[2], reverse=True)
        
        if potential_answers:
            logger.info(f"Generated response based on keywords: {content_words}")
            
            # Take the most relevant answer
            most_relevant = potential_answers[0][1]
            
            # Check if we've recently used this response
            recent_key = f"{subject_key}:generated_{' '.join(content_words)}"
            if recent_key in self.recent_responses and self.recent_responses[recent_key] == most_relevant:
                # If it's a repeat, try the second most relevant if available
                if len(potential_answers) > 1:
                    logger.info(f"Using alternative response to avoid repetition")
                    most_relevant = potential_answers[1][1]
            
            # Record this response as recently used
            self.recent_responses[recent_key] = most_relevant
            return most_relevant
        
        # If no related answers found, provide a fallback response
        return self._generate_fallback_response(question, subject_key)
    
    def _calculate_relevance(self, keyword, question, answer):
        """Calculate the relevance score of a question-answer pair to a keyword."""
        score = 0
        
        # Check keyword in question
        if keyword in question:
            score += 3  # Higher weight for question match
        
        # Check keyword in answer
        if keyword in answer:
            score += 2  # Lower weight for answer match
        
        # Add weight based on answer length (prefer more detailed answers)
        score += min(len(answer.split()) / 50, 1)  # Normalize by max length
        
        return score
    
    def _generate_fallback_response(self, question, subject):
        """Generate a fallback response when no good match is found."""
        logger.info(f"Using fallback response for '{question}' in {subject}")
        
        # Generic responses specific to subjects
        subject_responses = {
            "mathematics": [
                "I don't have specific information about that mathematical concept. Mathematics includes various branches like algebra, calculus, geometry, and statistics, each with their own principles and applications.",
                "That's an interesting mathematical question. Mathematics is a field that studies numbers, quantities, shapes, patterns, and logical reasoning."
            ],
            "science": [
                "I don't have detailed information about that scientific concept. Science encompasses fields like physics, chemistry, biology, and more, each studying different aspects of the natural world.",
                "That's an interesting scientific question. Science is the systematic study of the structure and behavior of the physical and natural world through observation and experiment."
            ],
            "history": [
                "I don't have specific information about that historical event or figure. History records and analyzes past events, particularly human activities and their impacts on society and civilization.",
                "That's an interesting historical question. History helps us understand our past, which shapes our present and influences our future."
            ],
            "programming": [
                "I don't have specific details about that programming concept. Programming involves creating instructions for computers to follow, using various languages and paradigms.",
                "That's an interesting programming question. Programming is the process of creating sets of instructions that tell a computer how to perform specific tasks."
            ]
        }
        
        # Default fallback if subject not recognized
        default_responses = [
            f"I don't have specific information about that in {subject}. Can you ask something else or try rephrasing your question?",
            f"I'm not sure about that specific topic in {subject}. Could you provide more context or ask about a related concept?",
            f"That's an interesting question about {subject}, but I don't have enough information to provide a complete answer. Can I help with something else?"
        ]
        
        # Get appropriate responses for the subject
        appropriate_responses = subject_responses.get(subject.lower(), default_responses)
        
        # Choose a random response
        return random.choice(appropriate_responses)
    
    def save_chat_history(self, username, subject, question, answer):
        """Save a chat interaction to the user's history."""
        if username not in self.chat_history:
            self.chat_history[username] = {}
        
        if subject not in self.chat_history[username]:
            self.chat_history[username][subject] = []
        
        self.chat_history[username][subject].append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        logger.info(f"Saved chat history for {username} in {subject}")
    
    def get_chat_history(self, username, subject=None):
        """Get the chat history for a user, optionally filtered by subject."""
        if username not in self.chat_history:
            return []
        
        if subject:
            return self.chat_history[username].get(subject, [])
        
        # Combine all subjects if none specified
        all_history = []
        for subj, history in self.chat_history[username].items():
            for entry in history:
                entry_with_subject = entry.copy()
                entry_with_subject["subject"] = subj
                all_history.append(entry_with_subject)
        
        # Sort by timestamp
        all_history.sort(key=lambda x: x["timestamp"])
        return all_history
    
    def save_user_progress(self, username, subject, progress_data, progress_file='data/user_progress.pkl'):
        """Save a user's learning progress."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            
            # Load existing progress data
            user_progress = {}
            if os.path.exists(progress_file):
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
            
            # Update progress data
            session_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "questions": progress_data.get("questions", []),
                "duration": progress_data.get("duration", 0)
            }
            
            user_progress[username][subject]["sessions"].append(session_data)
            user_progress[username][subject]["last_session"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_progress[username][subject]["questions_asked"] += len(progress_data.get("questions", []))
            
            # Update mastery level - simple algorithm based on number of questions
            questions_asked = user_progress[username][subject]["questions_asked"]
            mastery_thresholds = {10: 20, 25: 40, 50: 60, 100: 80, 200: 95}
            
            for threshold, level in mastery_thresholds.items():
                if questions_asked >= threshold:
                    user_progress[username][subject]["mastery_level"] = level
            
            # Save updated progress
            with open(progress_file, 'wb') as f:
                pickle.dump(user_progress, f)
            
            logger.info(f"Saved progress for {username} in {subject}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving user progress: {str(e)}")
            return False
    
    def get_user_progress(self, username, progress_file='data/user_progress.pkl'):
        """Get a user's learning progress across all subjects."""
        try:
            if not os.path.exists(progress_file):
                logger.warning(f"Progress file not found: {progress_file}")
                return {}
            
            with open(progress_file, 'rb') as f:
                user_progress = pickle.load(f)
            
            if username not in user_progress:
                logger.info(f"No progress data found for user: {username}")
                return {}
            
            return user_progress[username]
        
        except Exception as e:
            logger.error(f"Error loading user progress: {str(e)}")
            return {} 