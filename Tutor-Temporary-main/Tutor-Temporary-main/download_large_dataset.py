import requests
import os
import json
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# Create necessary directories
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Downloading larger educational datasets...")

# URLs for larger educational datasets
DATASET_URLS = {
    "Mathematics": [
        "https://raw.githubusercontent.com/huggingface/datasets/master/datasets/math_qa/sample_data/sample.json",
        "https://raw.githubusercontent.com/hendrycks/math/master/train_sample.json"
    ],
    "Science": [
        "https://raw.githubusercontent.com/allenai/sciq/master/sciq_data/train_sciq.json"
    ],
    "History": [
        "https://raw.githubusercontent.com/manindersingh030/HistoryGPT-Dataset/main/data-sample.json"
    ],
    "Programming": [
        "https://raw.githubusercontent.com/coding-horror/basic-computer-games/master/00_Alternate_Languages/README.md",
        "https://raw.githubusercontent.com/karpathy/minGPT/master/README.md"
    ]
}

# Define base data with high-quality examples for each subject
BASE_DATA = {
    "Mathematics": [
        "What is a quadratic equation?", "A quadratic equation is a second-degree polynomial equation in the form ax² + bx + c = 0, where a, b, and c are constants and a ≠ 0. These equations are fundamental in algebra and are used to model many real-world scenarios. Quadratic equations can have zero, one, or two real solutions, and can be solved using factoring, completing the square, or the quadratic formula. The graph of a quadratic equation is a parabola, which can open upward or downward depending on the sign of the 'a' coefficient.",
        "What are matrices?", "Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are fundamental mathematical objects used in linear algebra and have applications in computer graphics, physics, engineering, statistics, and many other fields. Key operations on matrices include addition, subtraction, multiplication, and finding inverses and determinants. An m×n matrix has m rows and n columns. Matrices can represent linear transformations, systems of linear equations, graphs, and data in machine learning. Special types include identity matrices, diagonal matrices, and symmetric matrices. The study of matrices is central to linear algebra, providing tools for solving complex multi-dimensional problems efficiently."
    ],
    "Science": [
        "What is biology?", "Biology is the scientific study of life and living organisms. It examines the structure, function, growth, origin, evolution, and distribution of living things. Biology is divided into various specialized fields such as anatomy, physiology, botany, zoology, microbiology, genetics, ecology, and more. Each field focuses on different aspects of life, from molecular processes within cells to interactions between entire ecosystems. The core principles of biology include cell theory (all living things are made of cells), evolution (populations change over time through natural selection), genetics (traits are passed from parents to offspring through genes), homeostasis (organisms maintain internal stability), and energy processing (organisms require energy for survival). Understanding biology is essential for advances in medicine, agriculture, environmental conservation, and biotechnology.",
        "What is an amoeba?", "An amoeba is a type of single-celled organism (protozoan) that belongs to the phylum Sarcodina. Characterized by their ability to change shape, amoebas move by extending temporary foot-like projections called pseudopodia ('false feet'). They are typically found in freshwater environments such as ponds, lakes, and slow-moving streams, though some species live in soil or as parasites in animals. Amoebas feed by engulfing food particles through phagocytosis, where the cell membrane surrounds the food and brings it inside the cell within a food vacuole. They reproduce asexually through binary fission, where one cell divides into two identical daughter cells. While most amoebas are harmless, certain species like Entamoeba histolytica can cause diseases such as amoebic dysentery in humans. Amoebas are studied extensively in biology as examples of simple cellular organisms and are important in understanding cellular processes and evolution."
    ],
    "History": [
        "Who was Abraham Lincoln?", "Abraham Lincoln was the 16th President of the United States, serving from March 1861 until his assassination in April 1865. Lincoln led the United States through the American Civil War, preserving the Union, abolishing slavery, strengthening the federal government, and modernizing the U.S. economy. He is remembered for his character, his speeches and letters, and for issuing the Emancipation Proclamation (1863) that began the process of ending slavery. His Gettysburg Address of 1863 became an iconic statement of America's dedication to the principles of nationalism, republicanism, equal rights, liberty, and democracy. Lincoln is consistently ranked by scholars and the public as one of the greatest U.S. presidents. His assassination made him a martyr for the ideals of national unity and equality.",
        "Who was Rana Pratap Singh?", "Maharana Pratap Singh I was a renowned Hindu Rajput king of Mewar, a region in northwestern India in the present-day state of Rajasthan. Born on May 9, 1540, he was the eldest son of Udai Singh II and ruled from 1572 until his death in 1597. Maharana Pratap is widely recognized for his resistance against the expansion of the Mughal Empire under Emperor Akbar and his refusal to submit to Mughal rule, maintaining Mewar's independence. His most famous battle was the Battle of Haldighati in 1576 against Akbar's forces led by Man Singh I of Amber. Although Maharana Pratap was forced to retreat from Haldighati, he never surrendered to the Mughals and continued guerrilla warfare from the hills of Mewar. He is celebrated as a symbol of Rajput valor, patriotism, and the struggle for independence, and is revered as a hero across India for his courage and principles."
    ],
    "Programming": [
        "What is Python?", "Python is a high-level, interpreted programming language known for its readability and versatility. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It features a dynamic type system, automatic memory management, and a comprehensive standard library. Python is widely used in web development, data analysis, artificial intelligence, scientific computing, automation, and many other fields. Its simplicity and readability make it an excellent language for beginners, while its powerful libraries and frameworks make it valuable for advanced developers. Major implementations include CPython (the reference implementation), PyPy, Jython, and IronPython."
    ]
}

def download_large_datasets():
    print("Starting comprehensive dataset download...")
    
    # Initialize the combined dataset with base data
    combined_data = BASE_DATA.copy()
    
    # Keep track of how many items were added
    added_counts = {subject: len(questions)//2 for subject, questions in combined_data.items()}
    
    # Process each subject and URL
    for subject, urls in DATASET_URLS.items():
        if subject not in combined_data:
            combined_data[subject] = []
        
        for url in urls:
            print(f"Downloading data from {url}...")
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Different handling based on file extension or content
                    if url.endswith('.json'):
                        # Try to parse JSON
                        try:
                            data = response.json()
                            
                            # Handle different JSON formats
                            if isinstance(data, list):
                                # List format
                                for item in data[:50]:  # Limit to 50 items per source
                                    try:
                                        if 'question' in item and ('answer' in item or 'correct_answer' in item):
                                            q = item['question']
                                            a = item.get('answer', item.get('correct_answer', ''))
                                            
                                            # Clean and format
                                            if len(q) > 10 and len(a) > 20:  # Minimum lengths
                                                combined_data[subject].append(q)
                                                combined_data[subject].append(a)
                                                added_counts[subject] = added_counts.get(subject, 0) + 1
                                    except:
                                        continue
                            elif isinstance(data, dict):
                                # Dictionary format
                                if 'data' in data and isinstance(data['data'], list):
                                    # Handle nested data structure
                                    for item in data['data'][:50]:
                                        try:
                                            if 'question' in item and ('answer' in item or 'correct_answer' in item):
                                                q = item['question']
                                                a = item.get('answer', item.get('correct_answer', ''))
                                                
                                                # Clean and format
                                                if len(q) > 10 and len(a) > 20:  # Minimum lengths
                                                    combined_data[subject].append(q)
                                                    combined_data[subject].append(a)
                                                    added_counts[subject] = added_counts.get(subject, 0) + 1
                                        except:
                                            continue
                                else:
                                    # Try to extract top-level QA pairs
                                    for k, v in data.items():
                                        if isinstance(v, str) and len(k) > 10 and len(v) > 20:
                                            combined_data[subject].append(k)
                                            combined_data[subject].append(v)
                                            added_counts[subject] = added_counts.get(subject, 0) + 1
                        except json.JSONDecodeError:
                            print(f"Could not parse JSON from {url}")
                    else:
                        # Handle markdown or text content by extracting potential QA pairs
                        content = response.text
                        lines = content.split('\n')
                        
                        question = None
                        answer_lines = []
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            # Detect questions (lines ending with ? or with ## headings)
                            if line.endswith('?') or (line.startswith('##') and len(line) > 5):
                                # If we already have a question and answer, save them
                                if question and answer_lines:
                                    combined_data[subject].append(question)
                                    combined_data[subject].append('\n'.join(answer_lines))
                                    added_counts[subject] = added_counts.get(subject, 0) + 1
                                
                                # Start a new QA pair
                                question = line.lstrip('#').strip()
                                answer_lines = []
                            elif question:  # If we have a question, collect answer lines
                                answer_lines.append(line)
                        
                        # Save the last QA pair
                        if question and answer_lines:
                            combined_data[subject].append(question)
                            combined_data[subject].append('\n'.join(answer_lines))
                            added_counts[subject] = added_counts.get(subject, 0) + 1
                            
                    print(f"Successfully processed data from {url}")
                else:
                    print(f"Failed to download from {url}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
    
    # Add some hard-coded QA pairs for rare topics or specific needs
    special_qa_pairs = [
        ("History", "Who was Rana Pratap Singh?", "Maharana Pratap Singh I was a renowned Hindu Rajput king of Mewar, a region in northwestern India in the present-day state of Rajasthan. Born on May 9, 1540, he was the eldest son of Udai Singh II and ruled from 1572 until his death in 1597. Maharana Pratap is widely recognized for his resistance against the expansion of the Mughal Empire under Emperor Akbar and his refusal to submit to Mughal rule, maintaining Mewar's independence. His most famous battle was the Battle of Haldighati in 1576 against Akbar's forces led by Man Singh I of Amber. Although Maharana Pratap was forced to retreat from Haldighati, he never surrendered to the Mughals and continued guerrilla warfare from the hills of Mewar. He is celebrated as a symbol of Rajput valor, patriotism, and the struggle for independence, and is revered as a hero across India for his courage and principles."),
        ("Mathematics", "What are matrices?", "Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are fundamental mathematical objects used in linear algebra and have applications in computer graphics, physics, engineering, statistics, and many other fields. Key operations on matrices include addition, subtraction, multiplication, and finding inverses and determinants. An m×n matrix has m rows and n columns. Matrices can represent linear transformations, systems of linear equations, graphs, and data in machine learning. Special types include identity matrices, diagonal matrices, and symmetric matrices. The study of matrices is central to linear algebra, providing tools for solving complex multi-dimensional problems efficiently.")
    ]
    
    for subject, question, answer in special_qa_pairs:
        if question not in combined_data[subject]:
            combined_data[subject].append(question)
            combined_data[subject].append(answer)
            added_counts[subject] = added_counts.get(subject, 0) + 1
    
    # Print summary statistics
    total_qa_pairs = sum(len(combined_data[subject])//2 for subject in combined_data)
    print(f"\nDownload complete! Total Q&A pairs in dataset: {total_qa_pairs}")
    for subject in combined_data:
        print(f"  - {subject}: {len(combined_data[subject])//2} pairs (Added {added_counts.get(subject, 0)} new pairs)")
    
    return combined_data

def create_enhanced_model():
    # Download the enhanced dataset
    training_data = download_large_datasets()
    
    try:
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        all_questions = [q for subj in training_data for q in training_data[subj][::2]]
        X = vectorizer.fit_transform(all_questions)
        
        # Save the model components
        model_path = 'model/ai_tutor_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump((vectorizer, X, training_data), f)
        
        print(f"\nEnhanced model successfully created and saved to {model_path}")
        print(f"Dataset contains {len(all_questions)} questions across {len(training_data)} subjects")
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        print("Model creation failed.")

if __name__ == "__main__":
    create_enhanced_model()
    print("Done! Run the app.py file to start the AI Tutor with the enhanced model.") 