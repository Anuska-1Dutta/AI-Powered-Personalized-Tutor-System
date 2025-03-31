import os
import pickle
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

print("Creating enhanced science dataset...")

# Create necessary directories
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Basic biology concepts
biology_qa = [
    ("What is biology?", "Biology is the scientific study of living organisms and their interactions with each other and their environments. It encompasses various specialized fields such as molecular biology, cellular biology, genetics, ecology, evolutionary biology, and physiology."),
    ("What is an algae?", "Algae are a diverse group of aquatic organisms that can conduct photosynthesis. They range from single-celled microalgae to large seaweeds (macroalgae). Unlike land plants, algae lack true roots, stems, and leaves. They are important primary producers in aquatic ecosystems, generating much of Earth's oxygen. Algae can be classified into several groups including green algae, red algae, brown algae, and diatoms. They have various applications in food, biofuel production, and pharmaceuticals."),
    ("What is a cell?", "A cell is the basic structural, functional, and biological unit of all living organisms. Cells are the smallest unit of life and are often called the 'building blocks of life.' There are two main types of cells: prokaryotic cells (like bacteria, without a nucleus) and eukaryotic cells (with a nucleus, found in plants, animals, fungi, and protists). Cells contain organelles that perform specific functions, and they reproduce through cell division."),
    ("What is DNA?", "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix. It carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms. DNA consists of four nucleotide bases: adenine (A), guanine (G), cytosine (C), and thymine (T). The sequence of these bases determines the genetic information, similar to how letters of the alphabet appear in a specific order to form words and sentences."),
    ("What is evolution?", "Evolution is the process by which populations of organisms change over successive generations. The theory of evolution by natural selection, first formulated by Charles Darwin, is the process by which organisms change over time as a result of changes in heritable physical or behavioral traits. Changes that allow an organism to better adapt to its environment will help it survive and reproduce, passing those beneficial traits to offspring. Over time, this process can result in new species."),
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants, algae, and certain bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars. This process takes place primarily in the chloroplasts of plant cells, particularly in the leaves. The overall equation for photosynthesis is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. Photosynthesis consists of light-dependent reactions and light-independent reactions (Calvin cycle). It is essential for life on Earth as it provides oxygen and energy-rich organic compounds."),
    ("What is cellular respiration?", "Cellular respiration is the process by which cells convert nutrients into ATP, water, and carbon dioxide. It's essentially the opposite of photosynthesis, using oxygen to break down glucose and release energy. The process occurs in three main stages: glycolysis (in the cytoplasm), the Krebs cycle (in the mitochondrial matrix), and the electron transport chain (on the inner mitochondrial membrane). The overall equation is: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP (energy). This process is how most cells, including human cells, produce the energy needed for cellular functions."),
    ("What is an ecosystem?", "An ecosystem is a community of living organisms (plants, animals, and microbes) interacting with each other and their physical environment (air, water, soil, weather). Ecosystems can be of various sizes and types, from a small pond to a vast forest. They include biotic factors (living components) and abiotic factors (non-living components) that function together as a unit. Energy typically flows through ecosystems from producers (plants) to consumers (animals) and decomposers (fungi, bacteria). Ecosystems provide various services like nutrient cycling, water purification, and climate regulation."),
    ("What is genetics?", "Genetics is the study of genes, genetic variation, and heredity in living organisms. It examines how traits are passed from parents to offspring and how these traits are expressed. Genes, composed of DNA, are the basic units of heredity. The field includes classical genetics (studying patterns of inheritance), molecular genetics (studying the structure and function of genes at the molecular level), and population genetics (studying genetic variations within populations). Genetics has applications in medicine, agriculture, and forensic science, among others."),
    ("What is a protein?", "Proteins are large, complex molecules essential for the structure, function, and regulation of the body's tissues and organs. They are made up of hundreds or thousands of smaller units called amino acids, which are attached in long chains. There are 20 different types of amino acids that can be combined to make a protein. The sequence of amino acids determines a protein's unique 3D structure and specific function. Proteins perform a vast array of functions including catalyzing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules."),
]

# Chemistry concepts
chemistry_qa = [
    ("What is chemistry?", "Chemistry is the scientific discipline that studies the composition, structure, properties, and changes of matter. It focuses on atoms, ions, and molecules which make up substances and how they interact with each other. Chemistry encompasses understanding the behavior of particles at the atomic and molecular levels to explain observations at the macroscopic level. The five main branches of chemistry are analytical chemistry, biochemistry, inorganic chemistry, organic chemistry, and physical chemistry. Chemistry is vital to many aspects of modern life, from developing new materials and medicines to addressing environmental challenges."),
    ("What is an atom?", "An atom is the basic unit of matter that forms a chemical element. It consists of a dense central nucleus surrounded by a cloud of negatively charged electrons. The nucleus contains positively charged protons and electrically neutral neutrons (except in hydrogen-1, which has no neutrons). The number of protons determines the element's atomic number and its position in the periodic table. Atoms are extremely small, typically around 100 picometers across. Despite their small size, atoms consist mostly of empty space, as the electrons orbit at relatively large distances from the nucleus."),
    ("What is the periodic table?", "The periodic table is a tabular arrangement of chemical elements, organized on the basis of their atomic numbers, electron configurations, and recurring chemical properties. Elements are presented in order of increasing atomic number (the number of protons in an atom's nucleus). The table is arranged in rows (periods) and columns (groups), with elements in the same group having similar chemical properties. The periodic table is one of the most fundamental tools in chemistry, providing a framework for classifying, systematizing, and predicting chemical properties and behaviors of all known elements, as well as those yet to be discovered."),
    ("What is a chemical reaction?", "A chemical reaction is a process where one or more substances (reactants) are transformed into one or more different substances (products). This transformation involves the breaking and forming of chemical bonds between atoms. Chemical reactions are characterized by changes in energy, often appearing as changes in temperature, light emission, or sound. Common types of reactions include synthesis, decomposition, single replacement, double replacement, and combustion. Chemical reactions can be represented by chemical equations, showing the reactants, products, and conditions under which the reaction occurs."),
    ("What is a molecule?", "A molecule is a group of two or more atoms held together by chemical bonds. It is the smallest unit of a chemical compound that can take part in a chemical reaction. Molecules can be simple, consisting of only two atoms like oxygen (O₂), or complex, containing thousands of atoms like proteins. The geometry of molecules—the arrangement of atoms in three-dimensional space—affects their physical and chemical properties. The study of molecules and their structures is central to many fields, including chemistry, physics, biology, and materials science."),
    ("What is an acid?", "An acid is a chemical substance that donates hydrogen ions (H⁺) or protons when dissolved in water, or accepts electrons in chemical reactions. Acids have a sour taste, can change litmus paper from blue to red, and react with bases to form salts. Common acids include hydrochloric acid (HCl), sulfuric acid (H₂SO₄), and acetic acid (CH₃COOH, found in vinegar). Acids are measured using the pH scale, with a pH less than 7 indicating acidity (the lower the pH, the stronger the acid). Acids play important roles in industrial processes, biochemistry, and everyday life."),
    ("What is a base?", "A base is a chemical substance that accepts hydrogen ions (H⁺) or protons in water solutions, or donates electrons in chemical reactions. Bases feel slippery, taste bitter, and can change litmus paper from red to blue. They react with acids to form salts and water. Common bases include sodium hydroxide (NaOH), ammonia (NH₃), and baking soda (sodium bicarbonate, NaHCO₃). Bases are measured using the pH scale, with a pH greater than 7 indicating basicity (the higher the pH, the stronger the base). Bases are used in cleaning products, medications, and various industrial processes."),
    ("What is the pH scale?", "The pH scale is a logarithmic scale used to specify the acidity or basicity (alkalinity) of an aqueous solution. It quantitatively measures the hydrogen ion concentration, [H⁺], with a lower pH indicating a higher concentration of H⁺ ions. The scale typically ranges from 0 to 14, with 7 being neutral (like pure water). Values less than 7 indicate acidity, while values greater than 7 indicate basicity. Each unit on the scale represents a tenfold change in acidity/basicity. The pH scale is crucial in chemistry, biology, agriculture, medicine, and many other fields for measuring and regulating acidity levels."),
    ("What are isotopes?", "Isotopes are variants of a particular chemical element which differ in neutron number, and consequently in nucleon number (mass number). All isotopes of a given element have the same number of protons but different numbers of neutrons. For example, carbon-12, carbon-13, and carbon-14 are isotopes of carbon: each has 6 protons but they have 6, 7, and 8 neutrons respectively. Some isotopes are stable, while others are radioactive and decay over time. Isotopes have applications in dating archaeological materials, medical diagnostics and treatments, and studying chemical and biological processes."),
    ("What is organic chemistry?", "Organic chemistry is the study of the structure, properties, composition, reactions, and preparation of carbon-containing compounds. These compounds include hydrocarbons and their derivatives such as proteins, lipids, carbohydrates, nucleic acids, and many other substances. Organic compounds form the basis of all earthly life and are found in materials like fossil fuels, plastics, pharmaceuticals, and food. Organic chemistry has practical applications in the chemical industry, drug design, biotechnology, and nanotechnology. It is a vast and complex field due to carbon's ability to form stable covalent bonds with many elements."),
]

# Physics concepts
physics_qa = [
    ("What is physics?", "Physics is the natural science that studies matter, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines, with its main goal being to understand how the universe behaves. Physics encompasses a wide range of phenomena, from the smallest subatomic particles to the entire cosmos. The field is divided into several branches, including classical mechanics, thermodynamics, optics, electromagnetism, relativity, and quantum mechanics. Physics forms the foundation for other natural sciences and engineering disciplines."),
    ("What is Newton's first law?", "Newton's first law of motion, also known as the law of inertia, states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force. This principle directly challenges the previously held Aristotelian view that objects naturally seek rest. It explains why objects in space continue moving indefinitely, why passengers feel pushed back when a car accelerates, and why seat belts are essential safety devices. The law highlights the property of inertia - the resistance of any physical object to change its state of motion."),
    ("What is gravity?", "Gravity is the natural force by which objects with mass attract one another. On Earth, gravity gives weight to physical objects and causes them to fall toward the planet's center when dropped. Sir Isaac Newton's law of universal gravitation describes gravity as a force that is directly proportional to the product of the masses and inversely proportional to the square of the distance between them. Albert Einstein's general theory of relativity reframes gravity as a consequence of the curvature of spacetime caused by mass and energy. Gravity plays a crucial role in the formation of stars, planets, and galaxies, and in determining the large-scale structure of the universe."),
    ("What is energy?", "Energy is the quantitative property that is transferred to an object to perform work or to heat the object. Energy exists in various forms such as kinetic (motion), potential (stored), thermal (heat), chemical, electrical, nuclear, and electromagnetic. According to the law of conservation of energy, energy cannot be created or destroyed, only transformed from one form to another. The SI unit of energy is the joule (J). Energy is a fundamental concept in physics and is essential to understanding processes in nature, technology, and daily life, from photosynthesis to the operation of electronic devices."),
    ("What is electricity?", "Electricity is a form of energy resulting from the existence of charged particles such as electrons or protons. Electric current is the flow of electric charge, typically carried by moving electrons in a wire. Electricity has been harnessed for human use since the late 19th century and has become essential to modern society. It powers lights, appliances, industrial machinery, electronics, and communications systems. The study of electricity, called electromagnetism, includes phenomena such as electric fields, magnetic fields, and electromagnetic waves. Electricity is generated from various sources including fossil fuels, nuclear energy, and renewable sources like solar and wind."),
    ("What is quantum mechanics?", "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It differs from classical physics in that energy, momentum, angular momentum, and other quantities are often restricted to discrete values (quantization), objects have characteristics of both particles and waves (wave-particle duality), and there are limits to the precision with which quantities can be measured (Heisenberg's uncertainty principle). Quantum mechanics forms the basis for understanding atomic structure, chemical bonds, superconductivity, and many other phenomena. It has led to practical applications such as lasers, transistors, and magnetic resonance imaging (MRI)."),
    ("What is thermodynamics?", "Thermodynamics is the branch of physics that deals with heat, work, and temperature, and their relation to energy, radiation, and physical properties of matter. The behavior of these quantities is governed by the four laws of thermodynamics. The first law states that energy cannot be created or destroyed, only transformed (conservation of energy). The second law introduces the concept of entropy, indicating that the total entropy of an isolated system always increases over time. The third law states that as temperature approaches absolute zero, the entropy of a system approaches a constant minimum. Thermodynamics has applications in engines, refrigeration, chemical reactions, and many natural processes."),
    ("What is a wave?", "A wave is a disturbance that transfers energy through matter or space, with little or no associated mass transport. Waves are characterized by wavelength, frequency, amplitude, and speed. There are two main types of waves: mechanical waves, which require a medium to travel through (like sound waves), and electromagnetic waves, which can travel through a vacuum (like light). Waves can be transverse (vibration perpendicular to the direction of travel) or longitudinal (vibration parallel to the direction of travel). Wave phenomena include reflection, refraction, diffraction, and interference. The study of waves is fundamental to understanding many aspects of physics, from sound and light to quantum mechanics."),
    ("What is special relativity?", "Special relativity is a physical theory developed by Albert Einstein in 1905. It introduces major changes to classical mechanics by demonstrating that space and time are interwoven into a single continuum known as spacetime. The theory is based on two postulates: the laws of physics are the same for all non-accelerating observers, and the speed of light in a vacuum is the same for all observers, regardless of their relative motion or the motion of the light source. Special relativity has several consequences including time dilation, length contraction, mass-energy equivalence (E=mc²), and the relativity of simultaneity. It has been confirmed by numerous experiments and is essential to modern physics and technologies like GPS."),
    ("What is a magnetic field?", "A magnetic field is a vector field that describes the magnetic influence on moving electric charges, electric currents, and magnetic materials. Magnetic fields surround magnetized materials and electric currents, exerting forces on other nearby magnets and electric currents. Magnetic fields are produced by moving electric charges and the intrinsic magnetic moments of elementary particles associated with a quantum mechanical property called spin. Earth has its own magnetic field, which protects us from solar wind and helps certain animals navigate. Magnetic fields have practical applications in electric motors, generators, transformers, MRI machines, and data storage devices. They are described mathematically using field lines and measured in units of tesla (T) or gauss (G)."),
]

# Earth Science concepts
earth_science_qa = [
    ("What is Earth science?", "Earth science is a comprehensive term for the sciences related to the planet Earth. It is also known as geoscience and encompasses geology, oceanography, meteorology, climatology, environmental science, and astronomy as it relates to Earth. Earth scientists use tools from physics, chemistry, biology, and mathematics to build a quantitative understanding of how the Earth works and evolves. The study of Earth science helps us understand natural disasters, utilize natural resources, and protect the environment. It provides insights into Earth's history spanning 4.5 billion years and helps predict future changes in climate and landforms."),
    ("What is geology?", "Geology is the science that deals with the Earth's physical structure and substance, its history, and the processes that act on it. Geologists study rocks, minerals, and the forces that shape the landscape, such as earthquakes, volcanoes, and erosion. The field is divided into various specialties including mineralogy, petrology, structural geology, geomorphology, paleontology, stratigraphy, and economic geology. Geology has practical applications in mining, petroleum exploration, groundwater management, civil engineering, and environmental protection. Understanding geological processes helps predict natural hazards and locate natural resources."),
    ("What is meteorology?", "Meteorology is the scientific study of the atmosphere, particularly focused on weather processes and forecasting. Meteorologists use scientific principles to understand atmospheric conditions, study weather patterns, and make predictions about the weather. They analyze data from satellites, radars, and weather stations to create forecasts. The field includes several subfields such as climate studies, atmospheric physics, and atmospheric chemistry. Meteorology has important applications in weather forecasting, climate change research, air quality monitoring, and agriculture. It helps prepare for and mitigate the impacts of severe weather events like hurricanes, tornadoes, and floods."),
    ("What is oceanography?", "Oceanography is the study of the physical, chemical, and biological features of the ocean, including the ocean's ancient history, its current condition, and its future. It encompasses four main areas: physical oceanography (the study of waves, currents, tides, and ocean circulation), chemical oceanography (the study of the composition of seawater), geological oceanography (the study of the ocean floor, including its mountains, valleys, and plains), and biological oceanography (the study of marine organisms and their interactions with the ocean environment). Oceanography is crucial for understanding climate regulation, resource management, coastal protection, and marine conservation."),
    ("What are tectonic plates?", "Tectonic plates are large sections of the Earth's lithosphere (crust and upper mantle) that move relative to one another. The Earth's surface is divided into about 15-20 major plates and several minor ones. These plates float on the semi-fluid asthenosphere beneath them. The boundaries between plates are sites of intense geological activity, including earthquakes, volcanoes, and mountain building. There are three types of plate boundaries: convergent (where plates move toward each other), divergent (where plates move apart), and transform (where plates move alongside each other). The theory of plate tectonics explains various geological phenomena including the formation of mountain ranges, ocean trenches, and the distribution of earthquakes and volcanoes."),
    ("What is the water cycle?", "The water cycle, also known as the hydrologic cycle, is the continuous movement of water on, above, and below the Earth's surface. It involves the processes of evaporation (water turning into vapor), transpiration (water vapor released from plants), condensation (water vapor turning into liquid water), precipitation (water falling from clouds), infiltration (water soaking into the ground), and runoff (water flowing over land). The water cycle is driven by solar energy and gravity. It is essential for maintaining Earth's water supply, regulating weather patterns and climate, shaping landscapes through erosion and deposition, and supporting all life on Earth. Human activities like deforestation, urbanization, and climate change can disrupt the natural water cycle."),
    ("What is climate change?", "Climate change refers to significant, long-term changes in the global climate. The term is now generally used to describe the changes in Earth's climate primarily driven by human activities since the pre-industrial period (mid-18th century), particularly the burning of fossil fuels, which releases carbon dioxide and other greenhouse gases into the atmosphere. These emissions enhance the natural greenhouse effect, trapping more heat in the Earth's atmosphere and causing global warming. Effects of climate change include rising temperatures, changing precipitation patterns, more frequent and intense extreme weather events, sea level rise, ocean acidification, and impacts on biodiversity. Addressing climate change requires both mitigation (reducing greenhouse gas emissions) and adaptation (adjusting to current and future climate impacts)."),
    ("What is a fossil?", "A fossil is the preserved remains or traces of an organism from a past geological age. Fossils can include bones, shells, imprints, tracks, and other preserved evidence of ancient life. They typically form when an organism dies and is quickly buried in sediment, which protects it from decay and scavengers. Over time, minerals seep into the remains, gradually replacing the organic material in a process called permineralization. Fossils provide valuable evidence about Earth's history, the evolution of life, ancient environments, and past climates. They are typically found in sedimentary rock and are studied by paleontologists. The oldest known fossils date back to about 3.5 billion years ago."),
    ("What is a volcano?", "A volcano is a rupture in the Earth's crust that allows hot magma, ash, and gases to escape from the magma chamber beneath the surface. Volcanoes are typically found at the boundaries of tectonic plates but can also occur at hotspots where mantle plumes bring magma to the surface. When a volcano erupts, it can release lava (molten rock), pyroclastic flows (hot gas and rock), ash, and various gases. Volcanoes are classified by their shape, composition, and eruption pattern, with major types including shield volcanoes, stratovolcanoes (composite volcanoes), and cinder cones. While volcanic eruptions can be destructive, they also create new land, provide rich soil for agriculture, and can be harnessed for geothermal energy."),
    ("What is a hurricane?", "A hurricane is a type of tropical cyclone, which is a rotating low-pressure weather system with organized thunderstorms but no fronts (boundaries between air masses). Hurricanes form over warm ocean waters (at least 26°C or 79°F) near the equator. As warm, moist air rises from the ocean surface, it creates an area of low pressure below. Air from surrounding areas fills the low pressure, warms, and rises, creating a cycle that fuels the storm. When wind speeds reach 74 mph (119 km/h) or higher, the storm is classified as a hurricane in the Atlantic and Northeast Pacific. These powerful storms are characterized by a well-defined center (eye), a circular rotation, and strong winds. Hurricanes can cause severe damage through high winds, heavy rainfall, storm surges, and flooding. They are called typhoons in the Northwest Pacific and cyclones in the South Pacific and Indian Ocean."),
]

# Combined science QA pairs
science_qa_pairs = biology_qa + chemistry_qa + physics_qa + earth_science_qa

# Try to download additional data from external sources
try:
    print("Attempting to download additional science datasets...")
    
    # Try to fetch additional science content
    science_urls = [
        "https://raw.githubusercontent.com/allenai/sciq/master/sciq_sample.json",
        "https://raw.githubusercontent.com/wiki/google-research/bert/squad-sample.json"
    ]
    
    for url in science_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        for item in data[:20]:  # Limit to first 20 items
                            if 'question' in item and ('answer' in item or 'correct_answer' in item):
                                q = item.get('question', '')
                                a = item.get('answer', item.get('correct_answer', ''))
                                
                                if len(q) > 10 and len(a) > 20:  # Ensure reasonable length
                                    science_qa_pairs.append((q, a))
                    elif isinstance(data, dict) and 'data' in data:
                        for item in data.get('data', [])[:20]:
                            for paragraph in item.get('paragraphs', [])[:5]:
                                for qa in paragraph.get('qas', [])[:2]:
                                    question = qa.get('question', '')
                                    if 'answers' in qa and qa['answers']:
                                        answer = qa['answers'][0].get('text', '')
                                        if len(question) > 10 and len(answer) > 20:
                                            science_qa_pairs.append((question, answer))
                    
                    print(f"Successfully processed data from {url}")
                except json.JSONDecodeError:
                    print(f"Could not parse JSON from {url}")
            else:
                print(f"Failed to download from {url}, status code: {response.status_code}")
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
    
    # Add fallback science topics as a backup
    science_topics = [
        "atom", "molecule", "cell", "gene", "solar system", 
        "chemical reaction", "force", "energy", "ecosystem", "climate"
    ]
    
    for topic in science_topics:
        question = f"What is a {topic} in science?"
        answer = f"In science, a {topic} is a fundamental concept that helps explain natural phenomena and the physical world. It is studied across various scientific disciplines and contributes to our understanding of how the universe works."
        science_qa_pairs.append((question, answer))
    
    print(f"Added {len(science_topics)} basic science topic definitions")

except Exception as e:
    print(f"Error downloading additional datasets: {str(e)}")
    print("Using only the predefined QA pairs...")

# Create a science-specific dataset
print(f"Creating science dataset with {len(science_qa_pairs)} QA pairs")

# Prepare the dataset structure
science_training_data = {"Science": []}
for question, answer in science_qa_pairs:
    science_training_data["Science"].append(question)
    science_training_data["Science"].append(answer)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
all_questions = [q for q in science_training_data["Science"][::2]]
X = vectorizer.fit_transform(all_questions)

# Save the model components
model_path = 'model/science_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump((vectorizer, X, science_training_data), f)

print(f"Science model successfully created and saved to {model_path}")
print(f"Dataset contains {len(all_questions)} questions")

# Optionally, merge this with existing AI tutor model if it exists
try:
    existing_model_path = 'model/large_ai_tutor_model.pkl'
    if os.path.exists(existing_model_path):
        print("Found existing large AI tutor model, merging science data...")
        
        with open(existing_model_path, 'rb') as f:
            existing_model_data = pickle.load(f)
            
        if isinstance(existing_model_data, tuple) and len(existing_model_data) == 3:
            combined_vectorizer, combined_X, combined_training_data = existing_model_data
            
            # Merge science data with existing data
            combined_training_data["Science"] = science_training_data["Science"]
            
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
    else:
        print("Large AI tutor model not found. Creating new combined model...")
        # Try to find standard AI tutor model
        standard_model_path = 'model/ai_tutor_model.pkl'
        if os.path.exists(standard_model_path):
            with open(standard_model_path, 'rb') as f:
                standard_model_data = pickle.load(f)
                
            if isinstance(standard_model_data, tuple) and len(standard_model_data) == 3:
                _, _, standard_training_data = standard_model_data
                
                # Create combined data
                combined_training_data = standard_training_data.copy()
                combined_training_data["Science"] = science_training_data["Science"]
                
                # Create vectorizer and matrix
                combined_questions = []
                for subject, qa_list in combined_training_data.items():
                    combined_questions.extend(qa_list[::2])
                    
                new_vectorizer = TfidfVectorizer()
                new_X = new_vectorizer.fit_transform(combined_questions)
                
                # Save as large model
                with open(existing_model_path, 'wb') as f:
                    pickle.dump((new_vectorizer, new_X, combined_training_data), f)
                    
                print(f"New combined model created at {existing_model_path}")
                print(f"Combined dataset contains {len(combined_questions)} questions across {len(combined_training_data)} subjects")
except Exception as e:
    print(f"Error merging with existing model: {str(e)}")
    print("Science-only model was still created successfully.")

print("Done! Run the app.py file to start using the enhanced science model.") 