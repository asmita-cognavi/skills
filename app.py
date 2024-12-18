import os
from dotenv import load_dotenv
import streamlit as st
import spacy
import pandas as pd
import numpy as np
import faiss
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
import re

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

def clean_response(response):
    # Use regex to remove everything except letters and spaces
    cleaned_response = re.sub(r'[^a-zA-Z\s]', '', response)
    cleaned_response=cleaned_response.lower()
    return cleaned_response.strip()

# Check if the API key is set correctly
if openai_api_key is None:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
else:
    # Define the skill classification prompt template for LangChain
    skill_classification_prompt = """
    Classify the following skill into 'hard vs soft' and 'cognitive, behavioral, or technical'. 
    Provide the results in the format: 
    'hard vs soft: <hard/soft>, cognitive/behavioral/technical: <cognitive/behavioral/technical>'.
    Skill: {skill}
    """

    # Define LangChain components
    prompt_template = PromptTemplate(input_variables=["skill"], template=skill_classification_prompt)
    llm = OpenAI(temperature=0.2, openai_api_key=openai_api_key)  # Pass the API key here
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    # Function to classify skill types using LangChain
    def classify_skill_types(skill):
        try:
            # Use LangChain to generate the classification response
            response = llm_chain.run(skill=skill)

            # Parsing the response (assumes format: 'hard vs soft: <hard/soft>, cognitive/behavioral/technical: <cognitive/behavioral/technical>')
            type1, type2 = response.split(", ")
            type1 = clean_response(type1.split(":")[1].strip())
            type2 = clean_response(type2.split(":")[1].strip())
            type1=type1[0].upper()+type1[1:]
            type2=type2[0].upper()+type2[1:]

            return type1, type2
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None, None

# Define file paths for embeddings and index
TAXONOMY_FILE = "taxonomy_df.csv"

# Define embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2": "all-MiniLM-L12-v2",
    "distilbert-base-nli-stsb-mean-tokens": "distilbert-base-nli-stsb-mean-tokens",
    "facebook-dpr-ctx_encoder-multiset-base": "facebook-dpr-ctx_encoder-multiset-base",
    "all-mpnet-base-v2":"all-mpnet-base-v2"
}

# Load taxonomy data
skills_taxonomy_df = pd.read_csv(TAXONOMY_FILE)
skills = list(skills_taxonomy_df["skill"].unique())

# Initialize spaCy and PhraseMatcher
nlp = spacy.blank("en")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in skills]
matcher.add("SKILLS", patterns)

# Function to load embedding models
@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

# Function to preprocess text (with stopword removal)
def preprocess_text(text):
    doc = nlp(text.lower())
    stopwords = nlp.Defaults.stop_words
    tokens = [token.text for token in doc if token.is_alpha and token.text not in stopwords]
    return " ".join(tokens)

# Function to find skill categories
def find_skill_categories(df, skills_list):
    df['skill'] = df['skill'].str.lower()
    filtered_df = df[df['skill'].isin(skills_list)]
    result_df = filtered_df[['skill', 'category', 'subcategory']]
    return result_df.to_dict(orient='records') if not result_df.empty else []

# Function to extract skills
def extract_skills(model_name, text):
    embedding_model = load_model(model_name)
    skill_embeddings = embedding_model.encode(skills, normalize_embeddings=True)

    # Build FAISS index
    dimension = skill_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(skill_embeddings)

    # Preprocess text
    cleaned_text = preprocess_text(text)
    doc = nlp(cleaned_text)

    # Phrase matching
    exact_matches = [doc[start:end].text for _, start, end in matcher(doc)]

    # Extract unmatched phrases
    unmatched_text = cleaned_text
    for match in exact_matches:
        unmatched_text = unmatched_text.replace(match, "")
    unmatched_phrases = [token.text for token in nlp(unmatched_text) if token.text.strip()]

    # Semantic matching
    semantic_matches = []
    match_details = []
    remaining_embeddings = embedding_model.encode(unmatched_phrases, normalize_embeddings=True)
    k = 5  # Top-k matches
    for i, phrase in enumerate(unmatched_phrases):
        embedding = remaining_embeddings[i].reshape(1, -1)
        distances, indices = index.search(embedding, k)
        for j in range(k):
            if distances[0][j] > 0.6:  # Threshold
                matched_skill = skills[indices[0][j]]
                # Classify matched skill using LangChain
                type1, type2 = classify_skill_types(matched_skill)
                # Append the match and details directly under the semantic matches
                semantic_matches.append({
                    "input": phrase,
                    "matched_skill": matched_skill.lower(),
                    "category": find_skill_categories(skills_taxonomy_df, [matched_skill.lower()])[0]["category"],
                    "subcategory": find_skill_categories(skills_taxonomy_df, [matched_skill.lower()])[0]["subcategory"],
                    "hard_soft": type1,
                    "broad skill": type2
                })

    # Now process exact matches similarly with LangChain classification
    exact_match_results = []
    for match in set(exact_matches):
        type1, type2 = classify_skill_types(match)
        exact_match_results.append({
            "skill": match,
            "category": find_skill_categories(skills_taxonomy_df, [match.lower()])[0]["category"],
            "subcategory": find_skill_categories(skills_taxonomy_df, [match.lower()])[0]["subcategory"],
            "hard_soft": type1,
            "broad skill": type2
        })

    results = {
        "exact_matches": exact_match_results,
        "semantic_matches": semantic_matches
    }

    return results

# Visualization function
# def plot_model_comparison(model_results):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for model, results in model_results.items():
#         counts = {
#             "Exact Matches": len(results["exact_matches"]),
#             "Semantic Matches": len(results["semantic_matches"]),
#         }
#         ax.bar(counts.keys(), counts.values(), alpha=0.7, label=model)
#     ax.set_title("Skill Match Counts by Model")
#     ax.set_ylabel("Count")
#     ax.legend()
#     st.pyplot(fig)

# Streamlit App
st.title("Skill Categorization and Model Comparison")

# Input text
text_input = st.text_area("Enter your text:")

# Select models
selected_models = st.multiselect("Select Embedding Models to Compare", options=list(EMBEDDING_MODELS.keys()), default=["all-MiniLM-L6-v2"])

if st.button("Categorize Skills and Compare Models"):
    if text_input.strip():
        model_results = {}
        for model_name in selected_models:
            with st.spinner(f"Processing with {model_name}..."):
                try:
                    results = extract_skills(EMBEDDING_MODELS[model_name], text_input)
                    model_results[model_name] = results

                    # Display results for each model
                    st.subheader(f"Results for {model_name}:")
                    st.json(results)

                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")

        # Plot comparison of exact and semantic matches
        # st.subheader("Comparison of Exact and Semantic Matches Across Models")
        # plot_model_comparison(model_results)
    else:
        st.warning("Please enter text to process.")
