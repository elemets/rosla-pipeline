import spacy
import scispacy
from scispacy.linking import EntityLinker
import spacy_transformers
import pandas as pd
import numpy as np
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
import typer

tqdm.pandas()

# Global variables that will be initialized based on embedding type
nlp = None
organic_cui_set = None
cui_pick = None
stop_words = None
embeddings_dict = None
tokenizer = None
model = None


def finding_cuis(input_df):
    """
    Generate CUIs for the text in the input DataFrame.

    Args:
        input_df (pd.DataFrame): DataFrame containing a column 'text' with text data.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'new_cui' containing the generated CUIs.
    """
    input_df["new_cui"] = input_df["text"].progress_apply(cui_gen)
    return input_df


def cui_gen(text):
    """
    Generate CUIs for the given text, filtering only those within the organic chemical CUIs.

    Args:
        text (str): Input text to process.

    Returns:
        list: List of CUIs that are within the organic chemical CUIs.
    """
    global nlp, organic_cui_set
    doc = nlp(text)
    cuis = []
    if doc.ents:
        for ent in doc.ents:
            if ent._.kb_ents:
                cuis.append(ent._.kb_ents[0][0])
            else:
                continue
        cuis = [cui for cui in cuis if cui in organic_cui_set]
    return cuis


def converting_cuis_2_vec(input_data):
    """
    Convert the found CUIs into vectors.

    Args:
        input_data (pd.DataFrame): DataFrame containing a column 'new_cui' with CUIs.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'vector' containing the vector representations of the CUIs.
    """
    print("Turning the found CUIS into Vectors")
    input_data["vector"] = input_data["new_cui"].progress_apply(get_summed_vector)
    input_data["vector"] = input_data["vector"].progress_apply(conv_to_list)
    return input_data


def conv_to_list(value):
    """
    Convert array-like objects to lists.

    Args:
        value: Input value to convert.

    Returns:
        list or int: List if the input is array-like, otherwise the input value.
    """
    if isinstance(value, int) and value == 0:
        return np.array(value)
    else:
        return np.array(value)


def get_summed_vector(cuis):
    """
    Fetch vectors for the CUIs and sum them.

    Args:
        cuis (list): List of CUIs.

    Returns:
        np.ndarray: Summed vector of the CUIs.
    """
    global cui_pick
    vectors = cui_pick[cui_pick["cui"].isin(cuis)]["vector"]
    summed_vector = np.sum(vectors)
    return summed_vector


def remove_stop_words(row):
    """
    Remove stop words from the given text.

    Args:
        row (str): Input text to process.

    Returns:
        list: List of words with stop words removed.
    """
    global stop_words
    tokens = word_tokenize(row)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_text = " ".join(filtered_tokens)
    original_list = [element.strip() for element in filtered_text.split(",")]
    original_list = [element for element in original_list if element]
    return original_list


def remove_if_more_than_five(original_list):
    """
    Remove elements that occur more than five times in the list.

    Args:
        original_list (list): List of elements to process.

    Returns:
        list: Filtered list with elements occurring more than five times removed.
    """
    counts = {element: original_list.count(element) for element in set(original_list)}
    to_remove = {element for element, count in counts.items() if count > 5}
    new_list = [element for element in original_list if element not in to_remove]
    filtered_list = [element for element in new_list if element]
    return filtered_list


def glove_to_embed(text):
    """
    Convert text to GloVe embeddings and sum them.

    Args:
        text (list): List of words to convert.

    Returns:
        np.ndarray: Summed GloVe embeddings of the words.
    """
    global embeddings_dict
    embeddings = []
    for word in text:
        word = word.lower().strip()
        try:
            embed = embeddings_dict[word]
        except:
            embed = np.zeros(100)
        embeddings.append(embed)
    return np.sum(embeddings, axis=0)


def clinbert_embed(text):
    """
    Convert text to ClinBERT embeddings.

    Args:
        text (str): Input text to process.

    Returns:
        np.ndarray: ClinBERT embeddings of the text.
    """
    global tokenizer, model
    with torch.no_grad():
        inputs = tokenizer(text, padding=True, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        # Sum the embeddings for each sample in the batch along the token dimension
        feature_vectors = outputs.last_hidden_state[:, 0, :].numpy()

    features_array = np.array(feature_vectors)
    return features_array


def initialize_cui_resources():
    """Initialize resources needed for CUI processing."""
    global nlp, organic_cui_set, cui_pick

    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe(
        "scispacy_linker",
        config={"resolve_abbreviations": True, "linker_name": "umls"},
    )

    # Reading the csv to determine which CUIs are organic chemicals
    organic_chemical_cuis = pd.read_csv("../data/required_for_conversion/df_cui.csv")
    organic_chemical_cuis = organic_chemical_cuis[
        organic_chemical_cuis["semantic_type"] == "Organic Chemical"
    ]
    # Creating a set of these to use in filtering the CUIs
    organic_cui_set = set(organic_chemical_cuis["cui"].astype(str))

    # Loading in the pickled CUI2Vec model so that we can convert them
    cui_pick = pd.read_pickle("../data/required_for_conversion/cui2vec_pickled.pkl")


def initialize_nltk_resources():
    """Initialize NLTK resources."""
    global stop_words

    nltk.download("punkt_tab")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def initialize_glove_embeddings():
    """Initialize GloVe embeddings dictionary."""
    global embeddings_dict

    embeddings_dict = {}
    with open(
        "../data/required_for_conversion/glove.6B.100d.txt",
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector


def initialize_clinbert_model():
    """Initialize ClinBERT model and tokenizer."""
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


app = typer.Typer(help="Building features from raw data")


@app.command()
def main(
    input_data: str = typer.Argument(
        "../data/beforepreproc/inputdatanopreprocessing.csv", help="Path to input data"
    ),
    embedding_type: str = typer.Argument(
        "bioclinicalbert",
        help="Embeddings to generate (cui, glove, tfidf, bioclinicalbert, noembeddings)",
    ),
):
    """Main function to process embeddings based on the specified type."""

    input_df = pd.read_csv(f"{input_data}")

    if embedding_type == "cui":
        if not os.path.exists("../data/different_embeddings/cui_vec_test.pkl"):
            print("Initializing CUI resources...")
            initialize_cui_resources()

            print("Finding CUIs in the text using scispacy and nltk")
            cuis_found = finding_cuis(input_df)
            input_df = converting_cuis_2_vec(cuis_found)
            input_df.to_pickle("../data/different_embeddings/cui_vec_test.pkl")
        else:
            print("CUI embeddings file already exists.")

    elif embedding_type == "glove":
        if not os.path.exists("../data/different_embeddings/glove_embeddings.pkl"):
            print("Initializing NLTK resources...")
            initialize_nltk_resources()

            print("First removing stop words")
            input_df["text"] = input_df["text"].progress_apply(remove_stop_words)

            print("Loading GloVe embeddings...")
            initialize_glove_embeddings()

            print("Now applying glove embeddings")
            input_df["GloVE_proc"] = input_df["text"].progress_apply(glove_to_embed)
            input_df.to_pickle("../data/different_embeddings/glove_embeddings.pkl")
        else:
            print("GloVe embeddings file already exists.")

    elif embedding_type == "tfidf":
        if not os.path.exists("../data/different_embeddings/tfidf_ready.pkl"):
            print("Initializing NLTK resources...")
            initialize_nltk_resources()

            print("First removing stop words")
            input_df["text"] = input_df["text"].progress_apply(remove_stop_words)

            print("Preprocessing the text ready for tfidf")
            input_df["text_proc"] = input_df["text"].progress_apply(
                remove_if_more_than_five
            )
            input_df.to_pickle("../data/different_embeddings/tfidf_ready.pkl")
        else:
            print("TF-IDF ready file already exists.")

    elif embedding_type == "bioclinicalbert":
        if not os.path.exists("../data/different_embeddings/clinbert_embeddings.pkl"):
            print("Initializing ClinBERT model...")
            initialize_clinbert_model()

            print("Converting into bioclinicalBERT embeddings")
            input_df["clinBERTEmbed"] = input_df["text"].progress_apply(clinbert_embed)
            input_df.to_pickle("../data/different_embeddings/clinbert_embeddings.pkl")
        else:
            print("ClinBERT embeddings file already exists.")

    elif embedding_type == "noembeddings":
        input_df.to_pickle("../data/different_embeddings/no_embeds.pkl")
        print("Saved data without embeddings.")

    else:
        raise ValueError(
            "Please provide a valid embedding type: cui, glove, tfidf, bioclinicalbert, or noembeddings"
        )

    print(f"Processing complete for {embedding_type} embeddings!")


if __name__ == "__main__":
    app()
