import gradio as gr
import numpy as np
import pandas as pd
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),"cover-not-found.png" , books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_des.txt" , encoding="utf8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0,separator= "\n")
documents = text_splitter.split_documents(raw_documents)

huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding = huggingface_embeddings)

def extract_isbn(text):
    """Extract ISBN13 from text using regex"""
    # Look for 13-digit ISBN at the beginning of the text
    match = re.match(r'^(\d{13})', text.strip())
    if match:
        return int(match.group(1))
    else:
        # Fallback: try to find any 13-digit number
        isbn_match = re.search(r'\b(\d{13})\b', text)
        if isbn_match:
            return int(isbn_match.group(1))
        else:
            # If no 13-digit found, try 10-digit ISBN
            isbn10_match = re.search(r'\b(\d{10})\b', text)
            if isbn10_match:
                return int(isbn10_match.group(1))
    return None

def retrieve_semantic_recommendations (
        query : str,
        category : str = None,
        tone : str = None,
        initial_top_k : int = 50,
        final_top_k : int = 16,
) -> pd.DataFrame :
    recs = db_books.similarity_search(query , k = initial_top_k)
    
    # Extract ISBNs using regex, filter out None values
    books_list = []
    for rec in recs:
        isbn = extract_isbn(rec.page_content)
        if isbn is not None:
            books_list.append(isbn)
    
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category and category != "All" :
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else :
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by = "joy" , ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by = "surprise" , ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by = "anger" , ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by = "fear" , ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by = "sadness" , ascending=False)
    
    return book_recs

def recommend_books(
        query : str ,
        category : str ,
        tone : str ,
       
) :
    
    recommendations = retrieve_semantic_recommendations(query , category , tone) 
    results = []

    for _, row in recommendations.iterrows() :
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2 :
            # Fixed the bug here - was using [-1] twice
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else :
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))  # Fixed tuple syntax
    
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy" , "Surprising" , "Angry" , "Suspenseful" , "Sad"]

with gr.Blocks(theme = gr.themes.Glass() ) as dashboard :
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row() :
        user_query = gr.Textbox(label="Please Enter a description of a book : ",placeholder="e.g , A story about forgiveness")
        category_dropdown = gr.Dropdown(choices= categories , label = "Select a category : " , value= "All")
        tone_dropdown = gr.Dropdown(choices= tones , label = "Select an emotional tone : " , value= "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books" , columns=8, rows=2)

    submit_button.click(fn=recommend_books, inputs=[user_query,category_dropdown,tone_dropdown], outputs=output)


if __name__ == "__main__":
    dashboard.launch()