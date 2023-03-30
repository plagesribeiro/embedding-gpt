import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import gradio as gr
import os
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
from semanticSearch import SemanticSearch

import sklearn
sklearn.__version__

def download_pdf(url, output_path):
    '''
    download file from URL and save it to `output_path`
    '''
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    '''
    preprocess chunks
    1. Replace new line character with whitespace.
    2. Replace redundant whitespace with a single whitespace
    '''
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    '''
    convert PDF document to text
    '''
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in tqdm(range(start_page-1, end_page)):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    '''
    convert list of texts to smaller chunks of length `word_length`
    '''
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


    
openai.api_key = "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"

recommender = SemanticSearch()

def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'


def generate_text(prompt, engine="text-davinci-003"):
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer(question):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given."\
              "Cite each reference using [number] notation (every result has this number at the beginning)."\
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects"\
              "with the same name, create separate answers for each. Only include information found in the results and"\
              "don't add any additional information. Make sure the answer is correct and don't output false content."\
              "If the text does not relate to the query, simply state 'Found Nothing'. Don't write 'Answer:'"\
              "Directly start the answer.\n"
    
    prompt += f"Query: {question}\n\n"
    answer = generate_text(prompt)
    return answer

def question_answer(url, file, question):
    if url.strip() == '' and file == None:
        return '[ERROR]: Both URL and PDF is empty. Provide atleast one.'
    
    if url.strip() != '' and file != None:
        return '[ERROR]: Both URL and PDF is provided. Please provide only one (eiter URL or PDF).'

    if url.strip() != '':
        glob_url = url
        download_pdf(glob_url, 'corpus.pdf')
        load_recommender('corpus.pdf')

    else:
        old_file_name = file.name
        file_name = file.name
        file_name = file_name[:-12] + file_name[-4:]
        os.rename(old_file_name, file_name)
        load_recommender(file_name)

    if question.strip() == '':
        return '[ERROR]: Question field is empty'

    return generate_answer(question)

title = 'BookGPT'
description = "BookGPT allows you to input an entire book and ask questions about its contents. This app uses GPT-3 to generate answers based on the book's information. BookGPT has ability to add reference to the specific page number from where the information was found. This adds credibility to the answers generated also helps you locate the relevant information in the book."

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        
        with gr.Group():
            url = gr.Textbox(label='URL')
            gr.Markdown("<center><h6>or<h6></center>")
            file = gr.File(label='PDF', file_types=['.pdf'])
            question = gr.Textbox(label='question')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='answer')

        btn.click(question_answer, inputs=[url, file, question], outputs=[answer])

demo.launch()