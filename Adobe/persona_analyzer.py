'''import os
import sys
import json
import time
import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import nltk
from collections import Counter

# --- NLTK Setup & Pathing ---
# This block fixes stubborn environment issues by explicitly adding the
# standard NLTK data path and then downloading required models.
def setup_nltk():
    """Adds NLTK data path and downloads required models."""
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/stopwords', 'stopwords')
    ]
    for path, pkg_id in required_data:
        try:
            nltk.data.find(path)
        except nltk.downloader.DownloadError:
            print(f"Downloading NLTK package: {pkg_id}...")
            nltk.download(pkg_id)
setup_nltk()

# --- CONFIGURATION ---
MODEL_PATH = "./sbert_model"
TOP_K_SECTIONS = 10 # Final number of sections to return
RE_RANKING_CANDIDATES = 30 # Number of top candidates from 1st pass to re-rank

class DocumentIntelligence:
    """A robust solution using a two-pass re-ranking system."""

    def __init__(self, model_path: str):
        print("Initializing Document Intelligence system...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please download it first.")
        self.device = "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        print("Model loaded successfully on CPU.")

    def extract_sections_from_pdf(self, pdf_path: str) -> list:
        """Extracts structured sections using a reliable heading detection method."""
        sections = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening {pdf_path}: {e}"); return sections

        current_content, current_title, current_page = "", "Introduction", 1
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block: continue
                for line in block["lines"]:
                    if not line["spans"]: continue
                    span = line["spans"][0]
                    text = span["text"].strip()
                    if not text: continue
                    is_heading = (span['flags'] & 2**4) and len(text.split()) < 15

                    if is_heading:
                        if current_content.strip(): sections.append({'document': os.path.basename(pdf_path), 'page': current_page, 'title': current_title, 'content': current_content.strip().replace("\n", " ")})
                        current_title, current_content, current_page = text, "", page_num + 1
                    else:
                        current_content += text + " "
        
        if current_content.strip(): sections.append({'document': os.path.basename(pdf_path), 'page': current_page, 'title': current_title, 'content': current_content.strip().replace("\n", " ")})
        print(f"Extracted {len(sections)} sections from {os.path.basename(pdf_path)}.")
        return sections

    def rank_and_analyze(self, persona: str, job_to_be_done: str, all_sections: list):
        if not all_sections: return [], []

        # --- Pass 1: Broad Search ---
        initial_query = f"User role: {persona}. Task: {job_to_be_done}."
        print(f"Initial Query: {initial_query}")
        query_embedding = self.model.encode(initial_query, convert_to_tensor=True, device=self.device)
        
        section_contents = [sec.get('content', '') for sec in all_sections]
        section_embeddings = self.model.encode(section_contents, convert_to_tensor=True, device=self.device)
        
        scores = util.cos_sim(query_embedding, section_embeddings)[0]
        top_candidate_indices = torch.topk(scores, k=min(RE_RANKING_CANDIDATES, len(all_sections))).indices

        # --- Context Discovery from Top Candidates ---
        top_sections_text = " ".join([all_sections[i]['content'] for i in top_candidate_indices])
        tokens = nltk.word_tokenize(top_sections_text.lower())
        tagged_words = nltk.pos_tag(tokens)
        
        # Extract meaningful nouns and adjectives, excluding stopwords
        initial_query_words = set(nltk.word_tokenize(initial_query.lower()))
        expansion_keywords = [word for word, tag in tagged_words if tag.startswith('NN') and word.isalpha() and word not in self.stopwords and word not in initial_query_words]
        
        # Find the most frequent new concepts
        most_common_keywords = [word for word, count in Counter(expansion_keywords).most_common(10)]
        keyword_string = ", ".join(set(most_common_keywords))

        # --- Pass 2: Refined Search (Re-Ranking) ---
        if keyword_string:
            refined_query = f"{initial_query} This involves concepts like: {keyword_string}."
            print(f"Refined Query with discovered concepts: {refined_query}")
            query_embedding = self.model.encode(refined_query, convert_to_tensor=True, device=self.device)
        
        # Re-rank only the top candidates from the first pass
        candidate_embeddings = section_embeddings[top_candidate_indices]
        new_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        
        # Get the new top indices relative to the candidate list
        final_top_indices_relative = torch.topk(new_scores, k=min(TOP_K_SECTIONS, len(top_candidate_indices))).indices
        
        # Map back to original section indices
        final_ranked_sections = [all_sections[top_candidate_indices[i]] for i in final_top_indices_relative]
        
        # --- Output Generation ---
        extracted_sections_output, subsection_analysis_output = [], []
        for rank, section in enumerate(final_ranked_sections):
            extracted_sections_output.append({"document": section["document"], "section_title": section["title"], "importance_rank": rank + 1, "page_number": section["page"]})
            refined_text = self._get_refined_text(query_embedding, section.get('content', ''))
            subsection_analysis_output.append({"document": section["document"], "refined_text": refined_text, "page_number": section["page"]})
        return extracted_sections_output, subsection_analysis_output

    def _get_refined_text(self, query_embedding, section_content: str, num_sentences=4) -> str:
        if not section_content: return ""
        replacements = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
        for old, new in replacements.items(): section_content = section_content.replace(old, new)
        sentences = nltk.sent_tokenize(section_content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        if not meaningful_sentences: return section_content[:500].strip()
        sentence_embeddings = self.model.encode(meaningful_sentences, convert_to_tensor=True, device=self.device)
        scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
        top_indices = torch.topk(scores, k=min(num_sentences, len(meaningful_sentences))).indices.tolist()
        top_sentences = [meaningful_sentences[i] for i in sorted(top_indices)]
        return " ".join(top_sentences)

def main(collection_path: str):
    start_pipeline_time = time.time()
    input_json_path = os.path.join(collection_path, "challenge1b_input.json")
    pdf_dir = os.path.join(collection_path, "PDFs")
    output_json_path = os.path.join(collection_path, "challenge1b_output.json")
    if not os.path.exists(input_json_path): print(f"Error: 'challenge1b_input.json' not found in {collection_path}"); return
    with open(input_json_path, 'r', encoding='utf-8') as f: input_data = json.load(f)
    persona, job_to_be_done = input_data.get("persona", {}).get("role", ""), input_data.get("job_to_be_done", {}).get("task", "")
    input_documents_info = input_data.get("documents", [])
    print(f"--- Processing Collection: {collection_path} ---\nPersona: {persona}\nTask: {job_to_be_done}")
    analyzer = DocumentIntelligence(model_path=MODEL_PATH)
    all_sections, pdf_files_to_process = [], [doc['filename'] for doc in input_documents_info]
    for filename in pdf_files_to_process:
        pdf_path = os.path.join(pdf_dir, filename)
        if os.path.exists(pdf_path): all_sections.extend(analyzer.extract_sections_from_pdf(pdf_path))
        else: print(f"Warning: PDF file '{filename}' not found in {pdf_dir}")
    if not all_sections: print("Error: No sections extracted."); return
    extracted, subsections = analyzer.rank_and_analyze(persona, job_to_be_done, all_sections)
    output_data = {"metadata": {"input_documents": pdf_files_to_process, "persona": persona, "job_to_be_done": job_to_be_done}, "extracted_sections": extracted, "subsection_analysis": subsections}
    with open(output_json_path, 'w', encoding='utf-8') as f: json.dump(output_data, f, indent=2)
    print(f"\n✅ Success! Analysis complete for {os.path.basename(collection_path)}.")
    print(f"Processing Time: {time.time() - start_pipeline_time:.2f} seconds.")
    print(f"Output saved to: {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        collection_path = sys.argv[1]
        if os.path.isdir(collection_path): main(collection_path)
        else: print(f"Error: Provided path '{collection_path}' is not a valid directory.")
    else:
        print("---")
        print("Usage: python persona_analyzer.py <path_to_collection_folder>")
        print("Example: python persona_analyzer.py Challenge_1b/Collection_1")
        print("---")'''
        
import os
import sys
import json
import time
import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import nltk
from collections import Counter

# --- NLTK Setup & Pathing ---
# This block fixes stubborn environment issues by explicitly adding the
# standard NLTK data path and then downloading required models.
def setup_nltk():
    """Adds NLTK data path and downloads required models."""
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/stopwords', 'stopwords')
    ]
    for path, pkg_id in required_data:
        try:
            nltk.data.find(path)
        except nltk.downloader.DownloadError:
            print(f"Downloading NLTK package: {pkg_id}...")
            nltk.download(pkg_id)
setup_nltk()

# --- CONFIGURATION ---
MODEL_PATH = "./sbert_model"
TOP_K_SECTIONS = 10 # Final number of sections to return
RE_RANKING_CANDIDATES = 30 # Number of top candidates from 1st pass to re-rank

class DocumentIntelligence:
    """A robust solution using a two-pass re-ranking system."""

    def __init__(self, model_path: str):
        print("Initializing Document Intelligence system...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please download it first.")
        self.device = "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        print("Model loaded successfully on CPU.")

    def extract_sections_from_pdf(self, pdf_path: str) -> list:
        """Extracts structured sections using a reliable heading detection method."""
        sections = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening {pdf_path}: {e}"); return sections

        current_content, current_title, current_page = "", "Introduction", 1
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block: continue
                for line in block["lines"]:
                    if not line["spans"]: continue
                    span = line["spans"][0]
                    text = span["text"].strip()
                    if not text: continue
                    is_heading = (span['flags'] & 2**4) and len(text.split()) < 15

                    if is_heading:
                        if current_content.strip(): sections.append({'document': os.path.basename(pdf_path), 'page': current_page, 'title': current_title, 'content': current_content.strip().replace("\n", " ")})
                        current_title, current_content, current_page = text, "", page_num + 1
                    else:
                        current_content += text + " "
        
        if current_content.strip(): sections.append({'document': os.path.basename(pdf_path), 'page': current_page, 'title': current_title, 'content': current_content.strip().replace("\n", " ")})
        print(f"Extracted {len(sections)} sections from {os.path.basename(pdf_path)}.")
        return sections

    def rank_and_analyze(self, persona: str, job_to_be_done: str, all_sections: list):
        if not all_sections: return [], []

        # --- Pass 1: Broad Search ---
        initial_query = f"User role: {persona}. Task: {job_to_be_done}."
        print(f"Initial Query: {initial_query}")
        query_embedding = self.model.encode(initial_query, convert_to_tensor=True, device=self.device)
        
        section_contents = [sec.get('content', '') for sec in all_sections]
        section_embeddings = self.model.encode(section_contents, convert_to_tensor=True, device=self.device)
        
        scores = util.cos_sim(query_embedding, section_embeddings)[0]
        top_candidate_indices = torch.topk(scores, k=min(RE_RANKING_CANDIDATES, len(all_sections))).indices

        # --- Context Discovery from Top Candidates ---
        top_sections_text = " ".join([all_sections[i]['content'] for i in top_candidate_indices])
        tokens = nltk.word_tokenize(top_sections_text.lower())
        tagged_words = nltk.pos_tag(tokens)
        
        # Extract meaningful nouns and adjectives, excluding stopwords
        initial_query_words = set(nltk.word_tokenize(initial_query.lower()))
        expansion_keywords = [word for word, tag in tagged_words if tag.startswith('NN') and word.isalpha() and word not in self.stopwords and word not in initial_query_words]
        
        # Find the most frequent new concepts
        most_common_keywords = [word for word, count in Counter(expansion_keywords).most_common(10)]
        keyword_string = ", ".join(set(most_common_keywords))

        # --- Pass 2: Refined Search (Re-Ranking) ---
        if keyword_string:
            refined_query = f"{initial_query} This involves concepts like: {keyword_string}."
            print(f"Refined Query with discovered concepts: {refined_query}")
            query_embedding = self.model.encode(refined_query, convert_to_tensor=True, device=self.device)
        
        # Re-rank only the top candidates from the first pass
        candidate_embeddings = section_embeddings[top_candidate_indices]
        new_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        
        # Get the new top indices relative to the candidate list
        final_top_indices_relative = torch.topk(new_scores, k=min(TOP_K_SECTIONS, len(top_candidate_indices))).indices
        
        # Map back to original section indices
        final_ranked_sections = [all_sections[top_candidate_indices[i]] for i in final_top_indices_relative]
        
        # --- Output Generation ---
        extracted_sections_output, subsection_analysis_output = [], []
        for rank, section in enumerate(final_ranked_sections):
            extracted_sections_output.append({"document": section["document"], "section_title": section["title"], "importance_rank": rank + 1, "page_number": section["page"]})
            refined_text = self._get_refined_text(query_embedding, section.get('content', ''))
            subsection_analysis_output.append({"document": section["document"], "refined_text": refined_text, "page_number": section["page"]})
        return extracted_sections_output, subsection_analysis_output

    def _get_refined_text(self, query_embedding, section_content: str, num_sentences=4) -> str:
        if not section_content: return ""
        replacements = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
        for old, new in replacements.items(): section_content = section_content.replace(old, new)
        sentences = nltk.sent_tokenize(section_content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        if not meaningful_sentences: return section_content[:500].strip()
        sentence_embeddings = self.model.encode(meaningful_sentences, convert_to_tensor=True, device=self.device)
        scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
        top_indices = torch.topk(scores, k=min(num_sentences, len(meaningful_sentences))).indices.tolist()
        top_sentences = [meaningful_sentences[i] for i in sorted(top_indices)]
        return " ".join(top_sentences)

def main(collection_path: str):
    start_pipeline_time = time.time()
    input_json_path = os.path.join(collection_path, "challenge1b_input.json")
    pdf_dir = os.path.join(collection_path, "PDFs")
    output_json_path = os.path.join(collection_path, "challenge1b_output.json")
    if not os.path.exists(input_json_path): print(f"Error: 'challenge1b_input.json' not found in {collection_path}"); return
    with open(input_json_path, 'r', encoding='utf-8') as f: input_data = json.load(f)
    persona, job_to_be_done = input_data.get("persona", {}).get("role", ""), input_data.get("job_to_be_done", {}).get("task", "")
    input_documents_info = input_data.get("documents", [])
    print(f"--- Processing Collection: {collection_path} ---\nPersona: {persona}\nTask: {job_to_be_done}")
    analyzer = DocumentIntelligence(model_path=MODEL_PATH)
    all_sections, pdf_files_to_process = [], [doc['filename'] for doc in input_documents_info]
    for filename in pdf_files_to_process:
        pdf_path = os.path.join(pdf_dir, filename)
        if os.path.exists(pdf_path): all_sections.extend(analyzer.extract_sections_from_pdf(pdf_path))
        else: print(f"Warning: PDF file '{filename}' not found in {pdf_dir}")
    if not all_sections: print("Error: No sections extracted."); return
    extracted, subsections = analyzer.rank_and_analyze(persona, job_to_be_done, all_sections)
    output_data = {"metadata": {"input_documents": pdf_files_to_process, "persona": persona, "job_to_be_done": job_to_be_done}, "extracted_sections": extracted, "subsection_analysis": subsections}
    with open(output_json_path, 'w', encoding='utf-8') as f: json.dump(output_data, f, indent=2)
    print(f"\n✅ Success! Analysis complete for {os.path.basename(collection_path)}.")
    print(f"Processing Time: {time.time() - start_pipeline_time:.2f} seconds.")
    print(f"Output saved to: {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        collection_path = sys.argv[1]
        if os.path.isdir(collection_path): main(collection_path)
        else: print(f"Error: Provided path '{collection_path}' is not a valid directory.")
    else:
        print("---")
        print("Usage: python persona_analyzer.py <path_to_collection_folder>")
        print("Example: python persona_analyzer.py Challenge_1b/Collection_1")
        print("---")