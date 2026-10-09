import os
import json
import time
import atexit
import logging
from typing import Dict, Any, List

import spacy
from stanfordcorenlp import StanfordCoreNLP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepLinguisticAnalyzer:
    def __init__(self, corenlp_path: str, spacy_model: str = "en_core_web_sm", memory: str = "4g"):
        self.corenlp_path = corenlp_path
        try:
            self.nlp = spacy.load(spacy_model)
            self.nlp.max_length = 2000000 
        except OSError:
            raise RuntimeError(f"spaCy model '{spacy_model}' not found.")

        if not os.path.exists(corenlp_path):
            raise FileNotFoundError(f"Stanford CoreNLP path not found: {corenlp_path}")
            
        self.corenlp_client = StanfordCoreNLP(self.corenlp_path, memory=memory, quiet=True, timeout=60000)
        atexit.register(self.close)

    def _get_sentence_chunks(self, text: str, max_chars: int = 40000, overlap_sentences: int = 2) -> List[str]:
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        chunks = []
        
        current_chunk = []
        current_len = 0
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            if current_len + len(sent) > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                i = max(0, i - overlap_sentences)
                current_chunk = []
                current_len = 0
            else:
                current_chunk.append(sent)
                current_len += len(sent)
                i += 1
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def analyze(self, text: str, required_annotators: List[str] = None) -> Dict[str, Any]:
        if not text.strip(): return self._empty_result()
            
        chunks = self._get_sentence_chunks(text)
        results = self._empty_result()
        
        default_annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref', 'openie']
        annotators_to_run = required_annotators if required_annotators else default_annotators
        props = {'annotators': ",".join(annotators_to_run), 'pipelineLanguage': 'en', 'outputFormat': 'json'}

        global_sent_offset = 0
        cluster_merge_map = {}
        global_coref_cluster_id = 1

        for chunk_idx, chunk in enumerate(chunks):
            doc = self.nlp(chunk)
            results["spacy_syntax"].extend(self._extract_spacy_features(doc))
            
            chunk_success = False
            for attempt in range(3):
                try:
                    output = self.corenlp_client.annotate(chunk, properties=props)
                    parsed_output = json.loads(output)
                    chunk_sentences = parsed_output.get('sentences', [])
                    
                    if 'openie' in annotators_to_run:
                        results["open_ie"].extend(self._extract_openie(chunk_sentences))
                    if 'parse' in annotators_to_run:
                        results["rhetorical_structure"].extend(self._extract_parse_trees(chunk_sentences))
                        
                    if 'coref' in annotators_to_run:
                        chunk_corefs = parsed_output.get('corefs', {})
                        for _, mentions in chunk_corefs.items():
                            if not mentions: continue
                            
                            rep_mention = next((m for m in mentions if m.get('isRepresentativeMention')), mentions[0])
                            rep_text = rep_mention['text'].lower().strip()
                            
                            if rep_text in cluster_merge_map:
                                target_cluster_id = cluster_merge_map[rep_text]
                            else:
                                target_cluster_id = str(global_coref_cluster_id)
                                cluster_merge_map[rep_text] = target_cluster_id
                                results["coreference"][target_cluster_id] = []
                                global_coref_cluster_id += 1

                            shifted_mentions = []
                            for m in mentions:
                                m_copy = dict(m)
                                m_copy['sentNum'] += global_sent_offset
                                shifted_mentions.append(m_copy)
                                
                            results["coreference"][target_cluster_id].extend(shifted_mentions)

                    overlap_deduct = 2 if chunk_idx < len(chunks) - 1 else 0
                    global_sent_offset += max(1, len(chunk_sentences) - overlap_deduct)
                    chunk_success = True
                    break
                    
                except Exception as e:
                    logger.warning(f"CoreNLP chunk failed (Attempt {attempt+1}): {str(e)}")
                    time.sleep(2 ** attempt)
                    
            if not chunk_success:
                logger.error("Chunk failed permanently. Generating partial fallback state to salvage document.")
                global_sent_offset += len(list(self.nlp(chunk).sents))
                
        return results

    def _extract_spacy_features(self, doc) -> List[Dict]:
        return [{
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.text,
            "is_stop": token.is_stop
        } for token in doc]

    def _extract_openie(self, sentences: List[Dict]) -> List[Dict[str, Any]]:
        triplets = []
        for sentence in sentences:
            for ie in sentence.get('openie', []):
                triplets.append({
                    "subject": ie.get('subject'), "relation": ie.get('relation'),
                    "object": ie.get('object'), "confidence": ie.get('confidence', 0.0)
                })
        return triplets

    def _extract_parse_trees(self, sentences: List[Dict]) -> List[str]:
        return [sent.get('parse', '') for sent in sentences]

    def _empty_result(self) -> Dict[str, Any]:
        return {"spacy_syntax": [], "coreference": {}, "open_ie": [], "rhetorical_structure": []}

    def close(self):
        if hasattr(self, 'corenlp_client') and self.corenlp_client:
            self.corenlp_client.close()
            self.corenlp_client = None