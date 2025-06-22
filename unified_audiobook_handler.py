#!/usr/bin/env python3
"""
🚀 PHASE 2.1: UNIFIED AUDIOBOOK HANDLER FOR RUNPOD
Containerized version with automatic engine selection:
- English→English: Coqui TTS  
- English→Indian languages: IndicTrans2→AI4Bharat TTS
- Other→Same: Direct Coqui TTS
"""

import torch
import os
import json
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from parler_tts import ParlerTTSForConditionalGeneration
from TTS.api import TTS
import fitz as PyMuPDF
import argparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedAudiobookHandler:
    """Unified handler for all audiobook conversion engines"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")
        
        # Engine states
        self.coqui_tts = None
        self.indictrans2_model = None
        self.indictrans2_tokenizer = None
        self.ai4bharat_model = None
        self.ai4bharat_tokenizer = None
        self.ai4bharat_desc_tokenizer = None
        
        # Language mappings
        self.INDIAN_LANGUAGES = {
            'hi': 'hin_Deva',  # Hindi
            'te': 'tel_Telu',  # Telugu
            'ta': 'tam_Taml',  # Tamil
            'bn': 'ben_Beng',  # Bengali
            'mr': 'mar_Deva',  # Marathi
        }
        
        self.COQUI_LANGUAGES = {
            'en': 'tts_models/en/ljspeech/vits',
            'es': 'tts_models/es/css10/vits', 
            'fr': 'tts_models/fr/css10/vits',
        }

    def load_coqui_tts(self, language='en'):
        """Load Coqui TTS for specified language"""
        if self.coqui_tts is None:
            model_name = self.COQUI_LANGUAGES.get(language, self.COQUI_LANGUAGES['en'])
            logger.info(f"Loading Coqui TTS: {model_name}")
            self.coqui_tts = TTS(model_name=model_name)
        return self.coqui_tts

    def load_indictrans2(self):
        """Load IndicTrans2 for Indian language translation"""
        if self.indictrans2_model is None:
            logger.info("Loading IndicTrans2...")
            model_name = "ai4bharat/indictrans2-en-indic-1B"
            self.indictrans2_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.indictrans2_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
            self.indictrans2_model.to(self.device)
        return self.indictrans2_model, self.indictrans2_tokenizer

    def load_ai4bharat_tts(self):
        """Load AI4Bharat TTS for Indian languages"""
        if self.ai4bharat_model is None:
            logger.info("Loading AI4Bharat TTS...")
            model_name = "ai4bharat/indic-parler-tts"
            self.ai4bharat_model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.ai4bharat_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ai4bharat_desc_tokenizer = AutoTokenizer.from_pretrained(
                self.ai4bharat_model.config.text_encoder._name_or_path
            )
        return self.ai4bharat_model, self.ai4bharat_tokenizer, self.ai4bharat_desc_tokenizer

    def translate_to_indian_language(self, text, target_lang_code):
        """Translate English text to Indian language using IndicTrans2"""
        model, tokenizer = self.load_indictrans2()
        
        target_lang = self.INDIAN_LANGUAGES.get(target_lang_code)
        if not target_lang:
            raise ValueError(f"Unsupported Indian language: {target_lang_code}")
        
        logger.info(f"Translating to {target_lang}...")
        
        input_text = f"eng_Latn {text} {target_lang}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return translation

    def generate_ai4bharat_audio(self, text, language_code, output_path="output.wav"):
        """Generate audio using AI4Bharat TTS"""
        model, tokenizer, desc_tokenizer = self.load_ai4bharat_tts()
        
        voice_descriptions = {
            'te': "A warm Telugu narrator with gentle tone",
            'hi': "A warm Hindi narrator with gentle tone", 
        }
        
        description = voice_descriptions.get(language_code, f"A warm {language_code} narrator")
        
        logger.info(f"Generating AI4Bharat audio ({language_code})")
        
        description_input_ids = desc_tokenizer(description, return_tensors="pt").to(self.device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generation = model.generate(
                input_ids=description_input_ids.input_ids,
                attention_mask=description_input_ids.attention_mask,
                prompt_input_ids=prompt_input_ids.input_ids,
                prompt_attention_mask=prompt_input_ids.attention_mask,
                do_sample=True,
                temperature=0.7,
                max_length=14336,
                min_length=50
            )
        
        audio_arr = generation.cpu().numpy().squeeze()
        sample_rate = model.config.sampling_rate
        
        # Trim leading silence
        threshold = 0.005
        start_idx = 0
        for i, sample in enumerate(audio_arr):
            if abs(sample) > threshold:
                start_idx = max(0, i - 1000)
                break
        
        clean_audio = audio_arr[start_idx:]
        sf.write(output_path, clean_audio, sample_rate)
        return output_path

    def generate_coqui_audio(self, text, language='en', output_path="output.wav"):
        """Generate audio using Coqui TTS"""
        tts = self.load_coqui_tts(language)
        logger.info(f"Generating audio with Coqui TTS ({language})")
        tts.tts_to_file(text=text, file_path=output_path)
        return output_path

    def extract_pdf_text(self, pdf_path, start_page=None, end_page=None):
        """Extract text from PDF using PyMuPDF"""
        logger.info(f"Extracting text from {pdf_path}")
        
        doc = PyMuPDF.open(pdf_path)
        text = ""
        
        start = start_page - 1 if start_page else 0
        end = end_page if end_page else len(doc)
        
        for page_num in range(start, min(end, len(doc))):
            page = doc[page_num]
            text += page.get_text()
        
        doc.close()
        return text.strip()

    def determine_engine(self, source_lang, target_lang):
        """Determine which engine to use based on language combination"""
        if source_lang == target_lang:
            if target_lang in self.COQUI_LANGUAGES:
                return "coqui"
            else:
                raise ValueError(f"Unsupported language for Coqui: {target_lang}")
        
        elif source_lang == 'en' and target_lang in self.INDIAN_LANGUAGES:
            return "indictrans2_ai4bharat"
        
        else:
            raise ValueError(f"Unsupported language combination: {source_lang} → {target_lang}")

    def process_audiobook(self, pdf_path, source_lang='en', target_lang='en', 
                         output_dir="output_audio", start_page=None, end_page=None):
        """Main processing function with automatic engine selection"""
        
        logger.info(f"Processing: {source_lang} → {target_lang}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        text = self.extract_pdf_text(pdf_path, start_page, end_page)
        logger.info(f"Extracted {len(text)} characters")
        
        engine = self.determine_engine(source_lang, target_lang)
        logger.info(f"Using engine: {engine}")
        
        pdf_name = Path(pdf_path).stem
        output_filename = f"{pdf_name}_{source_lang}_{target_lang}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        start_time = time.time()
        
        if engine == "coqui":
            self.generate_coqui_audio(text, target_lang, output_path)
            
        elif engine == "indictrans2_ai4bharat":
            translated_text = self.translate_to_indian_language(text, target_lang)
            logger.info(f"Translation: {translated_text[:100]}...")
            self.generate_ai4bharat_audio(translated_text, target_lang, output_path)
        
        processing_time = time.time() - start_time
        logger.info(f"Completed in {processing_time:.2f}s: {output_path}")
        
        return {
            "success": True,
            "output_path": output_path,
            "engine": engine,
            "processing_time": processing_time,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text_length": len(text)
        }

if __name__ == "__main__":
    handler = UnifiedAudiobookHandler()
    print("Handler ready")
