#!/usr/bin/env python3
"""
🚀 RUNPOD HANDLER FOR UNIFIED AUDIOBOOK PROCESSING
"""

import runpod
import json
import os
import base64
import tempfile
from pathlib import Path
from unified_audiobook_handler import UnifiedAudiobookHandler

# Initialize handler globally for model caching
handler = UnifiedAudiobookHandler()

def process_audiobook_request(job):
    """Process audiobook conversion request from RunPod"""
    
    try:
        input_data = job['input']
        
        pdf_base64 = input_data.get('pdf_data')
        source_lang = input_data.get('source_lang', 'en')
        target_lang = input_data.get('target_lang', 'en')
        start_page = input_data.get('start_page')
        end_page = input_data.get('end_page')
        
        if not pdf_base64:
            return {"error": "No PDF data provided"}
        
        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            pdf_data = base64.b64decode(pdf_base64)
            temp_pdf.write(pdf_data)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Process the audiobook
            result = handler.process_audiobook(
                pdf_path=temp_pdf_path,
                source_lang=source_lang,
                target_lang=target_lang,
                start_page=start_page,
                end_page=end_page
            )
            
            # Read output audio file
            with open(result['output_path'], 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Return result with audio data
            return {
                "success": True,
                "audio_data": audio_base64,
                "audio_format": "wav",
                "engine": result['engine'],
                "processing_time": result['processing_time'],
                "source_lang": result['source_lang'],
                "target_lang": result['target_lang'],
                "text_length": result['text_length']
            }
            
        finally:
            # Cleanup temporary files
            os.unlink(temp_pdf_path)
            if 'output_path' in result:
                if os.path.exists(result['output_path']):
                    os.unlink(result['output_path'])
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": process_audiobook_request})
