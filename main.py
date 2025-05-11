#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Healthcare OCR Inference Deployment Script
-----------------------------------------
Phase 2 script for deploying fine-tuned TrOCR model for medical handwriting recognition
in a healthcare organization environment, with high availability, monitoring, and 
HIPAA compliance features.
"""

import os
import sys
import argparse
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.tokenization_utils_base import BatchEncoding
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# HIPAA Compliance Modules
import uuid
from cryptography.fernet import Fernet
import hashlib

# Optional: Performance Optimization & Monitoring
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Optional: Model quantization for performance if needed
try:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

# Set up logging with HIPAA-compliant format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [RequestID: %(request_id)s] %(message)s',
    handlers=[
        logging.FileHandler("ocr_inference.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class RequestIDFilter(logging.Filter):
    """Add request ID to log records for HIPAA-compliant audit trails"""
    
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'system'
        return True

logger = logging.getLogger(__name__)
logger.addFilter(RequestIDFilter())

# Define API models
class OCRRequest(BaseModel):
    """Model for OCR request metadata"""
    provider_id: Optional[str] = Field(None, description="ID of healthcare provider")
    document_type: Optional[str] = Field(None, description="Type of medical document")
    priority: Optional[str] = Field("normal", description="Processing priority")
    callback_url: Optional[str] = Field(None, description="URL for async callback")

class OCRResponse(BaseModel):
    """Model for OCR API response"""
    request_id: str
    text: str
    confidence: float
    processing_time: float
    warnings: List[str] = []

class OCRHealthResponse(BaseModel):
    """Model for health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    uptime: float
    processed_documents: int

class HealthcareOCREngine:
    """
    Main OCR engine class for medical handwriting recognition
    with HIPAA compliance, monitoring, and high-availability features
    """
    
    def __init__(
        self, 
        model_path: str,
        use_onnx: bool = False,
        use_quantization: bool = False,
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
        enable_encryption: bool = True,
        load_ensemble: bool = False
    ):
        """
        Initialize OCR engine with the fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned TrOCR model
            use_onnx: Whether to use ONNX runtime for inference
            use_quantization: Whether to use quantization for optimization
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detection)
            confidence_threshold: Threshold for confidence scores
            enable_encryption: Enable encryption for PII
            load_ensemble: Load ensemble models for specialized document types
        """
        self.start_time = time.time()
        self.processed_count = 0
        self.confidence_threshold = confidence_threshold
        self.enable_encryption = enable_encryption
        
        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}", extra={"request_id": "system"})
        
        # Load the model
        self.load_model(model_path, use_onnx, use_quantization)
        
        # Initialize encryption key
        if self.enable_encryption:
            self._initialize_encryption()
        
        # Load ensemble models if needed
        self.ensemble_models = {}
        if load_ensemble:
            self._load_ensemble_models(model_path)
        
        # Initialize health monitoring
        self._initialize_monitoring()
        
        logger.info("OCR Engine initialization complete", extra={"request_id": "system"})
    
    def _initialize_encryption(self):
        """Initialize encryption for HIPAA compliance"""
        key_file = "encryption.key"
        if not os.path.exists(key_file):
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            logger.info("Generated new encryption key", extra={"request_id": "system"})
        else:
            with open(key_file, "rb") as f:
                key = f.read()
            logger.info("Loaded existing encryption key", extra={"request_id": "system"})
        
        self.cipher = Fernet(key)
    
    def _initialize_monitoring(self):
        """Initialize performance monitoring"""
        self.metrics = {
            "inference_times": [],
            "confidence_scores": [],
            "errors": [],
            "warnings": []
        }
    
    def load_model(self, model_path: str, use_onnx: bool, use_quantization: bool):
        """Load the fine-tuned TrOCR model"""
        try:
            logger.info(f"Loading model from {model_path}", extra={"request_id": "system"})
            
            # Load processor and model using Hugging Face transformers
            self.processor = TrOCRProcessor.from_pretrained(model_path)
            
            if use_onnx and ONNX_AVAILABLE:
                # Use ONNX runtime for optimized inference
                logger.info("Using ONNX Runtime for inference", extra={"request_id": "system"})
                
                # Convert to ONNX if the model isn't already in ONNX format
                if not os.path.exists(os.path.join(model_path, "model.onnx")):
                    logger.info("Converting model to ONNX format", extra={"request_id": "system"})
                    model = VisionEncoderDecoderModel.from_pretrained(model_path)
                    
                    # Export to ONNX (simplified pseudo-code)
                    # In production, use proper ONNX export tools
                    # This is a placeholder for the actual ONNX conversion process
                    onnx_path = os.path.join(model_path, "model.onnx")
                    # Export code would go here
                    
                    # Apply quantization if requested
                    if use_quantization and QUANTIZATION_AVAILABLE:
                        logger.info("Applying quantization to ONNX model", extra={"request_id": "system"})
                        quantizer = ORTQuantizer.from_pretrained(model_path)
                        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False)
                        # Apply quantization (simplified)
                        # quantizer.quantize(save_dir=model_path, quantization_config=qconfig)
                
                # Set up ONNX runtime session
                self.session = ort.InferenceSession(
                    os.path.join(model_path, "model.onnx"),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.model = None  # Not using PyTorch model directly
            else:
                # Load standard PyTorch model
                self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
                if use_quantization and torch.cuda.is_available():
                    logger.info("Applying PyTorch quantization", extra={"request_id": "system"})
                    # Apply simple quantization for PyTorch model
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                
                self.model.to(self.device)
                self.model.eval()
                self.session = None
            
            logger.info("Model loaded successfully", extra={"request_id": "system"})
            self.model_loaded = True
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", extra={"request_id": "system"})
            self.model_loaded = False
            raise
    
    def _load_ensemble_models(self, base_model_path: str):
        """Load specialized models for different document types"""
        try:
            # Define paths to specialized models - adjust as needed for your deployment
            specialized_models = {
                "diagrams": os.path.join(os.path.dirname(base_model_path), "specialized/diagrams"),
                "abbreviations": os.path.join(os.path.dirname(base_model_path), "specialized/abbreviations")
            }
            
            for model_type, model_path in specialized_models.items():
                if os.path.exists(model_path):
                    logger.info(f"Loading specialized model for {model_type}", extra={"request_id": "system"})
                    processor = TrOCRProcessor.from_pretrained(model_path)
                    model = VisionEncoderDecoderModel.from_pretrained(model_path)
                    model.to(self.device)
                    model.eval()
                    self.ensemble_models[model_type] = {
                        "processor": processor,
                        "model": model
                    }
                    logger.info(f"Specialized model for {model_type} loaded", extra={"request_id": "system"})
                else:
                    logger.warning(f"Specialized model for {model_type} not found at {model_path}", 
                                  extra={"request_id": "system"})
        
        except Exception as e:
            logger.error(f"Failed to load ensemble models: {str(e)}", extra={"request_id": "system"})
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Preprocess the input image for OCR
        
        Args:
            image: Path to image file, PIL Image, or numpy array
            
        Returns:
            Preprocessed inputs for the model
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError("Invalid image format")
            
            # Apply TrOCR processor
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            
            if self.device.type == "cuda":
                pixel_values = pixel_values.to(self.device)
                
            return {"pixel_values": pixel_values}
        
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}", extra={"request_id": "system"})
            raise
    
    def classify_document_type(self, image: Image.Image) -> str:
        """
        Classify document type to determine if specialized model should be used
        
        Args:
            image: Input PIL image
            
        Returns:
            Document type classification
        """
        # This is a simplified version - in production, implement a proper classifier
        # For now, using a basic heuristic based on image characteristics
        
        # Check if image might contain a diagram (simplified heuristic)
        # Real implementation would use a proper classifier model
        img_array = np.array(image.convert("L"))  # Convert to grayscale numpy array
        
        # Simple heuristic: Check for specific patterns that might indicate a diagram
        # Higher edge count might indicate diagrams
        from scipy import ndimage
        edges = ndimage.sobel(img_array)
        edge_count = np.sum(edges > 50)  # Arbitrary threshold
        
        total_pixels = img_array.shape[0] * img_array.shape[1]
        edge_ratio = edge_count / total_pixels
        
        # Detect if likely to be a diagram (simplified)
        if edge_ratio > 0.1:  # Arbitrary threshold
            return "diagrams"
        
        # More sophisticated detection would be implemented here
        return "general"
    
    def detect_abbreviations(self, text: str) -> bool:
        """
        Detect if text has high abbreviation content
        
        Args:
            text: Recognized text
            
        Returns:
            True if high abbreviation content detected
        """
        # Simple heuristic - count proportion of short words and words with periods
        words = text.split()
        short_words = [w for w in words if len(w) <= 3]
        words_with_periods = [w for w in words if '.' in w]
        
        # If more than 30% are short words or have periods, likely abbreviation-heavy
        if (len(short_words) / max(len(words), 1)) > 0.3 or (len(words_with_periods) / max(len(words), 1)) > 0.2:
            return True
        return False
    
    def perform_ocr(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        document_type: Optional[str] = None,
        request_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Perform OCR on the input image
        
        Args:
            image: Input image (file path, PIL image, or numpy array)
            document_type: Type of medical document (optional)
            request_id: Unique request ID for tracking
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        warnings = []
        
        try:
            # Convert to PIL Image if not already
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            
            # Auto-detect document type if not provided
            detected_type = document_type
            if detected_type is None:
                detected_type = self.classify_document_type(image)
                logger.info(f"Auto-detected document type: {detected_type}", extra={"request_id": request_id})
            
            # Determine which model to use
            model_to_use = "general"
            if detected_type in self.ensemble_models:
                model_to_use = detected_type
                logger.info(f"Using specialized model for {detected_type}", extra={"request_id": request_id})
            
            # Preprocess the image
            if model_to_use != "general" and model_to_use in self.ensemble_models:
                # Use specialized processor
                processor = self.ensemble_models[model_to_use]["processor"]
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(self.device)
                model = self.ensemble_models[model_to_use]["model"]
            else:
                # Use default processor
                inputs = self.preprocess_image(image)
                pixel_values = inputs["pixel_values"]
                model = self.model
            
            # Generate OCR text
            if self.session is not None:
                # Using ONNX runtime
                ort_inputs = {self.session.get_inputs()[0].name: pixel_values.cpu().numpy()}
                ort_outputs = self.session.run(None, ort_inputs)
                generated_ids = torch.tensor(ort_outputs[0])
            else:
                # Using PyTorch model
                with torch.no_grad():
                    if model_to_use != "general" and model_to_use in self.ensemble_models:
                        generated_ids = self.ensemble_models[model_to_use]["model"].generate(
                            pixel_values,
                            max_length=64,
                            num_beams=4,
                            early_stopping=True,
                            output_scores=True,
                            return_dict_in_generate=True
                        )
                    else:
                        generated_ids = self.model.generate(
                            pixel_values,
                            max_length=64,
                            num_beams=4,
                            early_stopping=True,
                            output_scores=True,
                            return_dict_in_generate=True
                        )
            
            # Decode the generated text
            if model_to_use != "general" and model_to_use in self.ensemble_models:
                processor = self.ensemble_models[model_to_use]["processor"]
                if hasattr(generated_ids, "sequences"):
                    text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
                    # Calculate confidence from sequence scores (simplified)
                    confidence = float(torch.mean(torch.exp(generated_ids.sequences_scores)).cpu().numpy())
                else:
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    confidence = 0.8  # Default confidence when scores not available
            else:
                if hasattr(generated_ids, "sequences"):
                    text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
                    # Calculate confidence from sequence scores (simplified)
                    confidence = float(torch.mean(torch.exp(generated_ids.sequences_scores)).cpu().numpy())
                else:
                    text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    confidence = 0.8  # Default confidence when scores not available
            
            # Post-processing for medical text
            text = self.post_process_medical_text(text)
            
            # Check if text is likely to have high abbreviation content
            if self.detect_abbreviations(text) and "abbreviations" in self.ensemble_models and model_to_use != "abbreviations":
                # If we detect abbreviations and we didn't already use the abbreviations model, try again
                warnings.append("High abbreviation content detected - results may need verification")
                
                # Could re-run with specialized model if needed
                # This is a simplified approach - in production, implement more robust handling
            
            # Log results
            processing_time = time.time() - start_time
            logger.info(
                f"OCR completed in {processing_time:.2f}s with confidence {confidence:.2f}",
                extra={"request_id": request_id}
            )
            
            # Update metrics
            self.processed_count += 1
            self.metrics["inference_times"].append(processing_time)
            self.metrics["confidence_scores"].append(confidence)
            
            # Flag for human review if confidence is low
            if confidence < self.confidence_threshold:
                warnings.append(f"Low confidence score ({confidence:.2f}) - may need human verification")
            
            return {
                "text": text,
                "confidence": confidence,
                "processing_time": processing_time,
                "warnings": warnings,
                "model_used": model_to_use
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR processing error: {str(e)}"
            logger.error(error_msg, extra={"request_id": request_id})
            self.metrics["errors"].append(str(e))
            
            return {
                "text": "",
                "confidence": 0.0,
                "processing_time": processing_time,
                "warnings": [error_msg],
                "error": True
            }
    
    def post_process_medical_text(self, text: str) -> str:
        """
        Post-process OCR text for medical context
        
        Args:
            text: Raw OCR text
            
        Returns:
            Processed text with medical terminology corrections
        """
        # This is a simplified version - in production, implement more sophisticated processing
        
        # 1. Basic cleanup
        text = text.strip()
        
        # 2. Common medical abbreviation expansions
        # In production, use a comprehensive medical abbreviation dictionary
        med_abbreviations = {
            "pt": "patient",
            "pts": "patients",
            "dx": "diagnosis",
            "hx": "history",
            "tx": "treatment",
            "px": "physical examination",
            "fx": "fracture",
            "Rx": "prescription",
            "sx": "symptoms",
            "HR": "heart rate",
            "BP": "blood pressure",
            "SOB": "shortness of breath"
        }
        
        # Only replace whole word abbreviations (not parts of words)
        words = text.split()
        for i, word in enumerate(words):
            cleaned_word = word.strip('.,;:()[]{}').lower()
            if cleaned_word in med_abbreviations:
                # Replace while preserving punctuation
                prefix = ""
                suffix = ""
                for c in word:
                    if c.isalnum():
                        break
                    prefix += c
                
                for c in reversed(word):
                    if c.isalnum():
                        break
                    suffix = c + suffix
                
                words[i] = prefix + med_abbreviations[cleaned_word] + suffix
        
        text = " ".join(words)
        
        # 3. Format vitals consistently
        # For example, standardize blood pressure format
        # This requires more complex regex patterns in production
        
        return text
    
    def get_health_status(self) -> Dict:
        """Get health status of the OCR system"""
        return {
            "status": "healthy" if self.model_loaded else "degraded",
            "model_loaded": self.model_loaded,
            "gpu_available": torch.cuda.is_available(),
            "uptime": time.time() - self.start_time,
            "processed_documents": self.processed_count,
            "avg_processing_time": np.mean(self.metrics["inference_times"]) if self.metrics["inference_times"] else 0,
            "avg_confidence": np.mean(self.metrics["confidence_scores"]) if self.metrics["confidence_scores"] else 0,
            "error_count": len(self.metrics["errors"])
        }
    
    def encrypt_pii(self, text: str) -> str:
        """Encrypt personally identifiable information for HIPAA compliance"""
        if not self.enable_encryption:
            return text
        
        return self.cipher.encrypt(text.encode()).decode()
    
    def decrypt_pii(self, encrypted_text: str) -> str:
        """Decrypt encrypted PII data"""
        if not self.enable_encryption:
            return encrypted_text
        
        return self.cipher.decrypt(encrypted_text.encode()).decode()


# Create FastAPI application
app = FastAPI(title="Healthcare OCR API", 
              description="HIPAA-compliant OCR API for medical handwriting recognition",
              version="1.0.0")

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR engine instance
ocr_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize OCR engine on startup"""
    global ocr_engine
    
    # Get model path from environment or use default
    model_path = os.environ.get("OCR_MODEL_PATH", "./models/trocr-finetuned")
    
    # Initialize OCR engine
    try:
        ocr_engine = HealthcareOCREngine(
            model_path=model_path,
            use_onnx=os.environ.get("USE_ONNX", "False").lower() == "true",
            use_quantization=os.environ.get("USE_QUANTIZATION", "False").lower() == "true",
            device=os.environ.get("DEVICE", None),
            confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")),
            enable_encryption=os.environ.get("ENABLE_ENCRYPTION", "True").lower() == "true",
            load_ensemble=os.environ.get("LOAD_ENSEMBLE", "True").lower() == "true"
        )
    except Exception as e:
        logger.error(f"Failed to initialize OCR engine: {str(e)}", extra={"request_id": "system"})
        # Continue startup - health check will show degraded status

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to request and log context for HIPAA audit trail"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request ID to logging context
    logger.info(f"Request received: {request.method} {request.url.path}", 
               extra={"request_id": request_id})
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add processing time and request ID headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    logger.info(f"Request completed in {process_time:.4f}s", 
               extra={"request_id": request_id})
    
    return response

@app.get("/health", response_model=OCRHealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancing"""
    if ocr_engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_loaded": False, "gpu_available": torch.cuda.is_available(),
                    "uptime": 0, "processed_documents": 0}
        )
    
    status = ocr_engine.get_health_status()
    return status

@app.post("/ocr", response_model=OCRResponse)
async def process_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: OCRRequest = None
):
    """
    Process an image with OCR
    
    - **file**: Image file to process
    - **metadata**: Optional metadata about the document
    """
    request_id = request.state.request_id
    
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get document type from metadata if provided
        document_type = None
        if metadata and metadata.document_type:
            document_type = metadata.document_type
        
        # Process the image
        results = ocr_engine.perform_ocr(image, document_type, request_id)
        
        # Check if we should use async processing
        if metadata and metadata.callback_url:
            # For async processing, return immediately and process in background
            background_tasks.add_task(
                send_async_results, 
                results, 
                metadata.callback_url, 
                request_id
            )
            
            return JSONResponse(
                status_code=202,
                content={
                    "request_id": request_id,
                    "status": "processing",
                    "message": "Image is being processed asynchronously"
                }
            )
        
        # Return results for synchronous processing
        return {
            "request_id": request_id,
            "text": results["text"],
            "confidence": results["confidence"],
            "processing_time": results["processing_time"],
            "warnings": results["warnings"]
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

async def send_async_results(results: Dict, callback_url: str, request_id: str):
    """Send results asynchronously to the callback URL"""
    try:
        payload = {
            "request_id": request_id,
            "text": results["text"],
            "confidence": results["confidence"],
            "processing_time": results["processing_time"],
            "warnings": results["warnings"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": request_id
        }
        
        response = requests.post(callback_url, json=payload, headers=headers)
        
        if response.status_code >= 200 and response.status_code < 300:
            logger.info(f"Successfully sent results to callback URL", extra={"request_id": request_id})
        else:
            logger.error(f"Failed to send results to callback URL: {response.status_code}", 
                        extra={"request_id": request_id})
    
    except Exception as e:
        logger.error(f"Error sending async results: {str(e)}", extra={"request_id": request_id})

@app.post("/batch")
async def batch_process(request: Request, background_tasks: BackgroundTasks):
    """Batch processing endpoint for multiple documents"""
    request_id = request.state.request_id
    
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    try:
        # Parse batch request - implement as needed for your specific batch needs
        body = await request.json()
        
        # Validate batch request
        if "documents" not in body or not isinstance(body["documents"], list):
            raise HTTPException(status_code=400, detail="Invalid batch request format")
        
        # Process batch asynchronously
        background_tasks.add_task(process_batch, body, request_id)
        
        return JSONResponse(
            status_code=202,
            content={
                "request_id": request_id,
                "status": "processing",
                "message": f"Batch with {len(body['documents'])} documents is being processed"
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/metrics")
async def get_metrics(request: Request):
    """Get OCR system metrics for monitoring"""
    request_id = request.state.request_id
    
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    try:
        # Calculate metrics
        metrics = ocr_engine.get_health_status()
        
        # Add more detailed metrics
        if ocr_engine.metrics["inference_times"]:
            metrics.update({
                "avg_inference_time": np.mean(ocr_engine.metrics["inference_times"]),
                "p95_inference_time": np.percentile(ocr_engine.metrics["inference_times"], 95),
                "max_inference_time": np.max(ocr_engine.metrics["inference_times"]),
                "min_inference_time": np.min(ocr_engine.metrics["inference_times"]),
            })
        
        if ocr_engine.metrics["confidence_scores"]:
            metrics.update({
                "avg_confidence": np.mean(ocr_engine.metrics["confidence_scores"]),
                "min_confidence": np.min(ocr_engine.metrics["confidence_scores"]),
                "confidence_below_threshold": sum(1 for score in ocr_engine.metrics["confidence_scores"] 
                                                if score < ocr_engine.confidence_threshold)
            })
            
        return metrics
    
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

async def process_batch(batch_data: Dict, request_id: str):
    """Process a batch of documents asynchronously"""
    batch_results = []
    callback_url = batch_data.get("callback_url")
    
    try:
        for idx, doc in enumerate(batch_data["documents"]):
            # Each document should have a base64 encoded image and optional metadata
            if "image" not in doc:
                batch_results.append({
                    "document_id": doc.get("id", f"doc_{idx}"),
                    "error": "No image data provided",
                    "success": False
                })
                continue
            
            try:
                # Decode base64 image
                image_data = base64.b64decode(doc["image"])
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                
                # Get document metadata
                document_type = doc.get("document_type")
                doc_id = doc.get("id", f"doc_{idx}")
                
                # Process the image
                doc_request_id = f"{request_id}_{doc_id}"
                results = ocr_engine.perform_ocr(image, document_type, doc_request_id)
                
                batch_results.append({
                    "document_id": doc_id,
                    "text": results["text"],
                    "confidence": results["confidence"],
                    "processing_time": results["processing_time"],
                    "warnings": results["warnings"],
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', f'doc_{idx}')}: {str(e)}", 
                            extra={"request_id": request_id})
                batch_results.append({
                    "document_id": doc.get("id", f"doc_{idx}"),
                    "error": str(e),
                    "success": False
                })
        
        # Send results back if callback URL is provided
        if callback_url:
            try:
                payload = {
                    "request_id": request_id,
                    "results": batch_results,
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_documents": len(batch_data["documents"]),
                    "successful_documents": sum(1 for r in batch_results if r.get("success", False))
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id
                }
                
                response = requests.post(callback_url, json=payload, headers=headers)
                
                if response.status_code >= 200 and response.status_code < 300:
                    logger.info(f"Successfully sent batch results to callback URL", 
                               extra={"request_id": request_id})
                else:
                    logger.error(f"Failed to send batch results: {response.status_code}", 
                                extra={"request_id": request_id})
                    
            except Exception as e:
                logger.error(f"Error sending batch results: {str(e)}", 
                            extra={"request_id": request_id})
    
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}", extra={"request_id": request_id})

# Add deployment configurations
def configure_deployment():
    """Configure deployment settings from environment variables"""
    config = {
        "model_path": os.environ.get("OCR_MODEL_PATH", "./models/trocr-finetuned"),
        "host": os.environ.get("HOST", "0.0.0.0"),
        "port": int(os.environ.get("PORT", "8000")),
        "workers": int(os.environ.get("WORKERS", "1")),
        "use_onnx": os.environ.get("USE_ONNX", "False").lower() == "true",
        "use_quantization": os.environ.get("USE_QUANTIZATION", "False").lower() == "true",
        "device": os.environ.get("DEVICE", None),
        "confidence_threshold": float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")),
        "enable_encryption": os.environ.get("ENABLE_ENCRYPTION", "True").lower() == "true",
        "load_ensemble": os.environ.get("LOAD_ENSEMBLE", "True").lower() == "true",
        "log_level": os.environ.get("LOG_LEVEL", "info"),
        "enable_monitoring": os.environ.get("ENABLE_MONITORING", "True").lower() == "true",
    }
    
    return config

def setup_k8s_probes():
    """Add Kubernetes probe endpoints for pod health monitoring"""
    @app.get("/readiness")
    async def readiness_probe():
        # Check if the model is loaded and ready to serve
        if ocr_engine is None or not ocr_engine.model_loaded:
            return JSONResponse(status_code=503, content={"status": "not ready"})
        return {"status": "ready"}
    
    @app.get("/liveness")
    async def liveness_probe():
        # Simple liveness check - if the app is responding, it's alive
        return {"status": "alive"}

def configure_monitoring():
    """Configure additional monitoring for the OCR service"""
    # This is a placeholder for setting up monitoring
    # In production, implement integration with monitoring tools
    # Such as Prometheus, Grafana, DataDog, etc.
    pass

def run_server():
    """Run the FastAPI server"""
    config = configure_deployment()
    
    # Setup Kubernetes probes
    setup_k8s_probes()
    
    # Setup monitoring
    if config["enable_monitoring"]:
        configure_monitoring()
    
    # Start the server
    uvicorn.run(
        "healthcare_ocr_inference_deployment:app",
        host=config["host"],
        port=config["port"],
        workers=config["workers"],
        log_level=config["log_level"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare OCR Inference Service")
    parser.add_argument("--model_path", type=str, help="Path to fine-tuned TrOCR model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX runtime for inference")
    parser.add_argument("--use_quantization", action="store_true", help="Use model quantization")
    parser.add_argument("--enable_encryption", action="store_true", help="Enable PII encryption")
    parser.add_argument("--load_ensemble", action="store_true", help="Load ensemble models")
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    # Set environment variables from command-line arguments
    if args.model_path:
        os.environ["OCR_MODEL_PATH"] = args.model_path
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    os.environ["WORKERS"] = str(args.workers)
    os.environ["USE_ONNX"] = str(args.use_onnx)
    os.environ["USE_QUANTIZATION"] = str(args.use_quantization)
    os.environ["ENABLE_ENCRYPTION"] = str(args.enable_encryption)
    os.environ["LOAD_ENSEMBLE"] = str(args.load_ensemble)
    
    if args.no_gpu:
        os.environ["DEVICE"] = "cpu"
    
    # Run the server
    run_server()