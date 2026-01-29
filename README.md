# DigiSaka GPT - AI-Powered WebGIS Tool Orchestrator

## Overview

DigiSaka GPT is an intelligent interface layer that bridges natural language interactions with DigiSaka WebGIS analytical tools. It uses a **fine-tuned language model** to understand user requests, generate appropriate API payloads, and convert technical responses into human-friendly language.

Users can upload images and request analyses in plain language (e.g., "count trees in this image"), and the fine-tuned model automatically determines which WebGIS tools to run and translates the results into readable responses.

**Current Status**: Prototype phase with support for:
- Tree Detection
- Crown Segmentation

> **Note**: This is a prototype implementation demonstrating the fine-tuning approach for WebGIS tool orchestration. The model is trained on synthetic data covering tree detection and crown segmentation workflows. Future versions will expand to additional geospatial analysis tools.

## Fine-Tuning Approach

DigiSaka GPT is built using a **fine-tuned language model** trained on two specific tasks:

### 1. Payload Generation (User Request → API JSON)
The model learns to convert natural language requests into properly formatted API payloads.

**Training Examples**:
```
User: "Detect trees in image img14156.png"
Output: {"tool": "tree_detection", "params": {"image_id": "img14156.png"}}

User: "Segment tree crowns in image img52843.png"
Output: {"tool": "crown_segmentation", "params": {"image_id": "img52843.png"}}
```

### 2. Response Translation (API JSON → Human Language)
The model learns to convert technical API responses into conversational, human-readable text.

**Training Examples**:
```
API Response: {"success": true, "total_detections": 155}
Output: "The tree detection tool successfully detected 155 trees in the uploaded image."

API Response: {"success": true, "total_detections": 243, "total_segments": 238}
Output: "The crown segmentation tool successfully detected 243 trees and segmented 238 crowns in the uploaded image."
```

### Training Dataset
- **1000+ payload conversion examples** (user input → API JSON)
- **20+ response translation examples** (API JSON → human text)
- Covers both tree detection and crown segmentation workflows
- Generated using systematic data augmentation

### Fine-Tuning Stack
- **Framework**: Unsloth + TRL (Transformer Reinforcement Learning)
- **Base Model**: Llama-based architecture
- **Optimization**: 4-bit quantization with bitsandbytes
- **Training**: Supervised fine-tuning (SFT) on instruction-following tasks

## How It Works

### Architecture Flow

```
User Input (Natural Language + Image)
          ↓
DigiSaka GPT (Fine-tuned LLM)
    [Payload Generation]
          ↓
Tool Selection Response (JSON)
          ↓
  Python Dispatcher
          ↓
DigiSaka WebGIS API Call
          ↓
  Raw API Response
          ↓
DigiSaka GPT (Fine-tuned LLM)
  [Response Translation]
          ↓
Human-Friendly Response
          ↓
         User
```

### Workflow Steps

1. **User Request**: User uploads an image and asks a question in natural language
   - Example: *"Count the trees in this image"*

2. **GPT Analysis**: DigiSaka GPT analyzes the request and determines:
   - Which WebGIS tool(s) to use
   - Required parameters
   - Processing workflow

3. **Tool Dispatch**: Python dispatcher receives GPT's tool selection and:
   - Constructs WebGIS API payload
   - Triggers the appropriate tool
   - Handles the API communication

4. **Result Processing**: Raw WebGIS response is sent back to DigiSaka GPT

5. **Human Translation**: GPT converts technical output to natural language
   - Example: *"55 trees were detected in your uploaded image"*

## Current Prototype

The current prototype supports two main WebGIS tools:
- **Tree Detection**: Identifies and counts individual trees in aerial/satellite imagery
- **Crown Segmentation**: Detects trees and segments their crown boundaries

### Fine-Tuning Approach

DigiSaka GPT is fine-tuned using synthetically generated training data with two key capabilities:

#### 1. User Request → API Payload Conversion
The model learns to convert natural language requests into structured API payloads.

**Training Example:**
```json
{
  "instruction": "Convert user request into API payload JSON only.",
  "input": "Detect trees in image img14156.png.",
  "output": "{\"tool\": \"tree_detection\", \"params\": {\"image_id\": \"img14156.png\"}}"
}
```

#### 2. API Response → Human-Friendly Text Conversion
The model learns to convert technical API responses into natural language summaries.

**Training Example:**
```json
{
  "instruction": "Convert API JSON result into human readable answer.",
  "input": "{\"success\": true, \"total_detections\": 245}",
  "output": "The tree detection tool successfully detected 245 trees in the uploaded image."
}
```

### Training Dataset Generation

The fine-tuning dataset is automatically generated with:
- **1,000 payload examples**: User requests → API payloads (tree_detection & crown_segmentation)
- **20 response examples**: API results → Human-friendly summaries
- **Random variations**: Different image IDs, detection counts (100-300), and tool selections

This synthetic approach allows rapid iteration and expansion to new tools without manual labeling.

## Key Components

### 1. DigiSaka GPT (Fine-Tuned Language Model)
- **Role**: Natural language understanding and response generation
- **Base Model**: Fine-tuned using Unsloth framework
- **Training**: Synthetic dataset with payload/response conversion pairs
- **Responsibilities**:
  - Interpret user requests
  - Generate WebGIS API payloads
  - Convert API responses to human-friendly text
  - Maintain conversation context

### 2. Python Dispatcher
- **Role**: Bridge between GPT and WebGIS API
- **Responsibilities**:
  - Parse GPT's tool selection
  - Construct API payloads
  - Execute WebGIS tool calls
  - Handle errors and retries
  - Return raw results to GPT

### 3. DigiSaka WebGIS Tools
- **Role**: Perform actual geospatial analysis
- **Examples**:
  - Tree detection and counting
  - Land cover classification
  - Object detection
  - Change detection
  - Spatial analysis

## Payload & Response Examples

The following examples demonstrate the current **prototype implementation** with tree detection and crown segmentation tools. The fine-tuned model handles both payload generation and response translation.

### Example 1: Tree Detection

**User Input:**
```
"Detect trees in image img54321.png"
```

**GPT → Dispatcher (Tool Selection):**
```json
{
  "tool": "tree_detection",
  "params": {
    "image_id": "img54321.png"
  }
}
```

**Dispatcher → WebGIS API (Payload):**
```json
{
  "service": "object_detection",
  "tool": "tree_detection",
  "input": {
    "image_id": "img54321.png"
  }
}
```

**WebGIS API → Dispatcher (Raw Response):**
```json
{
  "success": true,
  "total_detections": 245,
  "processing_time": "1.8s"
}
```

**Dispatcher → GPT (Forwarded Response):**
```json
{
  "success": true,
  "total_detections": 245
}
```

**GPT → User (Human-Friendly Response):**
```
The tree detection tool successfully detected 245 trees in the uploaded image.
```

### Example 2: Crown Segmentation

**User Input:**
```
"Segment tree crowns in my uploaded image"
```

**GPT → Dispatcher (Tool Selection):**
```json
{
  "tool": "crown_segmentation",
  "params": {
    "image_id": "img67890.png"
  }
}
```

**WebGIS API → Dispatcher (Raw Response):**
```json
{
  "success": true,
  "total_detections": 187,
  "total_segments": 184,
  "processing_time": "3.2s"
}
```

**GPT → User (Human-Friendly Response):**
```
The crown segmentation tool successfully detected 187 trees and segmented 184 crowns in the uploaded image.
```

## Configuration

### Environment Variables
```bash
DIGISAKA_API_KEY=your_api_key
DIGISAKA_API_ENDPOINT=https://api.digisaka.com/v1
FINETUNED_MODEL_PATH=./models/digisaka-gpt-v1
MAX_IMAGE_SIZE=10485760  # 10MB
TIMEOUT_SECONDS=300
```

### Supported WebGIS Tools (Current Prototype)
- `tree_detection` - Count and locate individual trees in imagery
- `crown_segmentation` - Detect trees and segment their crown boundaries

### Future Tools (Planned)
- `building_detection` - Identify structures
- `land_cover_classification` - Classify terrain types
- `change_detection` - Compare temporal changes
- `vegetation_index` - Calculate NDVI, EVI, etc.
- `water_body_detection` - Identify water features
- `road_extraction` - Extract road networks

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/digisaka-gpt.git
cd digisaka-gpt
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt

# For fine-tuning (optional)
pip install unsloth trl peft accelerate bitsandbytes datasets
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your DigiSaka API credentials
```

### 4. Download or Fine-tune the Model

**Option A: Use Pre-trained Model**
```bash
# Download the fine-tuned model
python scripts/download_model.py
```

**Option B: Fine-tune Your Own Model**
```bash
# Generate training dataset
python generate_training_data.py

# Fine-tune the model
python fine_tune.py --base_model "unsloth/llama-3-8b" \
                    --dataset "generated_dataset.json" \
                    --output_dir "./models/digisaka-gpt-v1"
```

### 5. Run the Application
```bash
python main.py
```

## Quick Start - Prototype Workflow

### Example 1: Tree Detection

```python
from digisaka_gpt import DigiSakaGPT

# Initialize with fine-tuned model
gpt = DigiSakaGPT(model_path="./models/digisaka-gpt-v1")

# User uploads image and asks
user_input = "Detect trees in image img12345.png"

# Step 1: GPT generates API payload
payload = gpt.generate_payload(user_input)
# Output: {"tool": "tree_detection", "params": {"image_id": "img12345.png"}}

# Step 2: Python dispatcher calls WebGIS API
api_response = dispatcher.call_webgis_api(payload)
# Returns: {"success": true, "total_detections": 187}

# Step 3: GPT converts response to human language
human_response = gpt.translate_response(api_response)
# Output: "The tree detection tool successfully detected 187 trees in the uploaded image."

print(human_response)
```

### Example 2: Crown Segmentation

```python
user_input = "Segment tree crowns in image img67890.png"

# GPT generates payload
payload = gpt.generate_payload(user_input)
# {"tool": "crown_segmentation", "params": {"image_id": "img67890.png"}}

# API returns results
api_response = dispatcher.call_webgis_api(payload)
# {"success": true, "total_detections": 243, "total_segments": 238}

# GPT translates
human_response = gpt.translate_response(api_response)
# "The crown segmentation tool successfully detected 243 trees and segmented 238 crowns in the uploaded image."

print(human_response)
```

## Usage

### Basic Example - Tree Detection
```python
from digisaka_gpt import DigiSakaGPT

# Initialize
gpt = DigiSakaGPT(model_path="path/to/finetuned/model")

# Process user request
response = gpt.process_request(
    user_message="Detect trees in image img12345.png"
)

print(response)
# Output: "The tree detection tool successfully detected 245 trees in the uploaded image."
```

### Crown Segmentation Example
```python
# Segment tree crowns
response = gpt.process_request(
    user_message="Segment tree crowns in my aerial photo"
)

print(response)
# Output: "The crown segmentation tool successfully detected 187 trees and segmented 184 crowns in the uploaded image."
```

### Advanced Usage - Accessing Raw API Response
```python
# Get both human-friendly and raw responses
result = gpt.process_request(
    user_message="Count trees in img99999.png",
    return_raw=True
)

print(result['human_response'])
# "The tree detection tool successfully detected 320 trees in the uploaded image."

print(result['api_response'])
# {"success": true, "total_detections": 320, "processing_time": "2.1s"}
```

## API Response Format

All responses from DigiSaka GPT follow this structure:

```json
{
  "status": "success|error",
  "message": "Human-friendly response text",
  "data": {
    "tools_used": ["tool1", "tool2"],
    "processing_time": "3.2s",
    "raw_results": {...}
  },
  "metadata": {
    "timestamp": "2024-01-29T10:30:00Z",
    "model_version": "1.0.0"
  }
}
```

## Error Handling

DigiSaka GPT gracefully handles errors and provides helpful feedback:

```
User: "Count cars in this image"
GPT: "I apologize, but the image quality is too low for accurate vehicle 
detection. Could you please upload a higher resolution image? For best 
results, images should be at least 1024x1024 pixels."
```

## Benefits

1. **Natural Language Interface**: No need to understand technical GIS parameters
2. **Intelligent Tool Selection**: Fine-tuned model automatically chooses the right tools
3. **Human-Friendly Output**: Technical results translated to plain language
4. **Specialized for WebGIS**: Training focused on geospatial analysis tasks
5. **Fast & Efficient**: Optimized with 4-bit quantization for deployment

## Model Training

### Training Notebook
The fine-tuning process is documented in `Fine_Tuning.ipynb`, which includes:

1. **Dataset Generation**
   - Automated generation of 1,000+ payload conversion examples
   - 20+ response translation examples
   - Random variation in image IDs and detection counts
   - Covers both tree detection and crown segmentation

2. **Fine-tuning Process**
   ```python
   # Key training parameters
   num_payload_examples = 1000   # Payload generation training data
   num_response_examples = 20     # Response translation training data
   
   # Uses Unsloth for efficient fine-tuning
   # Supports 4-bit quantization for faster training
   ```

3. **Training Tasks**
   - **Task 1**: Convert user requests → API JSON payloads
   - **Task 2**: Convert API responses → Human-readable text

### Sample Training Data

**Payload Generation Example:**
```json
{
  "instruction": "Convert user request into API payload JSON only.",
  "input": "Detect trees in image img14156.png.",
  "output": "{\"tool\": \"tree_detection\", \"params\": {\"image_id\": \"img14156.png\"}}"
}
```

**Response Translation Example:**
```json
{
  "instruction": "Convert API JSON result into human readable answer.",
  "input": "{\"success\": true, \"total_detections\": 243, \"total_segments\": 238}",
  "output": "The crown segmentation tool successfully detected 243 trees and segmented 238 crowns in the uploaded image."
}
```

### Running the Training

1. Open `Fine_Tuning.ipynb` in Google Colab or Jupyter
2. Install dependencies (Unsloth, TRL, bitsandbytes)
3. Generate training dataset
4. Run fine-tuning script
5. Export and save the trained model

## Technical Stack

- **Language Model**: Fine-tuned LLM using Unsloth framework
- **Training Framework**: Unsloth, TRL (Transformer Reinforcement Learning), PEFT
- **Backend**: Python 3.8+
- **WebGIS API**: DigiSaka WebGIS Platform
- **Image Processing**: PIL, OpenCV
- **API Communication**: Requests, aiohttp
- **Fine-tuning**: Synthetic dataset generation (1,000+ examples)

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

[Your License Here]

---

**DigiSaka GPT** - Making geospatial analysis accessible through natural language
