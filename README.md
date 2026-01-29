# DigiSaka GPT - AI-Powered WebGIS Tool Orchestrator

## Overview

DigiSaka GPT is an intelligent interface layer that bridges natural language interactions with DigiSaka WebGIS analytical tools. It allows users to upload images and request analyses in plain language (e.g., "count trees in this image"), automatically determines which WebGIS tools to run, and returns results in a human-friendly format.

## How It Works

### Architecture Flow

```
User Input (Natural Language + Image)
          ↓
    DigiSaka GPT (LLM)
          ↓
Tool Selection Response (JSON)
          ↓
  Python Dispatcher
          ↓
DigiSaka WebGIS API Call
          ↓
  Raw API Response
          ↓
    DigiSaka GPT (LLM)
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

## Key Components

### 1. DigiSaka GPT (Language Model)
- **Role**: Natural language understanding and response generation
- **Responsibilities**:
  - Interpret user requests
  - Select appropriate WebGIS tools
  - Format technical responses into human-readable text
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

### Example 1: Tree Detection

**User Input:**
```
"How many trees are in this image?"
[uploads aerial image]
```

**GPT → Dispatcher (Tool Selection):**
```json
{
  "tool": "tree_detection",
  "parameters": {
    "image_id": "user_upload_12345",
    "detection_threshold": 0.7,
    "output_format": "count_and_coordinates"
  }
}
```

**Dispatcher → WebGIS API (Payload):**
```json
{
  "service": "object_detection",
  "tool": "tree_counter",
  "input": {
    "image_url": "https://storage.digisaka.com/uploads/12345.jpg",
    "model": "tree_detection_v2",
    "confidence_threshold": 0.7
  },
  "output": {
    "format": "geojson",
    "include_metadata": true
  }
}
```

**WebGIS API → Dispatcher (Raw Response):**
```json
{
  "status": "success",
  "results": {
    "total_count": 55,
    "high_confidence": 48,
    "medium_confidence": 7,
    "bounding_boxes": [...],
    "processing_time": "2.3s"
  },
  "metadata": {
    "image_size": "4096x4096",
    "model_version": "2.1.0"
  }
}
```

**Dispatcher → GPT (Forwarded Response):**
```json
{
  "status": "success",
  "results": {
    "total_count": 55,
    "high_confidence": 48,
    "medium_confidence": 7
  }
}
```

**GPT → User (Human-Friendly Response):**
```
I've analyzed your image and detected 55 trees in total. Of these, 48 were 
detected with high confidence and 7 with medium confidence. The analysis 
completed in 2.3 seconds.
```

### Example 2: Multi-Tool Workflow

**User Input:**
```
"Analyze the vegetation in this area and tell me if there's been any change 
compared to last year"
[uploads two images]
```

**GPT → Dispatcher (Tool Selection):**
```json
{
  "workflow": "sequential",
  "tools": [
    {
      "step": 1,
      "tool": "vegetation_index",
      "parameters": {
        "image_id": "current_image",
        "index_type": "NDVI"
      }
    },
    {
      "step": 2,
      "tool": "change_detection",
      "parameters": {
        "image1_id": "previous_image",
        "image2_id": "current_image",
        "analysis_type": "vegetation_change"
      }
    }
  ]
}
```

**GPT → User (Final Response):**
```
I've analyzed the vegetation in both images. The current NDVI (vegetation health 
index) shows moderate to high vegetation density across 78% of the area. 

Comparing with last year's image, I detected:
- 12% increase in vegetated areas (primarily in the northern section)
- 5% decrease in the southern region (possibly due to clearing or drought)
- Overall net gain of 7% in healthy vegetation

The changes are most significant in a 3.2 hectare area in the northeast corner, 
which appears to have undergone reforestation.
```

## Configuration

### Environment Variables
```bash
DIGISAKA_API_KEY=your_api_key
DIGISAKA_API_ENDPOINT=https://api.digisaka.com/v1
GPT_MODEL=gpt-4
MAX_IMAGE_SIZE=10485760  # 10MB
TIMEOUT_SECONDS=300
```

### Supported WebGIS Tools
- `tree_detection` - Count and locate trees
- `building_detection` - Identify structures
- `land_cover_classification` - Classify terrain types
- `change_detection` - Compare temporal changes
- `vegetation_index` - Calculate NDVI, EVI, etc.
- `water_body_detection` - Identify water features
- `road_extraction` - Extract road networks

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/digisaka-gpt.git
cd digisaka-gpt

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run the application
python main.py
```

## Usage

### Basic Example
```python
from digisaka_gpt import DigiSakaGPT

# Initialize
gpt = DigiSakaGPT(api_key="your_key")

# Process user request
response = gpt.process_request(
    user_message="Count trees in this image",
    image_path="aerial_image.jpg"
)

print(response)
# Output: "55 trees were detected in your uploaded image"
```

### Advanced Example with Custom Workflow
```python
# Complex multi-step analysis
response = gpt.process_request(
    user_message="""
    Analyze this agricultural field and tell me:
    1. Total area
    2. Crop health status
    3. Any signs of disease or stress
    """,
    image_path="farm_field.jpg",
    options={
        "detailed_analysis": True,
        "include_recommendations": True
    }
)
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
2. **Intelligent Tool Selection**: GPT automatically chooses the right tools
3. **Human-Friendly Output**: Technical results translated to plain language
4. **Multi-Tool Orchestration**: Complex workflows handled seamlessly
5. **Context Awareness**: Maintains conversation history for follow-up questions

## Technical Stack

- **Language Model**: GPT-4 / Custom LLM
- **Backend**: Python 3.8+
- **WebGIS API**: DigiSaka WebGIS Platform
- **Image Processing**: PIL, OpenCV
- **API Communication**: Requests, aiohttp

---

**DigiSaka GPT** - Making geospatial analysis accessible through natural language
