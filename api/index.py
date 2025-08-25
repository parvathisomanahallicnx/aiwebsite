import os
import json
import re
import requests
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# === ENV CONFIG ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PRODUCT_SEARCH_MCP_URL = "https://n5ycpj-wu.myshopify.com/api/mcp"
ORDER_MCP_URL = "https://cnx-demo-mcp-server.onrender.com/api/mcp"

# === MCP SERVER INTEGRATION ===
def call_mcp_server(url: str, tool_name: str, arguments: dict) -> dict:
    """Generic MCP server call function"""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 1
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if "result" in result and "content" in result["result"]:
            content = result["result"]["content"]
            if content and "text" in content[0]:
                return json.loads(content[0]["text"])
        
        return {"error": "Invalid MCP response format"}
    except Exception as e:
        return {"error": f"MCP server error: {str(e)}"}

# === LLM Setup using Google Generative AI directly ===
def call_gemini_llm(prompt: str) -> str:
    """Call Gemini LLM directly using google.generativeai"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        text = response.text if hasattr(response, 'text') else None
        
        if not text and response and getattr(response, "candidates", None):
            try:
                parts = []
                for c in response.candidates:
                    for p in getattr(c, "content", {}).parts:
                        if getattr(p, "text", None):
                            parts.append(p.text)
                text = "\n".join(parts)
            except Exception:
                text = None
        
        return text or ""
    except Exception as e:
        print(f"Gemini LLM call failed: {e}")
        return ""

# === STATE ===
class AgentState(TypedDict, total=False):
    user_message: str
    intent: str
    intent_details: dict
    products: dict
    order_result: dict
    order_status: dict
    info_result: dict
    final_response: str

# === INTENT ANALYSIS NODE ===
def analyze_user_intent(state: AgentState):
    """Analyze user message to determine intent: product_search, order_creation, order_status, or info_search"""
    
    prompt = f"""
    Analyze the user message and classify the intent. Return ONLY a JSON object with the following structure:
    {{
        "intent": "product_search" | "order_creation" | "order_status" | "info_search",
        "confidence": 0.0-1.0,
        "details": {{
            "extracted_info": "relevant information extracted from the message"
        }}
    }}

    Intent Classification Rules:
    - "product_search": User is looking for products, asking about availability, prices, or product information
    - "order_creation": User wants to buy/purchase/order something, mentions placing an order
    - "order_status": User wants to track/check order status, mentions order ID or tracking
    - "info_search": User is asking for business information such as return/exchange policy, contact details (phone/email/address), current offers/discounts/promotions

    User Message: "{state["user_message"]}"

    Examples:
    - "Show me floral shirts" -> product_search
    - "I want to buy this product" -> order_creation  
    - "What's the status of order 12345?" -> order_status
    - "Track my order" -> order_status
    - "What is your return policy?" -> info_search
    - "How can I contact support?" -> info_search
    - "Any offers or discounts right now?" -> info_search

    Return ONLY the JSON object, no other text.
    """
    
    result = call_gemini_llm(prompt)
    
    # Extract JSON from response
    try:
        # Remove markdown formatting if present
        cleaned = re.sub(r"```[a-zA-Z]*", "", result).strip("` \n")
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        
        if json_match:
            intent_data = json.loads(json_match.group())
            return {
                "intent": intent_data.get("intent", "product_search"),
                "intent_details": intent_data.get("details", {})
            }
    except Exception as e:
        print(f"Intent analysis error: {e}")
    
    # Fallback intent analysis
    message_lower = state["user_message"].lower()
    if any(word in message_lower for word in ["buy", "purchase", "order", "add to cart"]):
        return {"intent": "order_creation", "intent_details": {}}
    elif any(word in message_lower for word in ["track", "status", "order id", "tracking"]):
        return {"intent": "order_status", "intent_details": {}}
    elif any(word in message_lower for word in [
        "return", "refund", "exchange", "contact", "phone", "email", "support", "address",
        "offer", "discount", "sale", "promotion", "deal"
    ]):
        return {"intent": "info_search", "intent_details": {}}
    else:
        return {"intent": "product_search", "intent_details": {}}

# === PRODUCT SEARCH NODE ===
SEARCH_CONTEXT_TEMPLATE = (
    "Search Query: {message}\n"
    "Filtering Guidelines:\n"
    "- Prioritize products that match the search terms in title, description, or tags\n"
    "- For patterns (floral, striped, etc.): prefer products with matching patterns\n"
    "- For product types: include relevant category matches\n"
    "- For price constraints: filter by specified price ranges\n"
    "- Return relevant products even if not exact matches\n"
    "- Include similar or related products when appropriate"
)

def llm_parse_query(user_query: str) -> dict:
    """Extract structured shopping intent from user query"""
    prompt = (
        "Extract structured shopping intent from the following message and return STRICT JSON only. "
        "IMPORTANT: For pattern searches (floral, striped, etc.), include the pattern in BOTH 'query' and 'filters.design' fields.\n"
        "Fields: query (full search text including patterns), filters.price {min,max}, filters.availability (true|false|null), "
        "filters.sizes (array of strings), filters.colors (array of strings), filters.design (array of pattern keywords).\n"
        f"Message: '{user_query}'\n"
        "Examples:\n"
        "- 'floral shirts' -> {\"query\":\"floral shirts\",\"filters\":{\"design\":[\"floral\"]}}\n"
        "- 'striped dresses under 2000' -> {\"query\":\"striped dresses\",\"filters\":{\"price\":{\"max\":2000},\"design\":[\"striped\"]}}\n"
        "Output JSON:"
    )
    try:
        llm_out = call_gemini_llm(prompt)
        if llm_out:
            start = llm_out.find('{')
            end = llm_out.rfind('}')
            if start != -1 and end != -1:
                json_str = llm_out[start:end+1]
                return json.loads(json_str)
    except Exception as e:
        print(f"[DEBUG] Gemini LLM parse failed: {e}")
    # fallback
    return {"query": user_query, "filters": {}}

def product_search_node(state: AgentState):
    """Handle product search requests using direct MCP calls with pre-filtering"""
    try:
        # Step 1: Parse user query to extract structured filters
        parsed = llm_parse_query(state["user_message"])
        mcp_query = parsed.get("query", state["user_message"])
        context = SEARCH_CONTEXT_TEMPLATE.format(message=state["user_message"])
        
        # Step 2: Build MCP arguments with extracted filters
        arguments = {
            "query": mcp_query,
            "context": context,
        }
        
        # Map supported filters to top-level args expected by the MCP tool
        filters = parsed.get("filters", {}) or {}
        if isinstance(filters.get("price"), dict):
            arguments["price"] = filters["price"]
        if "availability" in filters:
            arguments["availability"] = filters.get("availability")
        
        print(f"[DEBUG] Query: '{state['user_message']}' | Parsed: {parsed} | MCP Args: {arguments}")
        
        # Step 3: Call MCP server with structured arguments
        mcp_result = call_mcp_server(PRODUCT_SEARCH_MCP_URL, "search_shop_catalog", arguments)
        
        # Extract products from MCP response
        raw_products = []
        if isinstance(mcp_result, dict) and "products" in mcp_result:
            raw_products = mcp_result["products"]
        elif isinstance(mcp_result, dict) and "error" in mcp_result:
            error_response = {"error": f"Product search failed: {mcp_result['error']}"}
            return {
                "products": error_response,
                "final_response": json.dumps(error_response, indent=2)
            }
        
        # If no raw products, return empty result with debug info
        if not raw_products:
            debug_response = {
                "products": [],
                "debug": {
                    "message": "No products returned from MCP server",
                    "mcp_response": mcp_result
                }
            }
            return {
                "products": debug_response,
                "final_response": json.dumps(debug_response, indent=2)
            }
        
        # Use AI for comprehensive filtering and formatting
        filter_prompt = f"""
        You are an intelligent product search assistant. Analyze the user query and filter the products based on ALL criteria mentioned.

        User Query: "{state["user_message"]}"

        INTELLIGENT FILTERING RULES:
        1. PRICE FILTERING:
           - "under X", "below X", "less than X" → include products where ALL variants ≤ X
           - "over X", "above X", "more than X" → include products where ALL variants ≥ X
           - "between X and Y" → include products where ALL variants are X ≤ price ≤ Y
           - "around X", "approximately X" → include products within ±20% of X

        2. PATTERN/DESIGN FILTERING:
           - "floral", "striped", "polka dot", etc. → match in title, description, or product type
           - Be flexible with variations (e.g., "flower" matches "floral")

        3. PRODUCT TYPE FILTERING:
           - "shirts", "dresses", "earrings", etc. → match product_type or title
           - Include related types (e.g., "tops" includes shirts, blouses, t-shirts)

        4. COLOR FILTERING:
           - Match colors in title or variant titles
           - Include color variations (e.g., "blue" matches "navy", "royal blue")

        5. SIZE FILTERING:
           - Match sizes in variant titles
           - Consider size ranges (S, M, L, XL, etc.)

        6. AVAILABILITY FILTERING:
           - Only include products that appear to be available/in-stock

        CRITICAL INSTRUCTIONS:
        - Apply ALL filters mentioned in the user query
        - Be strict but intelligent (use semantic understanding)
        - If no products match ALL criteria, return empty products array
        - Preserve original product structure exactly

        Required JSON format:
        {{
          "products": [
            {{
              "id": product_id,
              "title": "Product Title",
              "product_type": "Product Type",
              "variants": [
                {{
                  "id": variant_id,
                  "title": "Variant Title",
                  "price": "Price"
                }}
              ],
              "images": [
                {{
                  "id": image_id,
                  "src": "image_url"
                }}
              ]
            }}
          ]
        }}

        Raw product data to filter: {json.dumps(raw_products, indent=2)}

        Return ONLY the filtered JSON with products that match ALL criteria, no other text.
        """
        
        formatted_result = call_gemini_llm(filter_prompt)
        
        # Parse the AI-filtered result
        try:
            # Clean up the response to extract JSON
            cleaned = re.sub(r"```[a-zA-Z]*", "", formatted_result).strip("` \n")
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            
            if json_match:
                formatted_products = json.loads(json_match.group())
                return {
                    "products": formatted_products,
                    "final_response": json.dumps(formatted_products, indent=2)
                }
        except Exception as e:
            print(f"[DEBUG] AI filtering failed: {e}")
        
        # Fallback: return raw products if AI filtering fails
        fallback_response = {"products": raw_products}
        return {
            "products": fallback_response,
            "final_response": json.dumps(fallback_response, indent=2)
        }
        
    except Exception as e:
        error_response = {"error": f"Product search error: {str(e)}"}
        return {
            "products": error_response,
            "final_response": json.dumps(error_response, indent=2)
        }

# === ORDER CREATION NODE ===
def order_creation_node(state: AgentState):
    """Handle order creation requests"""
    try:
        # Extract product details from user message
        prompt = f"""
        Extract product information from the user message for order creation. Return ONLY a JSON object:
        {{
            "product_id": "extracted product ID if mentioned",
            "product_name": "product name or description",
            "quantity": number,
            "found_product": true/false
        }}

        User Message: "{state["user_message"]}"

        Look for product IDs, names, quantities. Return ONLY the JSON object.
        """
        
        result = call_gemini_llm(prompt)
        
        try:
            cleaned = re.sub(r"```[a-zA-Z]*", "", result).strip("` \n")
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            
            if json_match:
                product_info = json.loads(json_match.group())
                
                if not product_info.get("found_product", False):
                    error_response = {"error": "Please specify a product to order."}
                    return {
                        "order_result": error_response,
                        "final_response": json.dumps(error_response, indent=2)
                    }
                
                # Create order using MCP server
                order_data = {
                    "product_id": product_info.get("product_id", "1"),
                    "quantity": product_info.get("quantity", 1),
                    "product_name": product_info.get("product_name", "Product")
                }
                
                raw_order_result = call_mcp_server(ORDER_MCP_URL, "create_order", order_data)
                
                # Format response
                format_prompt = f"""
                Format the order creation result into the exact JSON structure below:

                Required JSON format:
                {{
                  "order_created": {{
                    "id": "ORDER_ID",
                    "order_id": "ORDER_NUMBER",
                    "product": "PRODUCT_TITLE",
                    "total_paid": "AMOUNT INR",
                    "message": "Your order has been placed successfully! Use the ID: ORDER_ID to track your order status at any time."
                  }}
                }}

                Raw order result: {json.dumps(raw_order_result, indent=2)}

                Extract the order ID, order number, product title, and total amount from the raw data.
                Return ONLY the formatted JSON, no other text.
                """
                
                formatted_result = call_gemini_llm(format_prompt)
                
                try:
                    cleaned = re.sub(r"```[a-zA-Z]*", "", formatted_result).strip("` \n")
                    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    
                    if json_match:
                        formatted_order = json.loads(json_match.group())
                        return {
                            "order_result": formatted_order,
                            "final_response": json.dumps(formatted_order, indent=2)
                        }
                except Exception as e:
                    print(f"Order formatting error: {e}")
                
                return {
                    "order_result": raw_order_result,
                    "final_response": json.dumps(raw_order_result, indent=2)
                }
                
        except Exception as e:
            error_response = {"error": f"Order parsing failed: {str(e)}"}
            return {
                "order_result": error_response,
                "final_response": json.dumps(error_response, indent=2)
            }
            
    except Exception as e:
        error_response = {"error": f"Order creation failed: {str(e)}"}
        return {
            "order_result": error_response,
            "final_response": json.dumps(error_response, indent=2)
        }

# === ORDER STATUS NODE ===
def order_status_node(state: AgentState):
    """Handle order status requests"""
    try:
        prompt = f"""
        Extract the order ID from the user message. Return ONLY a JSON object:
        {{
            "order_id": "extracted order ID",
            "found": true/false
        }}

        User Message: "{state["user_message"]}"

        Look for numbers that could be order IDs. Return ONLY the JSON object.
        """
        
        result = call_gemini_llm(prompt)
        
        try:
            cleaned = re.sub(r"```[a-zA-Z]*", "", result).strip("` \n")
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            
            if json_match:
                order_info = json.loads(json_match.group())
                
                if not order_info.get("found", False):
                    error_response = {"error": "Please provide a valid order ID to check status."}
                    return {
                        "order_status": error_response,
                        "final_response": json.dumps(error_response, indent=2)
                    }
                
                try:
                    order_id = int(order_info["order_id"])
                    raw_status_result = call_mcp_server(ORDER_MCP_URL, "get_order_status", {"order_id": order_id})
                except (ValueError, TypeError):
                    error_response = {"error": "Invalid order ID format."}
                    return {
                        "order_status": error_response,
                        "final_response": json.dumps(error_response, indent=2)
                    }
                
                return {
                    "order_status": raw_status_result,
                    "final_response": json.dumps(raw_status_result, indent=2)
                }
                
        except Exception as e:
            error_response = {"error": f"Order ID parsing failed: {str(e)}"}
            return {
                "order_status": error_response,
                "final_response": json.dumps(error_response, indent=2)
            }
            
    except Exception as e:
        error_response = {"error": f"Order status check failed: {str(e)}"}
        return {
            "order_status": error_response,
            "final_response": json.dumps(error_response, indent=2)
        }

# === INFO SEARCH NODE ===
def info_search_node(state: AgentState):
    """Handle informational queries using Gemini LLM"""
    try:
        # Use Gemini to provide business information
        prompt = f"""
        You are a customer service assistant for an e-commerce store. Answer the following customer query with helpful, accurate information:

        Customer Query: "{state["user_message"]}"

        Provide information about:
        - Return and exchange policies
        - Contact information (phone, email, address)
        - Current offers, discounts, or promotions
        - Shipping and delivery information
        - General business information

        If you don't have specific information, provide general helpful guidance and suggest contacting customer support.
        
        Format your response as a helpful, professional customer service response.
        """
        
        response = call_gemini_llm(prompt)
        
        info_result = {
            "info_response": response,
            "query": state["user_message"]
        }
        
        return {
            "info_result": info_result,
            "final_response": json.dumps(info_result, indent=2)
        }
        
    except Exception as e:
        error_response = {"error": f"Info search failed: {str(e)}"}
        return {
            "info_result": error_response,
            "final_response": json.dumps(error_response, indent=2)
        }

# === WORKFLOW SETUP ===
def process_user_message(user_message: str) -> dict:
    """Process user message through the complete LangGraph workflow"""
    try:
        # Initialize state
        state = AgentState(user_message=user_message)
        
        # Step 1: Analyze intent
        intent_result = analyze_user_intent(state)
        state.update(intent_result)
        
        # Step 2: Route to appropriate node based on intent
        if state["intent"] == "product_search":
            result = product_search_node(state)
        elif state["intent"] == "order_creation":
            result = order_creation_node(state)
        elif state["intent"] == "order_status":
            result = order_status_node(state)
        elif state["intent"] == "info_search":
            result = info_search_node(state)
        else:
            # Default to product search
            result = product_search_node(state)
        
        # Update state with result
        state.update(result)
        
        return {
            "intent": state["intent"],
            "intent_details": state.get("intent_details", {}),
            "final_response": state.get("final_response", ""),
            "full_state": dict(state)
        }
        
    except Exception as e:
        error_response = {
            "intent": "error",
            "intent_details": {"error": str(e)},
            "final_response": json.dumps({"error": f"Workflow processing failed: {str(e)}"}, indent=2),
            "full_state": {"error": str(e)}
        }
        return error_response

# === FASTAPI SETUP ===
app = FastAPI(title="LangGraph Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    messages: List[Dict[str, str]]

class AgentResponse(BaseModel):
    chat_message: str
    intent: Optional[str] = None
    intent_details: Optional[Dict[str, Any]] = None
    inner_messages: Optional[List[Dict[str, Any]]] = None
    user_intent: Optional[str] = None

@app.post("/agent-assistant/", response_model=AgentResponse)
async def agent_assistant(request: MessageRequest):
    """Process user messages through the complete LangGraph workflow"""
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Get the last user message
        last_message = None
        for msg in reversed(request.messages):
            if msg.get("source") == "user":
                last_message = msg.get("content", "")
                break
        
        if not last_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Process through complete LangGraph workflow
        result = process_user_message(last_message)
        
        # Get the formatted response from the workflow
        chat_message = result.get("final_response", "")
        
        return AgentResponse(
            chat_message=chat_message,
            intent=result.get("intent"),
            intent_details=result.get("intent_details"),
            inner_messages=[result.get("full_state", {})],
            user_intent=result.get("intent") or result.get("user_intent")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LangGraph Agent API"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "LangGraph Agent API is running on Vercel"}
