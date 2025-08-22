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
from dotenv import load_dotenv

# === ENV CONFIG ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBA1GSp3jXVMsLDSUapgJ9bFTq0p8ZOZYM")
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
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        # Extract text from response
        text = getattr(response, "text", None)
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
# Context sent to MCP to guide server-side filtering
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
        
        # Extract JSON from LLM response
        try:
            cleaned = re.sub(r"```[a-zA-Z]*", "", formatted_result).strip("` \n")
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            
            if json_match:
                formatted_products = json.loads(json_match.group())
                return {
                    "products": formatted_products,
                    "final_response": json.dumps(formatted_products, indent=2)
                }
        except Exception as e:
            print(f"LLM filtering failed: {e}")
        
        # Fallback: return raw products with minimal structure
        fallback_response = {"products": raw_products}
        return {
            "products": fallback_response,
            "final_response": json.dumps(fallback_response, indent=2)
        }
        
    except Exception as e:
        error_response = {"error": f"Product search failed: {str(e)}"}
        return {
            "products": error_response,
            "final_response": json.dumps(error_response, indent=2)
        }

# === ORDER CREATION NODE ===
def order_creation_node(state: AgentState):
    """Handle order creation requests"""
    try:
        # Extract order details from user message using LLM
        prompt = f"""
        Extract order information from the user message and return a JSON object:
        {{
            "variant_id": "extracted variant ID if mentioned",
            "email": "extracted email if mentioned", 
            "quantity": 1,
            "needs_more_info": true/false
        }}

        User Message: "{state["user_message"]}"

        If variant_id or email is missing, set needs_more_info to true.
        Return ONLY the JSON object.
        """
        
        result = call_gemini_llm(prompt)
        
        # Parse extraction result
        try:
            cleaned = re.sub(r"```[a-zA-Z]*", "", result).strip("` \n")
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            
            if json_match:
                order_info = json.loads(json_match.group())
                
                if order_info.get("needs_more_info", True):
                    error_response = {"error": "Missing information. Please provide variant ID and email address to create an order."}
                    return {
                        "order_result": error_response,
                        "final_response": json.dumps(error_response, indent=2)
                    }
                
                # Create order payload for MCP call
                order_payload = {
                    "order": {
                        "line_items": [{
                            "variant_id": int(order_info["variant_id"]),
                            "quantity": order_info.get("quantity", 1)
                        }],
                        "customer": {
                            "email": order_info["email"]
                        },
                        "financial_status": "paid",
                        "test": True
                    }
                }
                
                # Call MCP server directly for order creation
                raw_order_result = call_mcp_server(ORDER_MCP_URL, "create_order", order_payload)
                
                # Format response using LLM
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
                
                # Extract JSON from LLM response
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
                
                # Fallback to raw data if formatting fails
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
        # Extract order ID from user message
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
                
                # Get order status using direct MCP call
                try:
                    order_id = int(order_info["order_id"])
                    raw_status_result = call_mcp_server(ORDER_MCP_URL, "get_order_status", {"order_id": order_id})
                except (ValueError, TypeError):
                    error_response = {"error": "Invalid order ID format."}
                    return {
                        "order_status": error_response,
                        "final_response": json.dumps(error_response, indent=2)
                    }
                
                # Format response using LLM
                format_prompt = f"""
                Format the order status result into the exact JSON structure below:

                Required JSON format:
                {{
                  "order_id": order_id_number,
                  "order_number": "#ORDER_NUMBER",
                  "product": "PRODUCT_NAME",
                  "quantity": quantity_number,
                  "total_paid": "AMOUNT INR",
                  "status": "STATUS",
                  "fulfillment_status": "FULFILLMENT_STATUS",
                  "order_date": "YYYY-MM-DD HH:MM:SS"
                }}

                Raw order status result: {json.dumps(raw_status_result, indent=2)}

                Extract the order ID, order number, product name, quantity, total amount, status, fulfillment status, and order date from the raw data.
                For fulfillment_status, use "Not yet shipped" if null or empty, otherwise use the actual status.
                Return ONLY the formatted JSON, no other text.
                """
                
                formatted_result = call_gemini_llm(format_prompt)
                
                # Extract JSON from LLM response
                try:
                    cleaned = re.sub(r"```[a-zA-Z]*", "", formatted_result).strip("` \n")
                    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    
                    if json_match:
                        formatted_status = json.loads(json_match.group())
                        return {
                            "order_status": formatted_status,
                            "final_response": json.dumps(formatted_status, indent=2)
                        }
                except Exception as e:
                    print(f"Order status formatting error: {e}")
                
                # Fallback to raw data if formatting fails
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

# === INFO SEARCH (RAG) NODE ===
def info_search_node(state: AgentState):
    """Handle informational queries using RAG over existing knowledge base (Pinecone)."""
    user_q = state.get("user_message", "")
    topic = "general"
    
    print(f"[DEBUG] RAG query: {user_q}")

    # Try RAG with Pinecone + Gemini
    try:
        # Get API keys with proper fallback
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX")
        
        print(f"[DEBUG] API Keys - Google: {'✓' if google_key else '✗'}, Pinecone: {'✓' if pinecone_key else '✗'}, Index: {pinecone_index}")

        if not google_key:
            raise ValueError("Google API key not found (GOOGLE_API_KEY or GEMINI_API_KEY)")
        if not pinecone_key:
            raise ValueError("Pinecone API key not found (PINECONE_API_KEY)")
        if not pinecone_index:
            raise ValueError("Pinecone index not found (PINECONE_INDEX)")

        # Configure providers
        genai.configure(api_key=google_key)
        _pc = Pinecone(api_key=pinecone_key)
        print("[DEBUG] Pinecone client initialized")

        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large",
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[DEBUG] Embeddings initialized")
        
        vectordb = PineconeVectorStore.from_existing_index(
            index_name=pinecone_index,
            embedding=embeddings
        )
        print("[DEBUG] Vector store connected")
        
        # Test if index has data
        test_results = vectordb.similarity_search("test", k=1)
        if not test_results:
            raise ValueError("Pinecone index appears to be empty")
        print(f"[DEBUG] Index has {len(test_results)} test results")

        # Setup retrieval chain
        retriever = vectordb.as_retriever(search_type="similarity", k=8)
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_key),
            retriever=retriever
        )
        print("[DEBUG] QA chain initialized")

        # Execute query
        prompt_q = (
            "Answer strictly based on the retrieved documents. If nothing relevant is retrieved, say so. Then follow ALL formatting rules below.\n\n"
            "OUTPUT GUIDELINES:\n"
            "1) Always rephrase and organize retrieved content into a polished, conversational response.\n"
            "2) Use clear sections, bullet points, and bolded highlights where applicable.\n"
            "3) Tone must reflect CNXStore brand – warm, helpful, and modern.\n\n"
            "WHEN ANSWERING QUESTIONS LIKE:\n"
            "- 'Tell me about CNXStore'\n"
            "- 'What membership benefits do you offer?'\n"
            "- 'What makes CNXStore different?'\n"
            "- 'What are the store policies?'\n\n"
            "STRUCTURE THE RESPONSE LIKE THIS:\n"
            "- Start with a friendly welcome or heading (e.g., ### About cnxStore).\n"
            "- Include key subheadings such as:\n"
            "  - **Who We Are**\n"
            "  - **Product Range**\n"
            "  - **Why Choose Us**\n"
            "  - **Member Benefits**\n"
            "  - **Sustainability & Community**\n"
            "  - **How to Stay Updated**\n"
            "- Use bold text, sparse emojis, and clean markdown formatting for readability.\n"
            "- Make it scannable – avoid one long paragraph.\n\n"
            "ALWAYS INCLUDE:\n"
            "- Factual accuracy based on retrieved content.\n"
            "- A closing line inviting the user to explore or ask more.\n"
            "- Friendly, professional tone.\n\n"
            "NEVER INCLUDE:\n"
            "- Raw retrieval output.\n"
            "- Technical details (like source paths or IDs).\n"
            "- Overly robotic or generic phrases.\n\n"
            f"User question: {user_q}"
        )
        
        # Important: use the raw user question for retrieval relevance
        result = qa_chain({"query": user_q})
        raw_answer = result.get("result", "")

        # Second pass: format to CNXStore brand style (no sources shown)
        formatting_instructions = (
            "Rephrase and organize the content into a polished, conversational CNXStore-branded response.\n"
            "- Use headings, bullet points, and bold highlights.\n"
            "- Keep it warm, helpful, and modern.\n"
            "- Do not include citations, technical details, or raw snippets.\n"
        )

        # Offers-aware formatting
        offer_keywords = ["offer", "offers", "discount", "sale", "flash", "deal", "coupon", "membership", "loyalty"]
        is_offer_query = any(k in user_q.lower() for k in offer_keywords)

        if is_offer_query:
            format_prompt = f"""
            You are a CNX Store copywriter. Based strictly on the following content, produce a marketing-quality answer.

            {formatting_instructions}

            FORMAT THE ANSWER LIKE THIS:
            - Title: "Current Offers at CNX Store 🌟"
            - A warm one-line welcome.
            - Numbered sections for each distinct offer found (name + 1–2 bullets with percentages, codes, timing, or categories when available). Do not invent details.
            - Optional section: "Exclusive Member Benefits" if such info appears in the content.
            - Close with a friendly invitation to ask more.

            CONTENT TO USE:
            {raw_answer}
            """
        else:
            format_prompt = f"""
            You are a CNX Store copywriter. Based strictly on the following content, produce a structured, skimmable answer.

            {formatting_instructions}

            Preferred structure when applicable:
            - Start with a friendly heading (e.g., ### About CNX Store)
            - Include subheadings such as **Who We Are**, **Product Range**, **Why Choose Us**, **Member Benefits**, **Sustainability & Community**, **How to Stay Updated**.
            - Close with a helpful invitation to explore or ask more.

            CONTENT TO USE:
            {raw_answer}
            """

        formatted = call_gemini_llm(format_prompt) or raw_answer
        answer = formatted.strip()
        
        # Extract sources
        sources = []
        for doc in result.get("source_documents", []) or []:
            if hasattr(doc, 'metadata') and doc.metadata:
                src = doc.metadata.get("source")
                if src:
                    sources.append(src)
        
        print(f"[DEBUG] RAG successful - Answer length: {len(answer)}, Sources: {len(sources)}")

        payload = {
            "info": {
                "topic": topic,
                "answer": answer.strip()
            },
            "sources": list(dict.fromkeys(sources)) if sources else []
        }
        return {
            "info_result": payload,
            "final_response": json.dumps(payload, indent=2)
        }

    except Exception as e:
        print(f"[DEBUG] RAG failed: {str(e)}")
        
        # Enhanced fallback with better topic detection
        message_lower = user_q.lower()
        
        if any(word in message_lower for word in ["return", "refund", "exchange", "policy"]):
            topic = "return_policy"
            answer = "Our standard return/exchange window is 7–14 days for unused items with original tags and receipt. Certain items may be non-returnable. For exact policy details, please refer to our Return Policy page or contact support."
        elif any(word in message_lower for word in ["contact", "phone", "email", "support", "address", "reach"]):
            topic = "contact_details"
            answer = "You can reach support via email at support@example.com or phone at +1-000-000-0000. Business hours: Mon–Fri, 9am–6pm IST."
        elif any(word in message_lower for word in ["offer", "discount", "sale", "promotion", "deal", "coupon"]):
            topic = "current_offers"
            answer = "Current promotions vary by season. Please check the Offers page or sign up for our newsletter/app notifications for the latest discounts and coupon codes."
        else:
            answer = "I can help with return policy, contact details, or current offers. Please specify your question."
            
        payload = {
            "info": {
                "topic": topic,
                "answer": answer,
                "note": f"RAG not available; showing fallback information. Error: {str(e)}"
            }
        }
        return {
            "info_result": payload,
            "final_response": json.dumps(payload, indent=2)
        }

# === ROUTING FUNCTION ===
def route_by_intent(state: AgentState):
    """Route to appropriate node based on user intent"""
    intent = state.get("intent", "product_search")
    
    if intent == "order_creation":
        return "order_creation"
    elif intent == "order_status":
        return "order_status"
    elif intent == "info_search":
        return "info_search"
    else:
        return "product_search"

# === GRAPH CONSTRUCTION ===
def create_agent_workflow():
    """Create and return the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_intent", analyze_user_intent)
    workflow.add_node("product_search", product_search_node)
    workflow.add_node("order_creation", order_creation_node)
    workflow.add_node("order_status", order_status_node)
    workflow.add_node("info_search", info_search_node)
    
    # Set entry point
    workflow.set_entry_point("analyze_intent")
    
    # Add conditional routing from intent analysis
    workflow.add_conditional_edges(
        "analyze_intent",
        route_by_intent,
        {
            "product_search": "product_search",
            "order_creation": "order_creation", 
            "order_status": "order_status",
            "info_search": "info_search"
        }
    )
    
    # All nodes end the workflow
    workflow.add_edge("product_search", END)
    workflow.add_edge("order_creation", END)
    workflow.add_edge("order_status", END)
    workflow.add_edge("info_search", END)
    
    return workflow.compile()

# === MAIN EXECUTION FUNCTION ===
def process_user_message(user_message: str) -> dict:
    """Process a user message through the LangGraph workflow"""
    graph = create_agent_workflow()
    
    # Execute the workflow
    result = graph.invoke({"user_message": user_message})
    
    # Augment final_response JSON with user_intent when possible
    intent = result.get("intent")
    final_response = result.get("final_response") or ""
    augmented_final = final_response
    try:
        parsed = json.loads(final_response) if isinstance(final_response, str) else final_response
        if isinstance(parsed, dict):
            # Do not overwrite if already present
            parsed.setdefault("user_intent", intent)
            augmented_final = json.dumps(parsed)
    except Exception:
        # Leave as-is if not valid JSON
        pass
    
    return {
        "user_message": result.get("user_message"),
        "intent": intent,
        "intent_details": result.get("intent_details"),
        "final_response": augmented_final,
        "full_state": result,
        "user_intent": intent,
    }

# === FASTAPI INTEGRATION ===
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

app = FastAPI(title="LangGraph Agent API", version="1.0.0")

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
    """Process user messages through the LangGraph workflow"""
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
        
        # Process through LangGraph workflow
        result = process_user_message(last_message)
        
        # Get the formatted response from the workflow
        chat_message = result.get("final_response", "")
        
        return AgentResponse(
            chat_message=chat_message,
            intent=result.get("intent"),
            intent_details=result.get("intent_details"),
            inner_messages=[result.get("full_state", {})],
            user_intent=result.get("user_intent") or result.get("intent")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LangGraph Agent API"}

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Run as API server
        uvicorn.run(app, host="0.0.0.0", port=8002)
    else:
        # Run test examples
        test_messages = [
            "Show me floral shirts under 2000",
            "What is your return policy?",
            "How can I contact support?",
            "Any current offers or discounts?",
        ]
        # Example messages for testing
        # "I want to buy product with variant ID 42910880890963, my email is test@example.com",
        # "What's the status of order 5904242344019?"
        
        for message in test_messages:
            print(f"\n{'='*50}")
            print(f"User Message: {message}")
            print(f"{'='*50}")
            
            result = process_user_message(message)
            print(f"Intent: {result['intent']}")
            print(f"Response: {result['final_response']}")
