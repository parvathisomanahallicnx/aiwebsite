import streamlit as st
import requests
import json
import re

st.set_page_config(page_title="🤖 LangGraph Shopify Agent", layout="wide")
st.title("🤖 LangGraph Shopify Agent Assistant")
st.markdown("*Powered by LangGraph workflow with intelligent intent detection*")

API_URL = "http://localhost:8002/agent-assistant/"

# Session state for conversation and flow
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "products" not in st.session_state:
    st.session_state["products"] = []
if "selected_product" not in st.session_state:
    st.session_state["selected_product"] = None
if "awaiting_email" not in st.session_state:
    st.session_state["awaiting_email"] = False
if "order_result" not in st.session_state:
    st.session_state["order_result"] = None
if "filters" not in st.session_state:
    st.session_state["filters"] = []
if "instructions" not in st.session_state:
    st.session_state["instructions"] = ""
if "selected_filter" not in st.session_state:
    st.session_state["selected_filter"] = None
if "last_user_query" not in st.session_state:
    st.session_state["last_user_query"] = ""

# --- Intent Detection Info ---
st.sidebar.markdown("## 🎯 Supported Intents")
st.sidebar.markdown("""
**🔍 Product Search**
- "Show me floral shirts under 2000"
- "Find blue dresses"
- "I'm looking for electronics"

**🛒 Order Creation**  
- "I want to buy variant ID 123456, email: user@example.com"
- "Create order for product X"

**📦 Order Status**
- "What's the status of order 5904242344019?"
- "Track my order"
""")

# --- Chatbot input ---
user_query = st.text_input("Ask the assistant (e.g. 'Show me floral shirts under 2000')", key="user_query")
if st.button("Send Query"):
    if user_query:
        st.session_state["messages"].append({"content": user_query, "source": "user"})
        st.session_state["last_user_query"] = user_query
        st.session_state["selected_filter"] = None
        response = requests.post(
            API_URL,
            json={"messages": st.session_state["messages"]}
        )
        data = response.json()
        chat_message = data.get("chat_message", "No response")
        st.session_state["messages"].append({"content": chat_message, "source": "assistant"})
        
        # Show intent detection result
        if data.get("intent"):
            intent_emoji = {"product_search": "🔍", "order_creation": "🛒", "order_status": "📦"}
            st.info(f"{intent_emoji.get(data['intent'], '🤖')} Detected Intent: **{data['intent'].replace('_', ' ').title()}**")
        
        # Try to extract products, filters, and instructions from the response (JSON or string)
        products = []
        filters = []
        instructions = ""
        try:
            content = data["chat_message"]
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        products = parsed.get("products", [])
                        filters = parsed.get("available_filters", [])
                        instructions = parsed.get("instructions", "")
                except Exception:
                    start = content.find('{"products":')
                    if start != -1:
                        end = content.find('}}', start)
                        if end != -1:
                            json_str = content[start:end+2]
                            try:
                                parsed = json.loads(json_str)
                                if "products" in parsed:
                                    products = parsed["products"]
                                if "available_filters" in parsed:
                                    filters = parsed["available_filters"]
                                if "instructions" in parsed:
                                    instructions = parsed["instructions"]
                            except Exception:
                                pass
            elif isinstance(content, dict):
                products = content.get("products", [])
                filters = content.get("available_filters", [])
                instructions = content.get("instructions", "")
        except Exception:
            products = []
            filters = []
            instructions = ""
        for msg in data.get("inner_messages", []):
            if isinstance(msg, dict):
                if "products" in msg:
                    products = msg["products"]
                if "available_filters" in msg:
                    filters = msg["available_filters"]
                if "instructions" in msg:
                    instructions = msg["instructions"]
        st.session_state["products"] = products
        st.session_state["filters"] = filters
        st.session_state["instructions"] = instructions
        st.session_state["selected_product"] = None
        st.session_state["awaiting_email"] = False
        st.session_state["order_result"] = None

        # --- Automatic price filter mapping ---
        # If no products, and a price filter is available, and the query contains 'under X', 'below X', or 'less than X', auto-apply the price filter
        if not products and filters:
            price_limit = None
            m = re.search(r'(?:under|below|less than)\s*(\d+)', user_query, re.IGNORECASE)
            if m:
                try:
                    price_limit = float(m.group(1))
                except Exception:
                    price_limit = None
            if price_limit is not None:
                for f in filters:
                    if f["label"].lower() == "price":
                        for opt in f["values"].get("input_options", []):
                            price = opt["input"].get("price", {})
                            min_p = price.get("min", None)
                            max_p = price.get("max", None)
                            # If the filter's max is >= price_limit, use it
                            if min_p is not None and max_p is not None and float(max_p) >= price_limit:
                                filter_query = f"{user_query} price between 0 and {price_limit}"
                                st.session_state["messages"].append({"content": filter_query, "source": "user"})
                                st.session_state["selected_filter"] = f"Auto-applied: Price 0 - {price_limit}"
                                response = requests.post(
                                    API_URL,
                                    json={"messages": st.session_state["messages"]}
                                )
                                data = response.json()
                                st.session_state["messages"].append({"content": data["chat_message"], "source": "assistant"})
                                # Parse products, filters, instructions again
                                products = []
                                filters = []
                                instructions = ""
                                try:
                                    content = data["chat_message"]
                                    if isinstance(content, str):
                                        try:
                                            parsed = json.loads(content)
                                            if isinstance(parsed, dict):
                                                products = parsed.get("products", [])
                                                filters = parsed.get("available_filters", [])
                                                instructions = parsed.get("instructions", "")
                                        except Exception:
                                            start = content.find('{"products":')
                                            if start != -1:
                                                end = content.find('}}', start)
                                                if end != -1:
                                                    json_str = content[start:end+2]
                                                    try:
                                                        parsed = json.loads(json_str)
                                                        if "products" in parsed:
                                                            products = parsed["products"]
                                                        if "available_filters" in parsed:
                                                            filters = parsed["available_filters"]
                                                        if "instructions" in parsed:
                                                            instructions = parsed["instructions"]
                                                    except Exception:
                                                        pass
                                    elif isinstance(content, dict):
                                        products = content.get("products", [])
                                        filters = content.get("available_filters", [])
                                        instructions = content.get("instructions", "")
                                except Exception:
                                    products = []
                                    filters = []
                                    instructions = ""
                                for msg in data.get("inner_messages", []):
                                    if isinstance(msg, dict):
                                        if "products" in msg:
                                            products = msg["products"]
                                        if "available_filters" in msg:
                                            filters = msg["available_filters"]
                                        if "instructions" in msg:
                                            instructions = msg["instructions"]
                                st.session_state["products"] = products
                                st.session_state["filters"] = filters
                                st.session_state["instructions"] = instructions
                                st.session_state["selected_product"] = None
                                st.session_state["awaiting_email"] = False
                                st.session_state["order_result"] = None
                                st.info(f"Auto-applied price filter: Showing products with price between 0 and {price_limit}")
                                st.rerun()
                        break
        st.rerun()

# --- Filter button logic ---
def filter_to_str(f, opt):
    if f["label"].lower() == "price":
        price = opt["input"].get("price", {})
        min_p = price.get("min", "?")
        max_p = price.get("max", "?")
        return f"Price: {min_p} - {max_p}"
    elif f["label"].lower() == "availability":
        avail = opt["input"].get("available", None)
        return "In stock" if avail else "Out of stock"
    else:
        return opt["label"]

if st.session_state["filters"]:
    st.markdown("**Available Filters:**")
    for f in st.session_state["filters"]:
        for opt in f["values"].get("input_options", []):
            label = filter_to_str(f, opt)
            if st.button(label, key=f"filter_{f['label']}_{label}"):
                # Compose a new query using the last user query and the selected filter
                filter_query = st.session_state["last_user_query"]
                if f["label"].lower() == "price":
                    price = opt["input"].get("price", {})
                    min_p = price.get("min", None)
                    max_p = price.get("max", None)
                    if min_p is not None and max_p is not None:
                        filter_query = f"{st.session_state['last_user_query']} price between {min_p} and {max_p}"
                elif f["label"].lower() == "availability":
                    avail = opt["input"].get("available", None)
                    if avail is not None:
                        filter_query = f"{st.session_state['last_user_query']} {'in stock' if avail else 'out of stock'}"
                st.session_state["messages"].append({"content": filter_query, "source": "user"})
                st.session_state["selected_filter"] = label
                response = requests.post(
                    API_URL,
                    json={"messages": st.session_state["messages"]}
                )
                data = response.json()
                st.session_state["messages"].append({"content": data["chat_message"], "source": "assistant"})
                # Parse products, filters, instructions again
                products = []
                filters = []
                instructions = ""
                try:
                    content = data["chat_message"]
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                products = parsed.get("products", [])
                                filters = parsed.get("available_filters", [])
                                instructions = parsed.get("instructions", "")
                        except Exception:
                            start = content.find('{"products":')
                            if start != -1:
                                end = content.find('}}', start)
                                if end != -1:
                                    json_str = content[start:end+2]
                                    try:
                                        parsed = json.loads(json_str)
                                        if "products" in parsed:
                                            products = parsed["products"]
                                        if "available_filters" in parsed:
                                            filters = parsed["available_filters"]
                                        if "instructions" in parsed:
                                            instructions = parsed["instructions"]
                                    except Exception:
                                        pass
                    elif isinstance(content, dict):
                        products = content.get("products", [])
                        filters = content.get("available_filters", [])
                        instructions = content.get("instructions", "")
                except Exception:
                    products = []
                    filters = []
                    instructions = ""
                for msg in data.get("inner_messages", []):
                    if isinstance(msg, dict):
                        if "products" in msg:
                            products = msg["products"]
                        if "available_filters" in msg:
                            filters = msg["available_filters"]
                        if "instructions" in msg:
                            instructions = msg["instructions"]
                st.session_state["products"] = products
                st.session_state["filters"] = filters
                st.session_state["instructions"] = instructions
                st.session_state["selected_product"] = None
                st.session_state["awaiting_email"] = False
                st.session_state["order_result"] = None
                st.rerun()

if st.session_state["selected_filter"]:
    st.info(f"Selected Filter: {st.session_state['selected_filter']}")

# --- Show conversation ---
st.subheader("Conversation")
for msg in st.session_state["messages"]:
    if msg["source"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# --- Show assistant instructions ---
if st.session_state["instructions"]:
    st.info(st.session_state["instructions"])

# --- Product display and selection ---
if st.session_state["products"]:
    st.subheader("Select a Product")
    cols = st.columns(2)
    for idx, product in enumerate(st.session_state["products"]):
        with cols[idx % 2]:
            # Handle both string and dict product data
            if isinstance(product, dict):
                title = product.get('title', f'Product {idx + 1}')
                product_url = product.get('url', '#')
                st.markdown(f"#### [{title}]({product_url})")
                
                if product.get("image_url"):
                    st.image(product["image_url"], width=200)
                elif product.get("images") and len(product["images"]) > 0:
                    image_src = product["images"][0].get("src", "")
                    if image_src:
                        st.image(image_src, width=200)
                
                # Handle price from variants
                if product.get("variants") and len(product["variants"]) > 0:
                    price = product["variants"][0].get("price", "?")
                    st.write(f"**Price:** ₹{price}")
                else:
                    price = product.get("price_range", {}).get("min", "?")
                    currency = product.get("price_range", {}).get("currency", "INR")
                    st.write(f"**Price:** {price} {currency}")
                
                st.write(product.get("description", ""))
                product_type = product.get("product_type", "")
                if product_type:
                    st.write(f"**Type:** {product_type}")
                
                tags = product.get("tags", [])
                if tags:
                    st.write(f"**Tags:** {', '.join(tags)}")
                
                if st.button(f"Select '{title}'", key=f"select_{idx}"):
                    st.session_state["selected_product"] = product
                    st.session_state["awaiting_email"] = True
                    st.session_state["order_result"] = None
                    st.session_state["messages"].append({"content": f"Create order for product: {title}", "source": "user"})
                    st.rerun()
            else:
                # Handle string product data - display as text
                st.write(f"**Product {idx + 1}:** {str(product)}")
                if st.button(f"Select Product {idx + 1}", key=f"select_{idx}"):
                    st.session_state["selected_product"] = {"title": f"Product {idx + 1}", "data": str(product)}
                    st.session_state["awaiting_email"] = True
                    st.session_state["order_result"] = None
                    st.session_state["messages"].append({"content": f"Create order for product: Product {idx + 1}", "source": "user"})
                    st.rerun()
else:
    if st.session_state["messages"] and not st.session_state["products"]:
        st.warning("No products found. Try a different query or use available filters.")

# --- Email input and order confirmation ---
if st.session_state["awaiting_email"] and st.session_state["selected_product"]:
    st.subheader("Enter your email to complete the order")
    email = st.text_input("Email", key="order_email")
    if st.button("Confirm Order"):
        if not email:
            st.warning("Please enter your email.")
        else:
            product = st.session_state["selected_product"]
            variant_id = product["variants"][0]["variant_id"] if product.get("variants") else ""
            order_msg = f"My email is {email}. Please create the order for product: {product['title']} (variant_id: {variant_id})"
            st.session_state["messages"].append({"content": order_msg, "source": "user"})
            response = requests.post(
                API_URL,
                json={"messages": st.session_state["messages"]}
            )
            data = response.json()
            st.session_state["messages"].append({"content": data["chat_message"], "source": "assistant"})
            st.session_state["order_result"] = data["chat_message"]
            st.session_state["awaiting_email"] = False
            st.rerun()

# --- Show order result ---
if st.session_state["order_result"]:
    st.success(f"Order Result: {st.session_state['order_result']}") 