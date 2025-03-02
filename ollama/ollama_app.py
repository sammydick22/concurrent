import streamlit as st
import requests
import json
import time
import threading
import asyncio

# Blocking streaming call using requests (as before)
def stream_api_call(messages, max_tokens, is_final_answer=False):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.2:3b",
        "messages": messages,
        "options": {"temperature": 0.2, "num_predict": max_tokens},
        "stream": True,
        "format": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "next_action": {"type": "string"}
            },
            "required": ["title", "content", "next_action"]
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, stream=True)
    accumulated_text = ""
    for line in response.iter_lines():
        if line:
            raw_chunk = line.decode('utf-8')
            print("Raw response chunk:", raw_chunk)  # Debug print
            try:
                data = json.loads(raw_chunk)
            except Exception as e:
                print("Decoding error:", e)
                continue
            # Extract only the token from the JSON
            token = data.get("message", {}).get("content", "")
            accumulated_text += token
            done = data.get("done", False)
            yield accumulated_text, done
            if done:
                break

# A helper function to run the blocking stream in a separate thread and update a shared dict
def stream_worker(messages, max_tokens, shared_output):
    try:
        for partial_text, done in stream_api_call(messages, max_tokens):
            shared_output["text"] = partial_text
            shared_output["done"] = done
            time.sleep(0.1)  # slight delay to avoid tight loop
    except Exception as e:
        shared_output["text"] = f"Error: {str(e)}"
        shared_output["done"] = True

# Asynchronous wrapper using asyncio.to_thread for the streaming worker
async def run_stream_worker(messages, max_tokens, shared_output):
    await asyncio.to_thread(stream_worker, messages, max_tokens, shared_output)

# Main async generator that concurrently streams two steps
async def generate_response_dual(prompt):
    # Set up initial conversation context
    messages = [
        {"role": "system", "content": (
            "You are an expert AI assistant that explains your reasoning step by step. "
            "For each step, output valid JSON with keys 'title', 'content', and 'next_action' "
            "(either 'continue' or 'final_answer'). Do not include any extra text. Only include 'final_answer' if you are 100 percent sure that the final answer is correct."
            "If the answer still needs verification, output 'continue'. If you are sure that the answer is correct, output 'final_answer'. "
        )},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step."}
    ]
    steps = []
    step_count = 1
    total_time = 0

    while True:
        # Shared dicts for current and next step outputs
        current_stream = {"text": "", "done": False}
        next_stream = {"text": "", "done": False}
        current_messages = messages.copy()

        # Launch current step stream
        task_current = asyncio.create_task(run_stream_worker(current_messages, 300, current_stream))

        # Wait until some tokens accumulate (e.g., >50 chars) to launch next step prefetch
        while len(current_stream["text"]) < 50 and not current_stream["done"]:
            await asyncio.sleep(0.1)
        # Now, prepare candidate messages with the partial current output
        candidate_messages = messages + [{"role": "assistant", "content": current_stream["text"]}]
        task_next = asyncio.create_task(run_stream_worker(candidate_messages, 300, next_stream))

        start_time = time.time()
        # Loop until current step streaming is done
        while not current_stream["done"]:
            # Yield updates: show current step and next step preview
            yield {
                "current": (f"Step {step_count} (Current)", current_stream["text"], time.time() - start_time),
                "next": (f"Step {step_count + 1} (Prefetch)", next_stream["text"], None)
            }
            await asyncio.sleep(0.1)
        end_time = time.time()
        step_time = end_time - start_time
        total_time += step_time

        # Finalize current step output
        try:
            step_data = json.loads(current_stream["text"])
        except Exception as e:
            step_data = {"title": "Error", "content": f"Parsing error: {str(e)}", "next_action": "final_answer"}
        steps.append((f"Step {step_count}: {step_data.get('title', 'No Title')}", step_data.get("content", ""), step_time))
        # Append the finalized JSON (as string) to the messages
        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        # Cancel the next step task if it’s still running (because we'll start a new one in the next iteration)
        if not next_stream["done"]:
            task_next.cancel()

        # Yield update: current step finalized; also show the latest next stream preview.
        yield {
            "current": (f"Step {step_count} (Final)", step_data.get("content", ""), step_time),
            "next": (f"Step {step_count + 1} (Prefetch)", next_stream["text"], None)
        }

        # If the step indicates final answer or maximum steps reached, break out.
        if step_data.get("next_action") == "final_answer" or step_count >= 25:
            break
        step_count += 1

    # Generate final answer similarly
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    final_stream = {"text": "", "done": False}
    final_task = asyncio.create_task(run_stream_worker(messages, 200, final_stream))
    final_start = time.time()
    while not final_stream["done"]:
        yield {
            "final": ("Final Answer (Streaming)", final_stream["text"], time.time() - final_start)
        }
        await asyncio.sleep(0.1)
    final_end = time.time()
    final_time = final_end - final_start
    total_time += final_time
    try:
        final_data = json.loads(final_stream["text"])
    except Exception as e:
        final_data = {"content": f"Error parsing final output: {str(e)}"}
    steps.append(("Final Answer", final_data.get("content", ""), final_time))
    yield {"final": ("Final Answer", final_data.get("content", ""), final_time), "total_time": total_time}

# Main function using asyncio.run
def main():
    st.set_page_config(page_title="g1 Dual Streaming Prototype", page_icon="🧠", layout="wide")
    st.title("g1: Dual Streaming Reasoning Steps with Ollama")
    st.markdown("""
    This prototype runs two streaming tasks concurrently: one for the current reasoning step and a prefetch of the next step.
    Both streams are displayed side by side.
    """)
    
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")
    if user_query:
        st.write("Generating response...")
        # Create persistent containers for updates
        col_current, col_next = st.columns(2)
        with col_current:
            current_title = st.empty()
            current_content = st.empty()
        with col_next:
            next_title = st.empty()
            next_content = st.empty()
        final_title = st.empty()
        final_content = st.empty()
        total_time_container = st.empty()
        
        async def run_generation():
            async for update in generate_response_dual(user_query):
                if "current" in update and "next" in update:
                    title_current, text_current, t_current = update["current"]
                    title_next, text_next, _ = update["next"]
                    current_title.markdown(f"### {title_current}")
                    current_content.markdown(text_current.replace("\n", "<br>"), unsafe_allow_html=True)
                    next_title.markdown(f"### {title_next}")
                    next_content.markdown(text_next.replace("\n", "<br>"), unsafe_allow_html=True)
                if "final" in update:
                    title_final, text_final, t_final = update["final"]
                    final_title.markdown(f"### {title_final}")
                    final_content.markdown(text_final.replace("\n", "<br>"), unsafe_allow_html=True)
                if "total_time" in update:
                    total_time_container.markdown(f"**Total thinking time: {update['total_time']:.2f} seconds**")
                await asyncio.sleep(0.1)
        
        asyncio.run(run_generation())

if __name__ == "__main__":
    main()
