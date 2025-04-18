from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import torch
import warnings
import re
import os
import time
import csv
import random
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Load models for query decomposition and response generation
print("Loading models and tokenizers...")
# Model for query decomposition
decomp_model_name = "mediocredev/open-llama-3b-v2-instruct"  # You can replace with fine-tuned Llama/Qwen
decomp_tokenizer = AutoTokenizer.from_pretrained(decomp_model_name, cache_dir='./cache')
decomp_model = AutoModelForCausalLM.from_pretrained(decomp_model_name, cache_dir='./cache', torch_dtype=torch.bfloat16).to('cuda')

# Model for response generation
gen_model_name = "mediocredev/open-llama-3b-v2-instruct"  # You can replace with fine-tuned Llama/Qwen
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, cache_dir='./cache')
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, cache_dir='./cache', torch_dtype=torch.bfloat16).to('cuda')

# Set token limits for context
MAX_TOKENS = 1000
TOKEN_MARGIN = 300

# Create pipelines for decomposition and generation
decomp_pipeline = transformers.pipeline(
    'text-generation',
    model=decomp_model,
    tokenizer=decomp_tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

gen_pipeline = transformers.pipeline(
    'text-generation',
    model=gen_model,
    tokenizer=gen_tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

def decompose_query(user_input):
    """
    Analyzes complex user queries and breaks them down into optimized search terms
    """
    # Simplified prompt to get better results
    System_prompt = "Your task is to convert this question into 3-5 keywords for a web search engine."
    prompt = f'{System_prompt}\n\nQuestion: {user_input}\n\nSearch keywords:'
    
    tokens = len(decomp_tokenizer.encode(prompt))
    print(f"Decomposition prompt length: {tokens} tokens")
    
    if tokens > MAX_TOKENS:
        print(f"Warning: Input too long ({tokens} tokens), truncating...")
        truncated_input = decomp_tokenizer.decode(decomp_tokenizer.encode(user_input)[:MAX_TOKENS-150])
        prompt = f'{System_prompt}\n\nQuestion: {truncated_input}\n\nSearch keywords:'
    
    try:
        response = decomp_pipeline(
            prompt,
            max_new_tokens=100,
            repetition_penalty=1.05,
        )
        response = response[0]['generated_text']
        # Extract the generated search terms
        if 'Search keywords:' in response:
            result = response.split('Search keywords:')[1].split('</s>')[0].strip()
        else:
            # Fallback extraction if the model doesn't follow the exact format
            result = response.split(user_input)[1].split('</s>')[0].strip()
            
        # If result is empty or too short, create a basic query from the question
        if not result or len(result) < 5:
            # Extract key nouns from the question
            words = user_input.split()
            # Remove common stop words and keep nouns/important words
            important_words = [w for w in words if len(w) > 3 and w.lower() not in 
                              ['what', 'which', 'where', 'when', 'how', 'why', 'who', 'was', 'were', 'did', 'does']]
            if important_words:
                result = ' '.join(important_words[:5])  # Take up to 5 important words
            else:
                result = user_input  # Fallback to original question
                
        # Add date filter for future-dated questions
        if '2025' in user_input and '2025' not in result:
            result += ' 2025'
            
        print(f"Decomposed query: {result}")
        return result
    except Exception as e:
        print(f"Error decomposing query: {e}")
        # Create a simple fallback query by extracting key terms
        words = user_input.split()
        important_words = [w for w in words if len(w) > 3 and w.lower() not in 
                          ['what', 'which', 'where', 'when', 'how', 'why', 'who', 'was', 'were', 'did', 'does']]
        if important_words:
            fallback_query = ' '.join(important_words[:5])
        else:
            fallback_query = user_input
            
        if '2025' in user_input and '2025' not in fallback_query:
            fallback_query += ' 2025'
            
        print(f"Using fallback query: {fallback_query}")
        return fallback_query

def search_web(query):
    search_results = []
    # Add future date filter
    if '2025' not in query:
        query += " 2025"
    
    print(f"---SEARCH DEBUG--- Final search query: \"{query}\"")
    print(f"---SEARCH DEBUG--- Starting search with timeout=15s")
    
    try:
        # Add delay and randomization to avoid rate limits
        delay = random.uniform(2.0, 5.0)
        print(f"---SEARCH DEBUG--- Waiting {delay:.2f}s before search to avoid rate limits")
        time.sleep(delay)
        
        # Add timeout for search to prevent hanging
        print(f"---SEARCH DEBUG--- Creating search generator")
        search_generator = search(query, num_results=5, timeout=15)
        
        # Use a timeout to prevent hanging
        import signal
        class TimeoutException(Exception): pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException("Search timed out")
        
        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(20)  # 20 second timeout
        print(f"---SEARCH DEBUG--- Set global timeout alarm for 20s")
        
        try:
            print(f"---SEARCH DEBUG--- Starting to iterate through search results")
            result_count = 0
            for url in search_generator:
                if url:
                    print(f"---SEARCH DEBUG--- Found URL: {url}")
                    search_results.append(url)
                    result_count += 1
                else:
                    print(f"---SEARCH DEBUG--- Received empty URL result")
                if len(search_results) == 3:
                    print(f"---SEARCH DEBUG--- Reached target of 3 URLs, stopping search")
                    break
            
            # Cancel the timeout
            signal.alarm(0)
            print(f"---SEARCH DEBUG--- Successfully completed search with {result_count} results")
        except TimeoutException:
            print("---SEARCH DEBUG--- Search operation timed out after 20s, proceeding with results found so far")
            print(f"---SEARCH DEBUG--- Collected {len(search_results)} results before timeout")
        except requests.exceptions.HTTPError as e:
            error_code = str(e).split()[0] if str(e).split() else "unknown"
            print(f"---SEARCH DEBUG--- HTTP error during search: {error_code} - {str(e)}")
            if '429' in str(e):
                print("---SEARCH DEBUG--- Rate limit (429) detected, will wait 30s before next request")
                time.sleep(30)  # Wait longer on rate limit
        except Exception as e:
            print(f"---SEARCH DEBUG--- Unexpected error during search iteration: {type(e).__name__}: {str(e)}")
        
    except Exception as e:
        print(f"---SEARCH DEBUG--- Fatal error in web search: {type(e).__name__}: {str(e)}")
    
    print(f"---SEARCH DEBUG--- Search completed with {len(search_results)} results: {search_results}")
    return search_results

def evaluate_search_results(urls, user_query):
    """
    Evaluates retrieved results based on relevance, timeliness, and source authority
    Returns a list of scored urls (url, score) sorted by score
    """
    evaluated_results = []
    
    for url in urls:
        # Initial score is 1.0
        score = 1.0
        
        # Simple heuristics for evaluation:
        # Domain authority (simple approach)
        if '.gov' in url or '.edu' in url:
            score += 0.3
        if 'news' in url or 'wikipedia' in url:
            score += 0.2
            
        # Relevance based on URL keywords match
        query_keywords = user_query.lower().split()
        url_lower = url.lower()
        keyword_matches = sum(1 for keyword in query_keywords if keyword in url_lower)
        score += 0.1 * keyword_matches
        
        # Add to results
        evaluated_results.append((url, score))
    
    # Sort by score, descending
    evaluated_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return sorted URLs
    return [url for url, _ in evaluated_results]

def truncate_content(content, max_tokens, tokenizer):
    """Truncate content to fit within token limit"""
    tokens = tokenizer.encode(content)
    if len(tokens) <= max_tokens:
        return content
    
    # Truncate and add indication that content was cut
    truncated_tokens = tokens[:max_tokens-10]  # Leave room for ellipsis
    truncated_content = tokenizer.decode(truncated_tokens) + "..."
    print(f"Truncated content from {len(tokens)} to {len(truncated_tokens)} tokens")
    return truncated_content

def extract_main_content(url):
    try:
        # Add timeout to requests to prevent hanging
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, verify=False, timeout=15, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find the main content
        article = soup.find('article')
        if article:
            # If there's an article element, use that
            main_content = article.get_text()
        else:
            # Otherwise use paragraphs
            paragraphs = soup.find_all('p')
            main_content = ' '.join([para.get_text() for para in paragraphs])
        
        # Clean up the content
        main_content = re.sub(r'\s+', ' ', main_content).strip()
        
        # Limit content size to avoid token overflow
        content_tokens = len(gen_tokenizer.encode(main_content))
        max_allowed = MAX_TOKENS // 4  # Even smaller chunk per source to be safe
        
        if content_tokens > max_allowed:
            print(f"Content too large ({content_tokens} tokens), truncating to {max_allowed}...")
            return truncate_content(main_content, max_allowed, gen_tokenizer)
        
        return main_content
    except requests.exceptions.Timeout:
        print(f"Request to {url} timed out")
        return ""
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""

def generate_response_with_rag(user_input, main_contents):
    """
    Synthesizes the original query with retrieved search results using a generation model
    """
    if len(main_contents) > 2:
        print("Too many content sources, limiting to top 2")
        main_contents = main_contents[:2]
    
    # Estimate token usage and limit content if needed
    total_content = "\n\n".join(main_contents)
    content_tokens = len(gen_tokenizer.encode(total_content))
    
    max_content_tokens = MAX_TOKENS - TOKEN_MARGIN
    if content_tokens > max_content_tokens:
        print(f"Warning: Content too large ({content_tokens} tokens), truncating to {max_content_tokens}...")
        # Reduce each content proportionally
        ratio = max_content_tokens / content_tokens
        
        truncated_contents = []
        for content in main_contents:
            content_tokens = len(gen_tokenizer.encode(content))
            target_tokens = max(100, int(content_tokens * ratio))
            truncated_contents.append(truncate_content(content, target_tokens, gen_tokenizer))
        
        main_contents = truncated_contents
        total_content = "\n\n".join(main_contents)
    
    # Final check to ensure we're under the limit
    if len(gen_tokenizer.encode(total_content)) > max_content_tokens:
        print("Content still too large after truncation, using more aggressive approach")
        total_content = truncate_content(total_content, max_content_tokens, gen_tokenizer)
    
    # Simplified prompt for better response extraction
    System_prompt = "Answer this question using the provided information:"
    
    prompt = f"{System_prompt}\n\nInformation:\n{total_content}\n\nQuestion: {user_input}\n\nAnswer:"
    tokens = len(gen_tokenizer.encode(prompt))
    print(f"RAG prompt length: {tokens} tokens")
    
    # Last-resort truncation if still too long
    if tokens > MAX_TOKENS:
        available_tokens = MAX_TOKENS - len(gen_tokenizer.encode(f'{System_prompt}\n\nQuestion: {user_input}\n\nAnswer:')) - 100
        if available_tokens > 200:  # Only proceed if we have enough space for useful content
            print(f"Final truncation to {available_tokens} tokens for content")
            total_content = truncate_content(total_content, available_tokens, gen_tokenizer)
            prompt = f"{System_prompt}\n\nInformation:\n{total_content}\n\nQuestion: {user_input}\n\nAnswer:"
        else:
            # Fall back to no RAG if we can't fit content
            print("Not enough space for RAG content, falling back to direct question answering")
            return generate_response_no_rag(user_input)
    
    try:
        response = gen_pipeline(
            prompt,
            max_new_tokens=200,
            repetition_penalty=1.05,
        )
        generated_text = response[0]['generated_text']
        
        # Try several extraction patterns
        if "Answer:" in generated_text:
            result = generated_text.split("Answer:")[1].strip()
            if "</s>" in result:
                result = result.split("</s>")[0].strip()
        elif user_input in generated_text:
            parts = generated_text.split(user_input)
            if len(parts) > 1:
                result = parts[1].strip()
                if "</s>" in result:
                    result = result.split("</s>")[0].strip()
        else:
            # Fallback to getting the last part of the text
            result = generated_text.replace(prompt, "").strip()
            if "</s>" in result:
                result = result.split("</s>")[0].strip()
                
        # Further cleanup
        result = result.strip()
        if not result:
            result = "Unable to generate a clear answer from content."
            
        return result
    except Exception as e:
        print(f"Error generating response with RAG: {e}")
        return "Error generating response with RAG."

def generate_response_no_rag(user_input):
    """
    Generates a response without RAG using just the model's knowledge
    """
    # Simplified prompt for better response extraction
    System_prompt = "Answer this question concisely:"
    prompt = f"{System_prompt}\n\nQuestion: {user_input}\n\nAnswer:"
    
    tokens = len(gen_tokenizer.encode(prompt))
    print(f"No-RAG prompt length: {tokens} tokens")
    
    if tokens > MAX_TOKENS:
        print(f"Warning: Input too long ({tokens} tokens), truncating...")
        truncated_input = gen_tokenizer.decode(gen_tokenizer.encode(user_input)[:MAX_TOKENS-150])
        prompt = f"{System_prompt}\n\nQuestion: {truncated_input}\n\nAnswer:"
    
    try:
        response = gen_pipeline(
            prompt,
            max_new_tokens=200,
            repetition_penalty=1.05,
        )
        generated_text = response[0]['generated_text']
        
        # Try several extraction patterns
        if "Answer:" in generated_text:
            result = generated_text.split("Answer:")[1].strip()
            if "</s>" in result:
                result = result.split("</s>")[0].strip()
        elif user_input in generated_text:
            parts = generated_text.split(user_input)
            if len(parts) > 1:
                result = parts[1].strip()
                if "</s>" in result:
                    result = result.split("</s>")[0].strip()
        else:
            # Fallback to getting the last part of the text
            result = generated_text.replace(prompt, "").strip()
            if "</s>" in result:
                result = result.split("</s>")[0].strip()
                
        # Further cleanup
        result = result.strip()
        if not result:
            result = "Unable to generate a response."
            
        return result
    except Exception as e:
        print(f"Error generating response without RAG: {e}")
        return "Error generating response without RAG."

def read_test_dataset(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract questions using regex
    questions = []
    matches = re.findall(r'\*\*Question \d+: (.*?)\*\*\n Correct Answer: (.*?)\n Potential Incorrect Answer: (.*?)\n', content, re.DOTALL)
    
    for i, match in enumerate(matches):
        question_text = match[0].strip()
        correct_answer = match[1].strip()
        incorrect_answer = match[2].strip()
        questions.append({
            "id": f"Question {i+1}",
            "question": question_text,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer
        })
    
    return questions

def process_questions(questions, output_file):
    """Process all questions and save results to a CSV file"""
    # Create CSV file and write header
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Question ID', 'Question', 'Correct Answer', 'No-RAG Answer', 'RAG Answer', 'Search Query', 'References']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each question
        for question_data in tqdm(questions):
            try:
                question_id = question_data["id"]
                user_input = question_data["question"]
                
                print(f"\nProcessing {question_id}: {user_input}")
                
                # First, decompose the query
                print(f"---QUERY DEBUG--- Decomposing query for: {user_input}")
                search_query = decompose_query(user_input)
                print(f"---QUERY DEBUG--- Final search terms: \"{search_query}\"")
                
                # Generate response without RAG
                no_rag_answer = generate_response_no_rag(user_input)
                print(f"No-RAG answer: {no_rag_answer}")
                
                # Search the web with exponential backoff for rate limits
                max_retries = 3
                print(f"---SEARCH DEBUG--- Starting search process with max_retries={max_retries}")
                
                for retry in range(max_retries):
                    print(f"---SEARCH DEBUG--- Search attempt {retry+1}/{max_retries}")
                    search_results = search_web(search_query)
                    
                    if search_results:
                        print(f"---SEARCH DEBUG--- Found {len(search_results)} results, proceeding")
                        break
                    elif retry == max_retries - 1:
                        print(f"---SEARCH DEBUG--- All {max_retries} attempts failed, giving up")
                        break
                    else:
                        wait_time = 10 * (2 ** retry)  # Exponential backoff: 10, 20, 40 seconds
                        print(f"---SEARCH DEBUG--- No search results on attempt {retry+1}, retrying after {wait_time} seconds...")
                        time.sleep(wait_time)
                
                # Evaluate search results
                print(f"---SEARCH DEBUG--- Evaluating {len(search_results)} search results")
                evaluated_results = evaluate_search_results(search_results, user_input)
                print(f"---SEARCH DEBUG--- After evaluation: {len(evaluated_results)} results")
                
                # Extract content
                main_contents = []
                print(f"---CONTENT DEBUG--- Extracting content from {len(evaluated_results)} URLs")
                
                for i, url in enumerate(evaluated_results):
                    print(f"---CONTENT DEBUG--- Extracting from URL {i+1}/{len(evaluated_results)}: {url}")
                    main_content = extract_main_content(url)
                    content_size = len(main_content) if main_content else 0
                    if main_content:
                        print(f"---CONTENT DEBUG--- Successfully extracted {content_size} chars from URL {i+1}")
                        main_contents.append(main_content)
                    else:
                        print(f"---CONTENT DEBUG--- Failed to extract content from URL {i+1}")
                
                # Generate RAG response if we have content
                if main_contents:
                    print(f"---RAG DEBUG--- Generating RAG response with {len(main_contents)} content sources")
                    rag_answer = generate_response_with_rag(user_input, main_contents)
                    print(f"RAG answer: {rag_answer}")
                else:
                    rag_answer = "No web content found for RAG"
                    print(f"---RAG DEBUG--- No content available for RAG generation")
                    print("No web content found for RAG")
                
                # Write results to CSV
                print(f"---OUTPUT DEBUG--- Writing results to CSV for question {question_id}")
                writer.writerow({
                    'Question ID': question_id,
                    'Question': user_input,
                    'Correct Answer': question_data['correct_answer'],
                    'No-RAG Answer': no_rag_answer,
                    'RAG Answer': rag_answer,
                    'Search Query': search_query,
                    'References': ', '.join(search_results)
                })
                
                # Flush to save progress
                csvfile.flush()
                print(f"---OUTPUT DEBUG--- Results saved to CSV")
                
                # Add random delay between questions to avoid rate limiting
                delay = random.uniform(5.0, 12.0)  # Increased delay to reduce rate limiting
                print(f"---RATE LIMIT--- Waiting {delay:.1f} seconds before next question...")
                time.sleep(delay)
                
            except Exception as e:
                print(f"---ERROR--- Error processing question {question_data['id']}: {type(e).__name__}: {str(e)}")
                
                # Write error to CSV
                writer.writerow({
                    'Question ID': question_data['id'],
                    'Question': question_data['question'],
                    'Correct Answer': question_data['correct_answer'],
                    'No-RAG Answer': f"Error: {str(e)}",
                    'RAG Answer': f"Error: {str(e)}",
                    'Search Query': "",
                    'References': ""
                })
                csvfile.flush()
                print(f"---ERROR--- Error details written to CSV")

def main():
    print("RAG system starting...")
    
    # Read the test dataset
    test_data_path = "Test dataset.md"
    output_file = "rag_evaluation_results.csv"
    
    # Read questions
    questions = read_test_dataset(test_data_path)
    print(f"Loaded {len(questions)} questions from dataset")
    
    # Process all questions
    process_questions(questions, output_file)
    
    print(f"\nRAG evaluation completed! Results saved to {output_file}")

if __name__ == "__main__":
    main()