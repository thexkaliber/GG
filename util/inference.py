import json
import re
import csv
import ollama

# CSV Helper: fetch game state by id (if needed)
def get_game_state_by_id(csv_file_path: str, entry_id: str) -> str:
    """
    Reads the CSV file and returns the formatted text for the given entry id.
    """
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("id") == entry_id:
                return row.get("formatted_text", "")

async def fetch_consequence(model: str, prompt: str):
    """
    Asynchronous function to fetch the consequence from the Ollama model.
    """
    try:
        # Use the Ollama Python client to generate a response
        response = ollama.generate(model=model, prompt=prompt, options={"num_ctx": 4096})
        
        # Extract the reasoning enclosed in <think>...</think> tags.
        reasoning_match = re.search(r"<think>(.*?)</think>", response["response"], re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
        
        # Extract all text after </think> as the consequence
        consequence_match = re.search(r"</think>\s*(.*)", response["response"], re.DOTALL)
        consequence = consequence_match.group(1).strip() if consequence_match else "No consequence provided."
        
        return reasoning, consequence
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

async def process_game_state(model: str, entry_id: str, game_state_text: str, results: list):
    """
    Processes a single game state entry and fetches the reasoning and consequence.
    """
    base_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "Write a response that appropriately completes the request.\n"
        "Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response. Keep your reasoning as brief as possible and within 5 sentences. You are only allowed to make use of the context provided to you.\n"
        "### Instruction:\n"
        "You are an expert rule engine for a game with knowledge about the game world, relationships, and rules. You will retrieve the knowledge about the game world, relationships, and rules through only the context provided to you.\n"
        "Decide the appropriate consequence of the given action with the observations on the changes of the game world from the past action, set of valid actions allowed, location, surroundings, and inventory context. Generate the consequence phrase with its description in a structured JSON format."
    )
    appended_prompt = f"{base_prompt}\n\n{game_state_text}"
    
    # Fetch the reasoning and consequence using the specified model
    reasoning, consequence = await fetch_consequence(model, appended_prompt)
    
    if reasoning is not None and consequence is not None:
        output_data = {
            "game_state_id": entry_id,
            "reasoning": reasoning,
            "consequence": consequence,
            "model_used": model
        }
        results.append(output_data)  # Add the result to the list
        print(f"Processed game state with id '{entry_id}' using model '{model}'.")
    else:
        with open("failed_entries.txt", "a", encoding='utf-8') as f:
            f.write(f"Failed to process game state with id '{entry_id}'.\n")
        print(f"Failed to process game state with id '{entry_id}'.")

async def main():
  model1 = "deepseek-r1:14b"
  model2 = "deepseek-r1:32b" #To generate inferences for thousands of prompts, we infer from two models parallely 
  csv_filepath = 'INSERT_FILE_PATH.csv'
  failed_entries = []
  results = []
  start_from = "ENTER_START_POSITION_TO_PROCESS_FROM"
  start_processing = False
  tasks = []
  batch_size = 50
  
    # Read rows from the CSV file; only process rows once the starting row is found.
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        count = 0
        for row in reader:
            desired_entry_id = row.get("id")
            # Check if we have reached the starting row.
            if not start_processing:
                if desired_entry_id == start_from:
                    start_processing = True
                else:
                    continue  # Skip rows until we find the starting row
            
            # If no game state text is present, log the failure.
            game_state_text = row.get("formatted_text")
            if not game_state_text:
                print(f"No game state found for id '{desired_entry_id}'.")
                failed_entries.append(desired_entry_id)
                continue

            # Alternate assignment: even count goes to model1, odd count to model2.
            chosen_model = model1 if count % 2 == 0 else model2

            # Schedule the task concurrently.
            tasks.append(asyncio.create_task(
                process_game_state(chosen_model, desired_entry_id, game_state_text, results)
            ))
            count += 1

            # If we have reached the batch size, run the tasks concurrently and checkpoint.
            if len(tasks) >= batch_size:
                await asyncio.gather(*tasks)
                tasks.clear()
                with open("results.json", "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                print(f"Checkpoint: Saved {count} responses to 'results.json'.")

    # Await any remaining tasks.
    if tasks:
        await asyncio.gather(*tasks)

    # Save all results to a single JSON file after processing.
    with open("results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Saved all results to 'all_results.json'.")

    # Print summary of processed and failed entries.
    print(f"\nNumber of entries processed: {count - len(failed_entries)}")
    print(f"Number of failed entries: {len(failed_entries)}")

# Run the main function in Jupyter Notebook or Python file. Code originally written for a Jupyter Server instance.
await main()

  
