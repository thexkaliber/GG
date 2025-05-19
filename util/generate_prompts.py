import json
import csv
import random
import os
import re
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Select random sample of data from the JerichoWorld Data for formatting into prompts for generating synthetic consequences. 
def select_random_game_states(input_file, output_file, num_samples=10000, max_game=26, max_state=1100):
    """
    Randomly select game states from specific ranges of game and state numbers.
    Creates a new JSON file with empty fields for consequence and reasoning.

    Args:
        input_file (str): Path to the input JSON file containing game states
        output_file (str): Path to save the output JSON file
        num_samples (int): Number of game states to randomly select
        max_game (int): Maximum game number to consider (0 to max_game, inclusive)
        max_state (int): Maximum state number to consider (0 to max_state, inclusive)
    """
    # Load the JSON data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {input_file}: {e}")
        return
    except Exception as e:
        logger.error(f"Error loading JSON file {input_file}: {e}")
        return

    # Check if the data is a list of game states or a dictionary with game states
    if isinstance(data, list):
        game_states = data
    elif isinstance(data, dict) and any(isinstance(data.get(key), dict) for key in data):
        # Assume it's a dictionary with game state objects
        game_states = list(data.values())
    else:
        logger.error("Unexpected JSON format. Expected a list of game states or a dictionary with game states.")
        return

    # Create a dictionary for quick lookup of game states by game and state numbers
    state_lookup = {}
    id_pattern = re.compile(r'game_(\d+)_state_(\d+)')

    valid_states_count = 0
    invalid_states_count = 0

    # First pass: build lookup table
    for state in game_states:
        try:
            if not isinstance(state, dict) or 'id' not in state:
                invalid_states_count += 1
                continue

            state_id = state['id']
            match = id_pattern.match(state_id)

            if not match:
                logger.debug(f"Skipping state with invalid ID format: {state_id}")
                invalid_states_count += 1
                continue

            game_num = int(match.group(1))
            state_num = int(match.group(2))

            # Skip if outside our ranges
            if game_num < 0 or game_num > max_game or state_num < 0 or state_num > max_state:
                logger.debug(f"Skipping state outside specified ranges: {state_id}")
                invalid_states_count += 1
                continue

            # Check if context field exists
            if 'context' not in state:
                logger.debug(f"Skipping state missing context field: {state_id}")
                invalid_states_count += 1
                continue

            # Add to lookup dictionary
            if game_num not in state_lookup:
                state_lookup[game_num] = {}

            state_lookup[game_num][state_num] = state
            valid_states_count += 1

        except Exception as e:
            logger.debug(f"Error processing state: {e}")
            invalid_states_count += 1
            continue

    logger.info(f"Found {valid_states_count} valid states within specified ranges")
    logger.info(f"Found {invalid_states_count} invalid states or states outside specified ranges")

    # Handle case where no valid states were found
    if not state_lookup:
        logger.error("No valid game states found within the specified ranges.")
        return

    # Random selection strategy
    selected_states = []
    attempts = 0
    max_attempts = num_samples * 10  # Limit attempts to avoid infinite loops

    while len(selected_states) < num_samples and attempts < max_attempts:
        attempts += 1

        # Randomly choose a game number and state number
        game_num = random.randint(0, max_game)
        state_num = random.randint(0, max_state)

        # Skip if this game doesn't exist in our lookup
        if game_num not in state_lookup:
            continue

        # Skip if this state doesn't exist in the selected game
        if state_num not in state_lookup[game_num]:
            continue

        # Get the state
        state = state_lookup[game_num][state_num]

        # Skip if we've already selected this state
        if state in selected_states:
            continue

        # Add the state to our selection
        selected_states.append(state)

        # Log progress periodically
        if len(selected_states) % 1000 == 0:
            logger.info(f"Selected {len(selected_states)} states so far...")

    # Check if we reached the target number of samples
    if len(selected_states) < num_samples:
        logger.warning(f"Could only find {len(selected_states)} unique states within the specified ranges")

    # Ensure consequence and reasoning fields are empty
    for state in selected_states:
        state['consequence'] = ""
        state['reasoning'] = ""

    # Save the selected states to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_states, f, indent=2, ensure_ascii=False)

        # Count unique games in final selection
        unique_games = set(id_pattern.match(state['id']).group(1) for state in selected_states if id_pattern.match(state['id']))
        unique_states = set(id_pattern.match(state['id']).group(2) for state in selected_states if id_pattern.match(state['id']))

        logger.info(f"Successfully created {output_file} with {len(selected_states)} randomly selected game states.")
        logger.info(f"States were selected from {len(unique_games)} different games.")
        logger.info(f"Selected state numbers range from {min(int(s) for s in unique_states)} to {max(int(s) for s in unique_states)}.")

    except Exception as e:
        logger.error(f"Error saving output file: {e}")

def format_game_state(json_data):
    """
    Format game state JSON data into a structured text representation with additional header information.

    Args:
        json_data: Either a JSON string or a dictionary containing the game state data.

    Returns:
        A formatted string representation of the game state.
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    output = []

    # Add Action section
    output.append("### Action:")
    action = data.get("action", "None")
    output.append(action)
    output.append("")  # Empty line for spacing

    # Add Context section
    output.append("### Context:")

    # Add Location information
    output.append("Location:")
    context = data.get("context", {})
    location_name = context.get("location_name", "Unknown")
    location_desc = context.get("location_desc", "No description available")

    output.append(f"-Name: {location_name}")
    output.append(f"-Description: {location_desc}")
    output.append("")  # Empty line for spacing

    # Add Surroundings section
    output.append("Surroundings:")
    surroundings = context.get("surroundings", {})

    if "surrounding_objects" in surroundings:
        surrounding_objects = surroundings["surrounding_objects"]
        output.append("-Surrounding Objects:")

        # If surrounding_objects is a list, iterate directly
        if isinstance(surrounding_objects, list):
            for obj in surrounding_objects:
                name = obj.get("name", "Unnamed")
                description = obj.get("description", "No description available")
                attributes = obj.get("attributes", [])
                if not attributes:
                    attributes = [name]
                output.append(f"--Name: {name}")
                output.append(f"--Description: {description}")
                output.append(f"--Attributes: {', '.join(attributes)}")
                output.append("")  # Empty line for spacing
        # Otherwise, assume it's a dictionary with keys for names, descriptions, etc.
        elif isinstance(surrounding_objects, dict):
            object_keys = surrounding_objects.get("name", {}).keys()
            for obj_key in object_keys:
                name = surrounding_objects.get("name", {}).get(obj_key, "Unnamed")
                description = surrounding_objects.get("description", {}).get(obj_key, "No description available")
                attributes = surrounding_objects.get("attributes", {}).get(obj_key, [])
                if not attributes:
                    attributes = [name]
                output.append(f"--Name: {name}")
                output.append(f"--Description: {description}")
                output.append(f"--Attributes: {', '.join(attributes)}")
                output.append("")  # Empty line for spacing
        else:
            output.append("Invalid structure for surrounding objects.")
            output.append("")
    else:
        output.append("No surrounding objects found.")
        output.append("")

    # Add Inventory section
    output.append("Inventory:")
    if "inventory" in context:
        inventory = context["inventory"]
        inventory_desc = inventory.get("desc", "You are not carrying anything.")
        output.append(f"-Description: {inventory_desc}")
        output.append("")

        if "objects" in inventory:
            inventory_objects = inventory["objects"]
            output.append("-Inventory Objects:")
            # If inventory_objects is a list, iterate over it
            if isinstance(inventory_objects, list):
                for obj in inventory_objects:
                    name = obj.get("name", "Unnamed")
                    description = obj.get("description", "No description available")
                    attributes = obj.get("attributes", [])
                    if not attributes:
                        attributes = [name]
                    output.append(f"--Name: {name}")
                    output.append(f"--Description: {description}")
                    output.append(f"--Attributes: {', '.join(attributes)}")
                    output.append("")
            # Otherwise, if it's a dictionary, process it accordingly.
            elif isinstance(inventory_objects, dict):
                for obj_key, obj_data in inventory_objects.items():
                    name = obj_key  # The key is the name
                    description = obj_data.get("desc", "No description available")
                    attributes = obj_data.get("attribute", [])
                    output.append(f"--Name: {description}")
                    output.append(f"--Description: {name}")
                    output.append(f"--Attributes: {', '.join(attributes)}")
                    output.append("")
            else:
                output.append("Invalid structure for inventory objects.")
                output.append("")
    else:
        output.append("No inventory information available.")
        output.append("")

    # Add Observation
    observation = context.get("observation") or context.get("location_desc", "")
    output.append(f"Observation: {observation}")

    # Add Valid Actions
    valid_actions = context.get("valid_act", [])
    if valid_actions:
        output.append(f"Valid Actions: {', '.join(valid_actions)}")
    else:
        output.append("No valid actions available.")

    return "\n".join(output)


def main():
  # Set default filenames
  input_filename = "INPUT.json"
  output_filename = "SELECTED_STATES.json"
  num_samples = 10000

  # Check if input file exists
  if not os.path.exists(input_filename):
      logger.error(f"Error: Input file '{input_filename}' not found.")
      sys.exit(1)
  
  select_random_game_states(input_filename, output_filename, num_samples)

  input_filename = "SELECTED_STATES.json"
  output_filename = "FORMATTED_STATES.csv"

  try:
      # Read the JSON file (expected to be a list of game state objects or a single object)
      with open(input_filename, 'r', encoding='utf-8') as json_file:
          data = json.load(json_file)

      # If the JSON file contains a single entry (a dictionary), wrap it in a list
      if not isinstance(data, list):
          data = [data]

      # Write each formatted game state to the CSV file.
      # Using csv.QUOTE_ALL to ensure newline characters are preserved in each cell.
      with open(output_filename, 'w', newline='', encoding='utf-8') as csv_file:
          writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
          writer.writerow(["id","formatted_text"])  # Optional header

          for game_state in data:
              formatted_text = format_game_state(game_state)
              state_id = game_state.get("id", "")
              writer.writerow([state_id, formatted_text])

      print(f"Successfully processed {len(data)} game state entries and saved to '{csv_file_path}'.")

  except Exception as e:
      print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
