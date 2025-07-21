import subprocess
import time
import select
from datasets import load_dataset

def launch_agent_process():
    """Launches the C++ agent as a subprocess."""
    try:
        print("Attempting to launch NeuroGen executable...")
        process = subprocess.Popen(
            ['./NeuroGen'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        print(f"Agent process launched with PID: {process.pid}")
        time.sleep(5)  # Give the agent time to initialize
        
        # Check if the process terminated unexpectedly right after launch
        if process.poll() is not None:
            print("Error: Agent process terminated unexpectedly after launch.")
            stdout, stderr = process.communicate()
            print(f"Agent stdout:\n{stdout}")
            print(f"Agent stderr:\n{stderr}")
            return None
            
        print("Agent process appears to be running.")
        return process
    except FileNotFoundError:
        print("Error: The 'NeuroGen' executable was not found.")
        print("Please ensure the C++ agent is compiled and in the correct directory.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during agent launch: {e}")
        return None

def send_command(process, command, expect_response=True, timeout=2.0):
    """Send a command to the agent and optionally read a response."""
    if process.poll() is not None:
        print("Agent process has terminated unexpectedly.")
        stdout, stderr = process.communicate()
        print(f"Agent stdout:\n{stdout}")
        print(f"Agent stderr:\n{stderr}")
        return "AGENT_TERMINATED"

    if not command.endswith("\n"):
        command += "\n"

    print(f"Sending command: {command.strip()}")
    process.stdin.write(command)
    process.stdin.flush()

    if not expect_response:
        return ""

    output_lines = []
    start_time = time.time()
    while time.time() - start_time < timeout:
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if ready:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                print(f"Agent raw output: {line}")
                output_lines.append(line)
                if line.startswith("NEXT_WORD_PREDICTION"):
                    break
        else:
            if process.poll() is not None:
                break

    return "\n".join(output_lines)

def train_agent(process, dataset):
    """Trains the agent on the provided dataset."""
    print("\n--- Starting Training Phase ---")
    # Ensure agent is in language training mode
    send_command(process, "SET_MODE:LANGUAGE_TRAINING")

    for i, item in enumerate(dataset):
        text = item['text']
        if not text.strip():
            print(f"Skipping empty text for item {i}.")
            continue

        # Truncate text to a manageable size
        max_length = 512
        truncated_text = text[:max_length].replace("\n", " ") # Avoid newlines in the middle of the command

        command = f"LANGUAGE_INPUT:{truncated_text}"
        send_command(process, command, expect_response=False)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} training examples.")

    print("--- Training Phase Complete ---")

def validate_agent(process):
    """Validates the agent's learning with a few prompts."""
    print("\n--- Starting Validation Phase ---")
    prompts = [
        "generate_text What is the capital of France?",
        "generate_text Explain the theory of relativity in simple terms.",
        "generate_text Write a short story about a robot who discovers music."
    ]

    for prompt in prompts:
        command = f"LANGUAGE_INPUT:{prompt}"
        print(f"\nValidation Prompt: {prompt}")
        output = send_command(process, command)
        print(f"Agent Response: {output}")

    print("--- Validation Phase Complete ---")

def main():
    """Main function to run the training and validation pipeline."""
    print("--- Launching NeuroGen Agent for Training ---")
    agent_process = launch_agent_process()
    if not agent_process:
        print("Agent launch failed. Exiting.")
        return

    # Load the SlimPajama dataset
    print("\n--- Loading SlimPajama Dataset ---")
    try:
        slimpajama_dataset = load_dataset("cerebras/slimpajama-627b", name="default", split='train', streaming=True)
        training_data = slimpajama_dataset.take(1000)  # Use a subset for demonstration
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        agent_process.terminate()
        return

    # Train the agent
    print("Proceeding to training...")
    train_agent(agent_process, training_data)

    # Validate the agent
    print("Proceeding to validation...")
    validate_agent(agent_process)

    print("\n--- Interactive Session ---")
    print("Type 'quit' to exit and shut down the agent.\n")
    try:
        while True:
            user_input = input("You> ")
            if user_input.strip().lower() in {"quit", "exit"}:
                break
            response = send_command(agent_process, f"LANGUAGE_INPUT:{user_input}")
            print(f"Agent> {response}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Shutdown the agent
    print("\n--- Shutting Down Agent ---")
    send_command(agent_process, "shutdown", expect_response=False)
    try:
        agent_process.wait(timeout=10)
        print("Agent process terminated gracefully.")
    except subprocess.TimeoutExpired:
        print("Agent process did not terminate within 10 seconds. Forcing shutdown.")
        agent_process.kill()
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()
