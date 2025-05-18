import argparse
import os
import json
# Assume these modules exist and have necessary functions
import config_manager
import novel_generator.architecture
import novel_generator.blueprint

def main():
    parser = argparse.ArgumentParser(description="AI小说生成器命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # generate-architecture command
    parser_architecture = subparsers.add_parser("generate-architecture", help="生成小说整体架构")
    parser_architecture.add_argument("--config", required=True, help="配置文件的路径")
    parser_architecture.add_argument("--output", required=True, help="保存架构文件的目录")
    parser_architecture.set_defaults(func=generate_architecture_command)

    # generate-blueprint command
    parser_blueprint = subparsers.add_parser("generate-blueprint", help="生成章节蓝图")
    parser_blueprint.add_argument("--config", required=True, help="配置文件的路径")
    parser_blueprint.add_argument("--architecture", required=True, help="架构文件的路径")
    parser_blueprint.add_argument("--chapter", required=True, help="章节号、范围（例如，1-5）或 'all'")
    parser_blueprint.add_argument("--output", required=True, help="保存蓝图文件的目录")
    parser_blueprint.set_defaults(func=generate_blueprint_command)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

def generate_architecture_command(args):
    print(f"Generating architecture with config: {args.config}")
    try:
        # Load configuration
        config = config_manager.load_config(args.config) # Assuming load_config exists

        # Ensure output directory exists
        os.makedirs(args.output, exist_ok=True)

        # Call architecture generation function
        # Assuming novel_generator.architecture has a function like generate_architecture
        architecture_data = novel_generator.architecture.generate_architecture(config)

        # Save architecture data
        output_path = os.path.join(args.output, "architecture.json") # Assuming JSON output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(architecture_data, f, indent=4, ensure_ascii=False)

        print(f"Architecture saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
    except Exception as e:
        print(f"An error occurred during architecture generation: {e}")


def generate_blueprint_command(args):
    print(f"Generating blueprints for chapter(s) {args.chapter} with config: {args.config} and architecture: {args.architecture}")
    try:
        # Load configuration
        config = config_manager.load_config(args.config) # Assuming load_config exists

        # Read architecture file
        with open(args.architecture, "r", encoding="utf-8") as f:
            architecture_data = json.load(f) # Assuming JSON architecture file

        # Parse chapter input
        chapters_to_generate = parse_chapter_input(args.chapter, architecture_data)

        if not chapters_to_generate:
            print(f"No valid chapters found for input: {args.chapter}")
            return

        # Ensure output directory exists
        os.makedirs(args.output, exist_ok=True)

        # Iterate and generate blueprints for specified chapters
        for chapter_num in chapters_to_generate:
            print(f"Generating blueprint for chapter {chapter_num}...")
            # Assuming novel_generator.blueprint has a function like generate_blueprint
            # This function might need the config, architecture data, and the specific chapter number/data
            blueprint_data = novel_generator.blueprint.generate_blueprint(config, architecture_data, chapter_num)

            # Save blueprint data
            output_path = os.path.join(args.output, f"blueprint_chapter_{chapter_num}.json") # Assuming JSON output
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(blueprint_data, f, indent=4, ensure_ascii=False)

            print(f"Blueprint for chapter {chapter_num} saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found ({args.config} or {args.architecture})")
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from architecture file {args.architecture}")
    except Exception as e:
        print(f"An error occurred during blueprint generation: {e}")

def parse_chapter_input(chapter_input, architecture_data):
    """Parses the chapter input string ('all', 'number', or 'range')"""
    chapters = []
    if chapter_input.lower() == "all":
        # Assuming architecture_data has a structure that allows iterating through chapters
        # For example, if architecture_data is a list of chapters or has a 'chapters' key
        # This part needs to be adapted based on the actual architecture_data structure
        print("Parsing 'all' chapters - Note: This requires knowing the structure of architecture_data.")
        # Placeholder: In a real scenario, you'd extract chapter numbers from architecture_data
        # For now, let's assume architecture_data is a dict with a 'chapters' list,
        # and each item in the list has a 'chapter_number' key.
        if isinstance(architecture_data, dict) and 'chapters' in architecture_data:
             chapters = [c.get('chapter_number') for c in architecture_data['chapters'] if isinstance(c, dict) and 'chapter_number' in c]
             chapters = [c for c in chapters if isinstance(c, (int, str))] # Filter out None or invalid types
        else:
             print("Warning: Could not determine chapters from architecture_data structure.")
             # Fallback or error handling if architecture structure is unknown
             pass # Or raise an error

    elif "-" in chapter_input:
        try:
            start, end = map(int, chapter_input.split("-"))
            chapters = list(range(start, end + 1))
        except ValueError:
            print(f"Invalid chapter range format: {chapter_input}. Use format 'start-end'.")
    else:
        try:
            chapters = [int(chapter_input)]
        except ValueError:
            print(f"Invalid chapter number format: {chapter_input}. Use a number, range (e.g., 1-5), or 'all'.")

    # Filter chapters to ensure they exist in the architecture data if possible
    # This filtering step depends heavily on the structure of architecture_data
    # For now, we'll return the parsed list, assuming the generation functions handle non-existent chapters
    return chapters


if __name__ == "__main__":
    main()