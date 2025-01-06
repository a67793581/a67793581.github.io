import re
import argparse

def generate_toc(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    toc = []
    for line in content:
        match = re.match(r'^(#+)\s+(.*)$', line)
        if match:
            level, title = match.groups()
            level = len(level)
            anchor = title.lower().replace(' ', '-')
            toc.append(f"{'  ' * (level - 1)}* [{title}](#{anchor})")

    toc_content = "\n".join(toc)
    return toc_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Table of Contents for a Markdown file.")
    parser.add_argument("file_path", type=str, help="Path to the Markdown file")
    args = parser.parse_args()

    toc_content = generate_toc(args.file_path)
    print(toc_content)
