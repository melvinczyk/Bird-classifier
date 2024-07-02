def sort_and_remove_duplicates(input_file, output_file):
    unique_lines = set()
    with open(input_file, 'r') as file:
        for line in file:
            unique_lines.add(line.strip())

    sorted_unique_lines = sorted(unique_lines)

    with open(output_file, 'w') as file:
        for line in sorted_unique_lines:
            file.write(line + '\n')


sort_and_remove_duplicates("popular_birds.txt", "popular_birds.txt")