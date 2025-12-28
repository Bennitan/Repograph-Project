def find_duplicates(items):
    # This is O(n^2) - Extremely Slow!
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                already_found = False
                for k in duplicates:
                    if k == items[i]:
                        already_found = True
                if not already_found:
                    duplicates.append(items[i])
    return duplicates