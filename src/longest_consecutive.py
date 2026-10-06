def longest_consecutive(nums: list[int]) -> int:
    """
    Finds the length of the longest consecutive elements sequence in O(n) time.
    """
    num_set = set(nums)
    longest_streak = 0

    for num in num_set:
        # Only start counting if 'num' is the beginning of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)

    return longest_streak


def run_tests():
    test_cases = [
        # Test Case 1: Standard unsorted array with multiple sequences
        ([100, 4, 200, 1, 3, 2], 4),  # Sequence: [1, 2, 3, 4]
        
        # Test Case 2: Array with duplicate numbers and negative values
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9),  # Sequence: [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        # Test Case 3: Empty list and single element list edge cases
        ([], 0),
        ([10], 1),
    ]

    for nums, expected in test_cases:
        result = longest_consecutive(nums)
        assert result == expected, f"Failed for {nums}: expected {expected}, got {result}"
        print(f"PASS: longest_consecutive({nums}) == {result}")

    print("\nAll test cases passed successfully!")


if __name__ == "__main__":
    run_tests()
