def is_palindrome(s: str) -> bool:
    """
    Checks if a given string is a palindrome.
    Ignores non-alphanumeric characters and case sensitivity.
    """
    cleaned = [ch.lower() for ch in s if ch.isalnum()]
    return cleaned == cleaned[::-1]


def run_tests():
    test_cases = [
        ("racecar", True),
        ("A man, a plan, a canal: Panama", True),
        ("hello", False),
        ("No 'x' in Nixon", True),
        ("12321", True),
        ("12345", False),
        ("", True),
    ]

    for s, expected in test_cases:
        result = is_palindrome(s)
        assert result == expected, f"Failed for '{s}': expected {expected}, got {result}"
        print(f"PASS: is_palindrome('{s}') == {result}")

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    run_tests()
