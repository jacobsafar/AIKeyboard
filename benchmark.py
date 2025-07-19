#!/usr/bin/env python3
"""
Benchmark helper for testing keyboard predictor accuracy
"""

from keyboard_predictor import KeyboardPredictor

def benchmark(seq_list, predictor=None):
    """
    Benchmark the keyboard predictor with a list of (sequence, expected_word) pairs
    
    Args:
        seq_list: List of tuples (button_sequence, expected_word)
        predictor: KeyboardPredictor instance (creates new if None)
    
    Returns:
        float: Top-1 accuracy rate (0.0 to 1.0)
    """
    if predictor is None:
        predictor = KeyboardPredictor()
    
    hits = 0
    total = len(seq_list)
    
    print(f"Running benchmark on {total} test cases...")
    print("-" * 50)
    
    for i, (seq, gold) in enumerate(seq_list):
        try:
            result = predictor.predict_word(seq)
            pred = result["top_predictions"][:1]
            hit = pred and pred[0] == gold.upper()
            hits += hit
            
            status = "✓" if hit else "✗"
            pred_word = pred[0] if pred else "NO_PRED"
            print(f"{status} Test {i+1}: {seq} -> Expected: {gold.upper()}, Got: {pred_word}")
            
        except Exception as e:
            print(f"✗ Test {i+1}: {seq} -> Error: {str(e)}")
    
    accuracy = hits / total
    print("-" * 50)
    print(f"Top-1 accuracy: {accuracy:.3f} ({hits}/{total})")
    return accuracy

if __name__ == "__main__":
    # Sample test cases
    test_cases = [
        ([2, 4, 1, 2, 1], "THERE"),
        ([1, 1], "EL"), 
        ([6, 3, 6], "MAN"),
        ([2, 1, 2, 1, 3], "CZECH"),
        ([2, 4, 1], "THE"),
        ([3, 6, 3], "AND"),
        ([4, 1], "HE"),
        ([5, 2], "IS"),
        ([2, 4], "TO"),
        ([5, 6], "IN")
    ]
    
    # Run benchmark
    kp = KeyboardPredictor()
    accuracy = benchmark(test_cases, kp)
    
    print(f"\nFinal accuracy: {accuracy:.1%}")