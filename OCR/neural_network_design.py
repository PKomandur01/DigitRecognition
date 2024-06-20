# Define a function to test the performance of the neural network
def test(data_matrix, data_labels, test_indices, nn):
    avg_sum = 0
    for _ in range(100):  # Use range() instead of xrange() in Python 3
        correct_guess_count = 0
        for i in test_indices:
            test_data = data_matrix[i]
            prediction = nn.predict(test_data)
            if data_labels[i] == prediction:
                correct_guess_count += 1

        avg_sum += (correct_guess_count / float(len(test_indices)))
    
    return avg_sum / 100

# Perform tests with different numbers of hidden nodes
for i in range(5, 50, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    performance = test(data_matrix, data_labels, test_indices, nn)
    print(f"{i} Hidden Nodes: {performance}")
