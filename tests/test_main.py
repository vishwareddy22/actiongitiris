from main import perform_prediction




def test_prediction():
    result = perform_prediction(5.1, 3.5, 1.4, 0.2)


    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
