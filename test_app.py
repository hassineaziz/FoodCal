
import pytest
from app import app

# This 'fixture' runs setup before each test
@pytest.fixture
def client():
    # set app config to testing mode
    app.config['TESTING'] = True
    
    # make a test client we can use
    with app.test_client() as client:
        yield client

# Test 1: Make sure the home page works
def test_home_page_loads(client):
    """
    Test that the home page loads fine (Status Code 200).
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.data  # check if we actually got html back

# Test 2: Check the Cook/Suggest Page
def test_cook_page_loads(client):
    """
    Verify the cook page loads.
    """
    response = client.get('/cook')
    assert response.status_code == 200
    # Look for keywords to confirm it's the right page
    assert b"Suggest" in response.data or b"Ingredients" in response.data

# Test 3: Test out the text search
def test_text_search_apple_pie(client):
    """
    Try searching for 'apple pie' and see if it works.
    Outcome: Should get a 200 OK and see nutrition info.
    """
    # pretend to submit the form
    response = client.post('/predict', data={'food_name': 'apple pie'})
    
    assert response.status_code == 200
    # The response should contain the formatted name "Apple Pie"
    assert b"Apple Pie" in response.data
    # It should show nutrition info (Calories)
    assert b"Calories" in response.data

# Test 4: See if Recipe Suggestion does anything
def test_suggest_recipe(client):
    """
    Test recipe suggestions with basic ingredients.
    """
    # send some ingredients over
    response = client.post('/suggest', data={'ingredients': 'chicken'})
    
    assert response.status_code == 200
    # Should show matches found
    assert b"We found" in response.data or b"Match" in response.data
