from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import requests
from io import BytesIO
import torch
import csv
import os

app = Flask(__name__)

# Where to load the fine-tuned model from.
# Accepts a local folder OR a Hugging Face Hub repo id (e.g. "username/dinov2-food101"),
# since transformers' from_pretrained() handles both transparently.
# The weights are ~347 MB and are NOT committed to git — see the README "Model" section.
MODEL_PATH = os.environ.get("MODEL_PATH", "./my_final_dinov2_food101_model_FULL")

# Load things up once so it's faster
print(f"Loading model and processor from '{MODEL_PATH}'...")
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model.eval()  # switch to eval mode
print("Model and processor loaded successfully!")

# Start loading the nutrition database
nutrition_data = {}
print("Loading nutrition data...")
try:
    with open('nutrition.csv', mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row['label']
            if label not in nutrition_data:
                nutrition_data[label] = []
            nutrition_data[label].append(row)
    print("Nutrition data loaded successfully!")
except Exception as e:
    print(f"Error loading nutrition data: {e}")

# Load ingredients data
ingredients_data = {}
all_recipes = []
print("Loading ingredients data...")
try:
    import json
    with open('ing_with_dish_jsn.json', 'r') as f:
        raw_ingredients = json.load(f)
        # Organize the data so it's easier to access (label -> ingredients)
        for key, value in raw_ingredients.items():
            ingredients_list = value[0]
            label = value[1]
            
            # Keep track of everything for the search feature
            all_recipes.append({'label': label, 'ingredients': ingredients_list})
            
            # Store the first list we find for each label
            if label not in ingredients_data:
                ingredients_data[label] = ingredients_list
    print(f"Ingredients data loaded successfully! Found ingredients for {len(ingredients_data)} foods (Total recipes: {len(all_recipes)}).")
except Exception as e:
    print(f"Error loading ingredients data: {e}")

def get_nutrition_info(label):
    """Helper to get nutrition info for a specific food label."""
    if label not in nutrition_data:
        return None
    
    options = nutrition_data[label]
    # We ideally want the 100g option (or whatever is closest)
    best_option = None
    min_diff = float('inf')
    
    for option in options:
        try:
            weight = float(option['weight'])
            diff = abs(weight - 100)
            if diff < min_diff:
                min_diff = diff
                best_option = option
        except ValueError:
            continue
            
    return best_option if best_option else (options[0] if options else None)

@app.route('/')
def index():
    return render_template('index.html', prediction=None, nutrition=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Main function to handle predictions (text, upload, or URL)"""
    image = None
    food_name_query = request.form.get('food_name', '').strip()
    
    # CASE 1: User typed something in the search box
    if food_name_query:
        # cleanup the input string
        query = food_name_query.lower().replace(' ', '_')
        
        # pretty basic search logic
        best_match = None
        
        # 1. Exact match
        if query in nutrition_data:
            best_match = query
        else:
            # 2. Contains match
            for label in nutrition_data.keys():
                if query in label:
                    best_match = label
                    break
        
        if best_match:
            formatted_label = best_match.replace('_', ' ').title()
            nutrition_info = get_nutrition_info(best_match)
            ingredients_list = ingredients_data.get(best_match, [])
            return render_template('index.html', 
                                 prediction=formatted_label, 
                                 nutrition=nutrition_info,
                                 ingredients=ingredients_list)
        else:
            return render_template('index.html', 
                                 prediction=f"Could not find nutrition info for '{food_name_query}'",
                                 nutrition=None)

    # CASE 2: Image Upload
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            try:
                image = Image.open(file.stream)
                image = image.convert('RGB')  # Make sure it's RGB so it doesn't crash
            except Exception as e:
                return render_template('index.html', 
                                     prediction=f"Error processing uploaded image: {str(e)}",
                                     nutrition=None)
    
    # CASE 3: Image URL
    if image is None:
        url = request.form.get('url', '').strip()
        if url:
            try:
                # Need a User-Agent so websites don't block us
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, timeout=10, headers=headers)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                image = image.convert('RGB')  # Ensure RGB format
            except Exception as e:
                return render_template('index.html', 
                                     prediction=f"Error downloading image from URL: {str(e)}",
                                     nutrition=None)
    
    # If no input provided at all
    if image is None:
        return render_template('index.html', 
                             prediction="Please provide an image or food name.",
                             nutrition=None)
    
    # Run prediction
    try:
        # Prepare the image for the model
        inputs = processor(image, return_tensors="pt")
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
        
        # Figure out what the label actually is
        id2label = model.config.id2label
        predicted_label = id2label[predicted_class_idx]
        
        # Clean up the label name (remove _ and capitalization)
        formatted_label = predicted_label.replace('_', ' ').title()
        
        # Get nutrition info
        nutrition_info = get_nutrition_info(predicted_label)
        ingredients_list = ingredients_data.get(predicted_label, [])
        
        return render_template('index.html', 
                             prediction=formatted_label, 
                             nutrition=nutrition_info,
                             ingredients=ingredients_list)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction=f"Error during prediction: {str(e)}",
                             nutrition=None)

@app.route('/cook')
def cook():
    return render_template('cook.html', suggestions=None, user_ingredients="")

@app.route('/suggest', methods=['POST'])
def suggest():
    """Find recipes that match what the user has in their fridge"""
    user_input = request.form.get('ingredients', '').lower()
    if not user_input:
        return render_template('cook.html', suggestions=None, user_ingredients="")
    
    # Parse nutrition limits
    filters_active = False
    try:
        max_cal = float(request.form.get('max_calories') or float('inf'))
        min_prot = float(request.form.get('min_protein') or 0)
        max_carbs = float(request.form.get('max_carbs') or float('inf'))
        max_fat = float(request.form.get('max_fat') or float('inf'))
        
        # Check if any filter is actually set to a non-default value
        if (max_cal != float('inf') or min_prot != 0 or 
            max_carbs != float('inf') or max_fat != float('inf')):
            filters_active = True
            
    except ValueError:
        max_cal, min_prot, max_carbs, max_fat = float('inf'), 0, float('inf'), float('inf')

    # If neither ingredients nor filters are provided, return empty
    if not user_input and not filters_active:
        return render_template('cook.html', suggestions=None, user_ingredients="")

    # Clean up the user's input list
    # Split by comma/newlines and remove empty spaces
    user_ingredients = set(filter(None, [i.strip() for i in user_input.replace('\n', ',').split(',')]))

    suggestions = []
    
    # Use a dictionary to avoid duplicates (same label with similar ingredients)
    # We want to show the BEST match for each unique dish label
    best_matches = {}
    
    for recipe in all_recipes:
        label = recipe['label']
        
        # Check macros first to save time (optimization)
        nutrition = get_nutrition_info(label)
        if not nutrition:
            continue
            
        try:
            cal = float(nutrition['calories'])
            prot = float(nutrition['protein'])
            carbs = float(nutrition['carbohydrates'])
            fat = float(nutrition['fats'])
            
            if (cal > max_cal or 
                prot < min_prot or 
                carbs > max_carbs or 
                fat > max_fat):
                continue
        except (ValueError, KeyError):
            continue

        dish_ingredients = set(recipe['ingredients'])
        
        # Special case: if no ingredients entered but filters are on, apply filters only
        if not user_ingredients:
            percentage = 1.0 # Treat as full match for sorting purposes if we are only filtering by nutrition
            missing = list(dish_ingredients) # Everything is missing
        else:
            if not dish_ingredients:
                continue
                
            # Calculate intersection
            match_count = 0
            missing = []
            
            for ing in dish_ingredients:
                found = False
                for user_ing in user_ingredients:
                    if user_ing in ing or ing in user_ing:
                        found = True
                        break
                
                if found:
                    match_count += 1
                else:
                    missing.append(ing)
            
            percentage = match_count / len(dish_ingredients)
        
        # Logic for showing matches:
        # 1. If ingredients provided, show anything with > 0% match
        # 2. If ONLY filters provided, show everything that fits criteria
        if percentage > 0 or (not user_ingredients and filters_active):
            formatted_label = label.replace('_', ' ').title()
            
            match_data = {
                'name': formatted_label,
                'match_percentage': int(percentage * 100) if user_ingredients else 100,
                'missing': missing,
                'nutrition': nutrition
            }
            
            # Only keep the best version of this recipe if we see it multiple times
            # For nutrition search, first one is fine
            if label not in best_matches or (user_ingredients and percentage > best_matches[label]['raw_percentage']):
                match_data['raw_percentage'] = percentage
                best_matches[label] = match_data
    
    # Convert to list and sort by percentage (descending)
    suggestions = sorted(best_matches.values(), key=lambda x: x['match_percentage'], reverse=True)[:5]
    
    return render_template('cook.html', 
                         suggestions=suggestions, 
                         user_ingredients=request.form.get('ingredients', ''))

if __name__ == '__main__':
    app.run(debug=True, port=5001)

